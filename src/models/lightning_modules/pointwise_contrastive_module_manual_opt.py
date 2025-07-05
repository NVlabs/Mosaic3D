from typing import Any, Dict, List, Optional, Callable, Union, Literal
from jaxtyping import Float

import os
import time
import random
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import MaxMetric

from lightning.pytorch.core.module import Optimizer, LightningOptimizer

from src.models.lightning_modules.pointwise_contrastive_module import (
    PointwiseContrastiveLanguageLitModule,
)
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class PointwiseContrastiveLanguageLitModuleManualOpt(PointwiseContrastiveLanguageLitModule):
    """
    Extension of PointwiseContrastiveLanguageLitModule with manual optimization
    and synchronized error handling for multi-GPU training.

    This module disables automatic optimization and manually handles the training loop
    to provide robust error recovery for:
    - Out-of-Memory (OOM) errors during forward or backward pass
    - Invalid loss values (NaN or Inf)

    All error handling is synchronized across GPUs to maintain model consistency.

    Scheduler Support:
    - Step-based schedulers: Updated after each optimizer step when scheduler_interval="step"
    - Epoch-based schedulers: Updated at the end of each epoch when scheduler_interval="epoch"

    The scheduler_interval is configured in the scheduler config file (e.g., onecyclelr.yaml, step.yaml)
    """

    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        compile: bool,
        loss_cfg: Dict,
        best_metric: str,
        eval_cfg: Optional[Dict] = None,
        use_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            compile=compile,
            loss_cfg=loss_cfg,
            best_metric=best_metric,
            eval_cfg=eval_cfg,
            use_prompt=use_prompt,
            **kwargs,
        )

        # Disable automatic optimization
        self.automatic_optimization = False

        # Invalid batch handling (OOM or invalid loss)
        self.invalid_batch_count = 0
        self.max_invalid_batches = loss_cfg.get("max_invalid_batches", 10)

        # Initialize CPU-only process group for OOM sync fallback
        self.cpu_sync_group = None

    def _initialize_cpu_sync_group(self):
        """Initialize CPU-only process group for OOM sync fallback."""
        if self.cpu_sync_group is None and torch.distributed.is_initialized():
            try:
                # Create CPU-only process group with GLOO backend
                world_size = torch.distributed.get_world_size()
                ranks = list(range(world_size))
                self.cpu_sync_group = torch.distributed.new_group(ranks=ranks, backend="gloo")
                log.info(
                    f"Rank {self.trainer.global_rank}: Created CPU sync group with GLOO backend"
                )
            except Exception as e:
                log.warning(
                    f"Rank {self.trainer.global_rank}: Failed to create CPU sync group: {e}"
                )
                self.cpu_sync_group = None

    def _sync_validity_status_cpu(self, local_invalid: bool) -> bool:
        """
        CPU-only fallback synchronization for OOM cases.
        Uses GLOO backend with CPU tensors only.
        """
        if not torch.distributed.is_initialized():
            return local_invalid

        assert self.cpu_sync_group is not None, "CPU sync group not available"

        # Create tensor on CPU only - never move to GPU
        invalid_tensor = torch.tensor(
            1.0 if local_invalid else 0.0, dtype=torch.float32, device="cpu"
        )

        # Use CPU-only sync with GLOO backend
        torch.distributed.all_reduce(
            invalid_tensor, op=torch.distributed.ReduceOp.MAX, group=self.cpu_sync_group
        )

        # Extract result - tensor is already on CPU
        any_gpu_invalid = invalid_tensor.item() > 0
        return any_gpu_invalid

    def on_train_start(self) -> None:
        """Initialize CPU sync group early if not already done."""
        if self.cpu_sync_group is None:
            self._initialize_cpu_sync_group()

    def training_step(self, batch, batch_idx) -> None:
        """
        Training step with manual optimization and OOM handling.
        """
        train_start = time.time()

        # Get optimizer and scheduler
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # Initialize tracking variables
        forward_oom = False
        loss_is_valid = True
        loss = None
        forward_time = 0
        loss_time = 0

        # Zero gradients before the forward so OOM failure in previous iteration does not affect current iteration
        opt.zero_grad()

        # Forward pass with OOM handling
        try:
            # Time forward pass
            forward_start = time.time()
            out_dict = self(batch)
            forward_time = time.time() - forward_start

            # Time loss computation
            loss_start = time.time()

            # Compute loss
            clip_feat = out_dict["clip_feat"]
            loss = self.caption_loss.loss(
                point_features=clip_feat,
                point_indices=batch["caption_data"]["point_indices"],
                caption_offsets=batch["caption_data"]["caption_offsets"],
                num_points_per_caption=batch["caption_data"]["num_points_per_caption"],
                captions=batch["caption_data"]["caption"],
                clip_encoder=self.clip_encoder,
            )

            loss_time = time.time() - loss_start

            # Check if loss is valid (not NaN or Inf)
            loss_is_valid = torch.isfinite(loss).all().item()
            if not loss_is_valid:
                log.warning(
                    f"Rank {self.trainer.global_rank}: Invalid loss detected (NaN or Inf) on batch {batch_idx}"
                )

        except torch.OutOfMemoryError:
            forward_oom = True
            log.warning(
                f"OOM error detected during forward pass in rank: {self.trainer.global_rank}. Total skips: {self.invalid_batch_count}/{self.max_invalid_batches}"
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Check if any GPU had invalid forward pass (OOM or invalid loss)
        forward_invalid = forward_oom or not loss_is_valid
        any_gpu_forward_invalid = self._sync_validity_status_cpu(forward_invalid)
        if any_gpu_forward_invalid:
            # Since the any_gpu_forward_invalid is synchronized, count does not need to be synced.
            self.invalid_batch_count += 1
            log.warning(
                f"Skipping batch {batch_idx} due to invalid forward pass. Total skips: {self.invalid_batch_count}/{self.max_invalid_batches}"
            )
            return

        # If forward pass was valid on all GPUs, proceed with backward pass
        backward_invalid = False
        backward_start = time.time()
        try:
            # Manual backward pass
            self.manual_backward(loss)
        except torch.OutOfMemoryError as e:
            log.warning(
                f"OOM error detected during backward pass in rank: {self.trainer.global_rank}: {e}"
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            backward_invalid = True

        any_gpu_backward_invalid = self._sync_validity_status_cpu(backward_invalid)
        if any_gpu_backward_invalid:
            log.warning(
                f"Skipping batch {batch_idx} due to invalid backward pass. Total skips: {self.invalid_batch_count}/{self.max_invalid_batches}"
            )
            # return
            # Backward OOM cannot be recovered due to gradient synchronization
            raise RuntimeError("OOM error detected during backward pass")

        # Check if we've exceeded the maximum allowed OOM skips
        if self.invalid_batch_count >= self.max_invalid_batches:
            raise RuntimeError(
                f"Too many invalid batches ({self.invalid_batch_count}). "
                f"Consider reducing batch size or checking data quality."
            )

        # If no issues, perform optimizer step
        valid_gradients = self._check_gradients()
        if not valid_gradients:
            log.warning(
                f"Skipping optimizer step due to invalid gradients in rank: {self.trainer.global_rank}"
            )
            return

        # Clip gradients if configured
        if self.trainer.gradient_clip_val is not None:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.trainer.gradient_clip_val,
                gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
            )

        # Optimizer step
        opt.step()

        # Scheduler step - respect the interval configuration
        # Step-based schedulers (e.g., OneCycleLR) are updated here
        # Epoch-based schedulers (e.g., StepLR) are updated in on_train_epoch_end()
        if sch is not None and self.hparams.scheduler_interval == "step":
            sch.step()

        # Log metrics
        backward_time = time.time() - backward_start

        lr = opt.param_groups[0]["lr"]
        log_metrics = dict(loss=loss, lr=lr)

        # useful metadata
        batch_size = len(batch["offset"]) - 1
        log_metrics["num_points"] = batch["coord"].shape[0] / batch_size
        log_metrics["num_objects"] = (
            batch["caption_data"]["caption_offsets"].shape[0] - 1
        ) / batch_size

        # Calculate training time and mark start of next data loading
        train_time = time.time() - train_start
        self.train_time(train_time)

        # Add timing metrics to existing logging
        log_metrics.update(
            {
                "time/forward": forward_time,
                "time/backward": backward_time,
                "time/loss": loss_time,
                "time/training": train_time,
            }
        )

        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=self.train_sync_dist,
        )

    def _check_gradients(self) -> bool:
        """
        Check for invalid gradients (NaN values).
        Returns True if gradients are valid.
        """
        valid_gradients = True
        params_list = list(self.named_parameters())
        if params_list:
            rand_param_idx = random.randint(0, len(params_list) - 1)
            for name, param in params_list[rand_param_idx:]:
                if param.grad is not None:
                    valid_gradients = not torch.isnan(param.grad).any()
                    if not valid_gradients:
                        break

        if not valid_gradients:
            self.skip_current_optimizer_step_count += 1
            log.warning(
                f"Detected NaN values in gradients. Not updating model parameters. "
                f"Skipped {self.skip_current_optimizer_step_count} optimizer steps."
            )
            if self.skip_current_optimizer_step_count > 10:
                raise ValueError("Too many optimizer steps skipped due to invalid gradients.")

        return valid_gradients

    def on_train_epoch_end(self) -> None:
        """Log OOM statistics at the end of training epoch and handle epoch-based schedulers."""
        super().on_train_epoch_end()

        # Handle epoch-based scheduler
        if self.hparams.scheduler_interval == "epoch":
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        if self.invalid_batch_count > 0:
            log.info(
                f"Training epoch ended with {self.invalid_batch_count} invalid batches "
                f"(OOM or invalid loss) out of {self.max_invalid_batches} allowed."
            )
            # Log epoch-level invalid batch statistics
            self.log(
                "train/epoch_skip_count",
                float(self.invalid_batch_count),
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
