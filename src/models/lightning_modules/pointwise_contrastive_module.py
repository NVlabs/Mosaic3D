from typing import Any, Dict, List, Optional, Callable
from jaxtyping import Float

import os
import time
import random
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import MaxMetric

from src.models.lightning_modules.module_base import LitModuleBase
from src.models.losses.caption_pointwise_contrastive_loss import CaptionPointwiseContrastiveLoss
from src.models.losses.caption_loss import CaptionLoss
from src.models.components.validation_evaluator import ValidationEvaluator, CLIPTextClassification
from src.models.components.text_encoder import get_text_encoder
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


# TODO(cchoy 2025-04-30): Remove the DenseLanguageLitModule in language_module.py in the future
class PointwiseContrastiveLanguageLitModule(LitModuleBase):
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
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        # loss functions
        self.caption_loss_type = loss_cfg["caption_loss"].get("type", "object_contrastive")
        if self.caption_loss_type == "object_contrastive":
            self.caption_loss = CaptionLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "point_contrastive":
            self.caption_loss = CaptionPointwiseContrastiveLoss(
                **loss_cfg["caption_loss"],
            )
        else:
            raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")

        # for tracking best so far validation accuracy
        self.val_best_metric = MaxMetric()

        # Save val_best_metric to hparams and restore if resuming
        self.save_hyperparameters({"val_best_metric": self.val_best_metric})

        # Sync distributed metrics
        self.train_sync_dist = loss_cfg.get("sync_dist", False)

        self.validation_evaluators = None
        self.val_dataset_names = {}  # To map dataloader_idx to postfix

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        self.net = self.hparams.net()
        # Print network on the first GPU
        if self.local_rank == 0:
            log.info(self.net)

        # Clip encoder
        self.clip_encoder = get_text_encoder(
            **self.hparams.clip_encoder,
            device=self.device,
            torch_dtype=torch.bfloat16,
        )

        # Setup evaluators
        for evaluator in self.validation_evaluators.values():
            evaluator.setup(self.clip_encoder)

    def on_load_checkpoint(self, checkpoint):
        # Restore the state of val_best_metric saved in hparams
        if "val_best_metric" in checkpoint.get("hyper_parameters", {}):
            # Load the state dict of the metric
            metric_state = checkpoint["hyper_parameters"]["val_best_metric"]
            # Create a temporary metric object to load the state, then update self.val_best_metric
            # This assumes the state dict format is compatible.
            # Alternatively, directly load 'max_value' if that's how it's saved.
            if isinstance(metric_state, dict) and "max_value" in metric_state:
                self.val_best_metric.update(metric_state["max_value"])
            elif isinstance(metric_state, MaxMetric):  # If the whole object was somehow saved
                self.val_best_metric.load_state_dict(metric_state.state_dict())
            else:
                log.warning("Could not restore val_best_metric state from checkpoint.")
        # Ensure superclass loading happens
        # super().on_load_checkpoint(checkpoint) # Call depends on base class implementation needs

    def setup(self, stage: str) -> None:
        """Setup validation datasets, metrics, and text embeddings."""
        # Check if already set up (by checking if the dict is populated)
        if self.validation_evaluators:
            return  # Already setup

        val_dataloaders = self.trainer.datamodule.val_dataloader()
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]

        val_dataset_names_dict = {}
        evaluators_dict = {}  # To hold ValidationEvaluator instances

        for i, val_dataloader in enumerate(val_dataloaders):
            dataset = val_dataloader.dataset
            if not hasattr(dataset, "CLASS_LABELS") or not hasattr(dataset, "log_postfix"):
                log.error(
                    f"Validation dataset {i} missing required attributes CLASS_LABELS or log_postfix."
                )
                continue  # Skip this dataset if essential info is missing

            class_names = dataset.CLASS_LABELS
            postfix = dataset.log_postfix
            assert postfix is not None, "log_postfix is required for clarity"
            val_dataset_names_dict[i] = postfix

            # Create and store the evaluator for this dataset
            evaluator = ValidationEvaluator(
                use_prompt=self.hparams.use_prompt,
                log_func=self.log_dict,
                trainer=self.trainer,
                device=self.device,
                postfix=postfix,
                class_names=class_names,
                ignore_label=getattr(dataset, "ignore_label", -100),
                fg_class_idx=getattr(dataset, "fg_class_idx", list(range(len(class_names)))),
                bg_class_idx=getattr(dataset, "bg_class_idx", []),
                base_class_idx=getattr(dataset, "base_class_idx", None),
                novel_class_idx=getattr(dataset, "novel_class_idx", None),
                subset_mapper=getattr(dataset, "subset_mapper", None),
                instance_ignore_class_idx=getattr(dataset, "instance_ignore_class_idx", []),
            )
            evaluators_dict[postfix] = evaluator

        self.validation_evaluators = evaluators_dict
        self.val_dataset_names = val_dataset_names_dict  # Store the index to name mapping
        log.info(
            f"Validation evaluators setup complete for datasets: {list(self.validation_evaluators.keys())}"
        )

    def forward(self, batch: Any) -> Dict[str, Any]:
        point = self.net(batch)
        out_dict = self._output_to_dict(point, batch)
        return out_dict

    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        """
        Convert the output of the network to a dictionary with the following keys:
        - "clip_feat": the clip features of the points
        """
        # This should be implemented by subclasses
        if isinstance(output, torch.Tensor):
            return {"clip_feat": output}
        elif "feat" in output and "v2p_map" in output:  # PointTransformerV3
            clip_feat = output.sparse_conv_feat.features[output.v2p_map]
            return {"clip_feat": clip_feat}
        elif isinstance(output, dict) and "clip_feat" in output:
            return output
        else:
            raise NotImplementedError(
                "Subclass must implement _output_to_dict to return {'clip_feat': ...}"
            )

    def training_step(self, batch, batch_idx):
        self._train_start = time.time()

        # Time forward pass
        self._forward_start = time.time()
        out_dict = self(batch)
        forward_time = time.time() - self._forward_start
        self.forward_time(forward_time)

        # Time loss computation
        self._loss_start = time.time()

        # loss
        clip_feat = out_dict["clip_feat"]
        loss = self.caption_loss.loss(
            point_features=clip_feat,
            point_indices=batch["caption_data"]["point_indices"],
            caption_offsets=batch["caption_data"]["caption_offsets"],
            num_points_per_caption=batch["caption_data"]["num_points_per_caption"],
            captions=batch["caption_data"]["caption"],
            clip_encoder=self.clip_encoder,
        )

        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        lr = self.optimizers().param_groups[0]["lr"]
        log_metrics = dict(loss=loss, lr=lr)

        # useful metadata
        batch_size = len(batch["offset"]) - 1
        log_metrics["num_points"] = batch["coord"].shape[0] / batch_size
        log_metrics["num_objects"] = (
            batch["caption_data"]["caption_offsets"].shape[0] - 1
        ) / batch_size

        # Calculate training time and mark start of next data loading
        train_time = time.time() - self._train_start
        self.train_time(train_time)
        self._data_load_start = time.time()

        # Add timing metrics to existing logging
        log_metrics.update(
            {
                "time/data_loading": self.data_load_time.compute(),
                "time/forward": self.forward_time.compute(),
                "time/loss": self.loss_time.compute(),
                "time/training": self.train_time.compute(),
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
        return loss

    def on_validation_epoch_start(self):
        """Called before the validation epoch begins."""
        if self.validation_evaluators:
            for evaluator in self.validation_evaluators.values():
                evaluator.reset()
        else:
            log.warning("Validation evaluators not initialized at epoch start.")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Processes a single validation batch."""
        postfix = self.val_dataset_names[dataloader_idx]
        evaluator = self.validation_evaluators[postfix]

        with torch.no_grad():
            out_dict = self(batch)
            evaluator.update(batch, out_dict)

    def on_validation_epoch_end(self) -> None:
        """Called after the validation epoch ends."""
        if not self.validation_evaluators:
            log.warning("Validation evaluators not initialized at epoch end.")
            return

        all_log_metrics = {}
        for postfix, evaluator in self.validation_evaluators.items():
            dataset_metrics = evaluator.compute()
            all_log_metrics.update(dataset_metrics)

        # Update the module's best metric tracker (for checkpointing)
        primary_metric_key = self.hparams.best_metric
        if primary_metric_key in all_log_metrics:
            current_value = all_log_metrics[primary_metric_key]
            # Ensure value is a tensor for MaxMetric
            current_value_tensor = torch.tensor(current_value, device=self.device)
            self.val_best_metric.update(current_value_tensor)
            # Add the best value (computed across epochs) to the logs
            all_log_metrics[f"{primary_metric_key}_best"] = self.val_best_metric.compute().item()
        else:
            log.warning(
                f"Best metric key '{primary_metric_key}' not found in computed metrics. Cannot update best score."
            )
            # Log the current best score even if not updated this epoch
            all_log_metrics[f"{primary_metric_key}_best"] = self.val_best_metric.compute().item()

        # Log all collected metrics if not sanity checking
        if not self.trainer.sanity_checking:
            self.log_dict(all_log_metrics, sync_dist=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Assuming test uses the same logic as validation
        if not self.validation_evaluators:
            log.info("Setting up evaluators for test stage.")
            self.setup(stage="test")

        if not self.validation_evaluators:
            log.error(
                f"Cannot run test step for batch {batch_idx}: evaluators failed to initialize."
            )
            return

        postfix = self.val_dataset_names[dataloader_idx]
        evaluator = self.validation_evaluators[postfix]

        with torch.no_grad():
            out_dict = self(batch)
            evaluator.update(batch, out_dict)

    def on_test_epoch_end(self) -> None:
        """Called after the test epoch ends."""
        # Re-use the validation epoch end logic, which computes and logs metrics
        log.info("Computing and logging metrics for test epoch...")
        self.on_validation_epoch_end()
        # Log final test metric
        final_best_metric_val = self.val_best_metric.compute().item()
        log.info(
            f"Test epoch finished. Final best metric ({self.hparams.best_metric}): {final_best_metric_val}"
        )
        # Optionally log it again explicitly as a test metric
        self.log(
            f"test/{self.hparams.best_metric}_best_overall",
            final_best_metric_val,
            sync_dist=True,
            logger=True,
        )

    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
