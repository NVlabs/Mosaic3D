import abc
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LitModuleBase(LightningModule, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        # Add timing metrics
        self.data_load_time = MeanMetric()
        self.forward_time = MeanMetric()
        self.loss_time = MeanMetric()
        self.train_time = MeanMetric()
        self._data_load_start = None
        self._train_start = None
        self._forward_start = None
        self._loss_start = None

    def configure_model(self) -> None:
        if self.net is not None:
            return
        else:
            self.net = self.hparams.net()

        # Define loss modules by calling instantiation functions
        losses = []
        if isinstance(self.hparams.losses, Dict):
            for loss_name, loss in self.hparams.losses.items():
                losses.append(loss)
        elif isinstance(self.hparams.losses, list):
            losses = self.hparams.losses
        self.loss = nn.ModuleList(losses)

        # Setup loss weights
        self.loss_weights = self.hparams.loss_weights

    def setup(self, stage: str) -> None:
        """Setup the model for training, validation, and testing."""

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        # Mark end of data loading time and start of potential CLIP processing
        if self._data_load_start is not None:
            data_load_time = time.time() - self._data_load_start
            self.data_load_time(data_load_time)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # Mark start of next data loading
        self._data_load_start = time.time()

    def forward(self, batch) -> Any:
        return self.net(batch)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step for the model."""
        self._train_start = time.time()

        # Time forward pass
        self._forward_start = time.time()
        ret_dict, tb_dict, dist_dict = self.forward(batch)
        forward_time = time.time() - self._forward_start
        self.forward_time(forward_time)

        # Time loss computation
        self._loss_start = time.time()
        # compute loss
        loss_dict = {}
        for loss_module in self.loss:
            loss_dict.update(loss_module(ret_dict, tb_dict, dist_dict))

        loss = 0
        for k, v in loss_dict.items():
            weight_name = k + "_weight"
            if weight_name in self.loss_weights:
                loss = loss + v.mean() * self.loss_weights[weight_name]
        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        log_metrics = {"train/loss": loss.item()}
        for k, v in loss_dict.items():
            log_metrics[f"train/{k}"] = v.mean().item()

        # Calculate total training time and mark start of next data loading
        train_time = time.time() - self._train_start
        self.train_time(train_time)
        self._data_load_start = time.time()

        # Add timing metrics to logging
        self.log_dict(
            {
                "time/data_loading": self.data_load_time.compute(),
                "time/forward": self.forward_time.compute(),
                "time/loss": self.loss_time.compute(),
                "time/training": self.train_time.compute(),
            },
            on_step=True,
            prog_bar=True,
            logger=True,
        )

        self.log_dict(
            log_metrics,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        ret_dict = self.forward(batch)
        self.val_results.append(ret_dict)

    def on_validation_epoch_end(self) -> None:
        """Validation epoch end hook."""
        # compute metrics
        # self.compute_metrics()

        # reset val_results
        # self.val_results = []

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.optimizer.func.__name__.startswith("build_"):
            optimizer = self.hparams.optimizer(model=self)
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            if self.hparams.scheduler.func.__name__ == "OneCycleLR":
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
                )
            elif self.hparams.scheduler.func.__name__ == "PolynomialLR":
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer,
                    total_iters=self.trainer.estimated_stepping_batches,
                )
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
