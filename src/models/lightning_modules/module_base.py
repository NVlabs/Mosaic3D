import abc
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule

from src.models.optimization.fastai_lrscheduler import OneCycle
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LitModuleBase(LightningModule, metaclass=abc.ABCMeta):
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

    def forward(self, batch) -> Any:
        return self.net(batch)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step for the model."""
        ret_dict, tb_dict, dist_dict = self.forward(batch)

        # compute loss
        loss_dict = {}
        for loss_module in self.loss:
            loss_dict.update(loss_module(ret_dict, tb_dict, dist_dict))

        loss = 0
        for k, v in loss_dict.items():
            weight_name = k + "_weight"
            if weight_name in self.loss_weights:
                loss = loss + v.mean() * self.loss_weights[weight_name]

        log_metrics = {"train/loss": loss.item()}
        for k, v in loss_dict.items():
            log_metrics[f"train/{k}"] = v.mean().item()

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
            elif self.hparams.scheduler.func.__name__.startswith("build_"):
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
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
        if isinstance(scheduler, OneCycle):
            scheduler.step(self.trainer.global_step)
        elif metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
