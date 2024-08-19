from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

from src.models.optimization.fastai_lrscheduler import OneCycle
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc.utils import caption_utils
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LitModuleBase(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        losses: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        self.val_results = []

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
        val_dataloader = self.trainer.datamodule.val_dataloader()
        self.class_names = val_dataloader.dataset.class_names

        self.val_confmat = MulticlassConfusionMatrix(
            num_classes=len(self.class_names),
            ignore_index=val_dataloader.dataset.ignore_label,
        )

    def forward(self, batch) -> Any:
        return self.net(batch)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
        # Aggregate results
        val_results = {}
        for key in self.val_results[0].keys():
            val_results[key] = torch.cat([x[key] for x in self.val_results], dim=0).mean(dim=0)
        self.val_results = []
        self.log_dict(val_results, sync_dist=True, logger=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.optimizer.func.__name__.startswith("build_"):
            optimizer = self.hparams.optimizer(model=self.net)
        else:
            optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            if self.hparams.scheduler.func.__name__ == "OneCycleLR":
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
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
