from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

from src.models.lightning_modules.module_base import LitModuleBase
from src.models.optimization.fastai_lrscheduler import OneCycle
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc.utils import caption_utils
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class WarpConvNetLitModule(LitModuleBase):
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        dataset = self.trainer.train_dataloader.dataset
        batch = caption_utils.get_caption_batch(
            dataset.caption_cfg,
            {},
            batch,
            self.text_encoder,
            local_rank=self.local_rank,
        )
        ret_dict, tb_dict, dist_dict = self.forward(batch)

        loss = ret_dict["loss"].mean()

        # update and log metrics
        self.train_loss(loss)

        log_metrics = {
            "train/loss": self.train_loss,
            "train/loss_caption": tb_dict["caption_view"],
            "train/loss_segment": tb_dict.get("loss_seg", 0),
            "train/loss_binary": tb_dict.get("binary_loss", 0),
        }
        self.log_dict(
            log_metrics,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["batch_size"],
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        ret_dict = self.forward(batch)
        preds, labels = ret_dict["seg_preds"], ret_dict["seg_labels"]

        # update and log metrics
        self.val_confmat(preds, labels)

    def on_validation_epoch_end(self) -> None:
        confmat = self.val_confmat.compute().cpu().numpy()
        class_ious = {}
        class_accs = {}
        for i, class_name in enumerate(self.class_names):
            tp = confmat[i, i]
            fp = confmat[:, i].sum() - tp
            fn = confmat[i, :].sum() - tp

            class_ious[class_name] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            class_accs[class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0

        valid_class_idx = self.net.task_head.valid_class_idx
        miou = np.nanmean([class_ious[self.class_names[i]] for i in valid_class_idx])
        macc = np.nanmean([class_accs[self.class_names[i]] for i in valid_class_idx])
        self.val_miou_best.update(miou)

        log_metrics = {f"val/iou_{k}": v for k, v in class_ious.items()}
        log_metrics.update(
            {
                "val/best_miou": self.val_miou_best.compute(),
                "val/miou": miou,
                "val/macc": macc,
            }
        )

        if (
            hasattr(self.net.task_head, "base_class_idx")
            and self.net.task_head.base_class_idx is not None
            and len(self.net.task_head.base_class_idx) > 0
        ):
            base_class_idx = self.net.task_head.base_class_idx
            novel_class_idx = self.net.task_head.novel_class_idx
            miou_base = np.nanmean([class_ious[self.class_names[i]] for i in base_class_idx])
            miou_novel = np.nanmean([class_ious[self.class_names[i]] for i in novel_class_idx])
            hiou = 2 * miou_base * miou_novel / (miou_base + miou_novel + 1e-8)
            log_metrics.update(
                {
                    "val/miou_base": miou_base,
                    "val/miou_novel": miou_novel,
                    "val/hiou": hiou,
                }
            )

        self.log_dict(log_metrics, sync_dist=True, logger=True)

        self.val_confmat.reset()

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
