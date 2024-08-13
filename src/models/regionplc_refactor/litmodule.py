import functools
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

import src.models.regionplc.utils.caption_utils as caption_utils
from src.models.components.structure import Point
from src.models.optimization.fastai_lrscheduler import OneCycle
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc_refactor.backbone import SparseUNet
from src.models.regionplc_refactor.head import BinaryHead, CaptionHead, TextSegHead
from src.models.regionplc_refactor.modules import MLP
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class RegionPLCLitModule(LightningModule):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        text_encoder: Dict,
        compile: bool,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_confmat = None
        self.val_miou_best = MaxMetric()

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        backbone = SparseUNet(**self.hparams.net.backbone_cfg)

        mid_channel, text_channel = (
            self.hparams.net.backbone_cfg.mid_channel,
            self.hparams.net.adapter_cfg.text_channel,
        )
        adapter_channel_list = [mid_channel, text_channel]
        if self.hparams.net.adapter_cfg.num_layers == 2:
            multiplier = int(np.log2(text_channel / mid_channel))
            adapter_channel_list = [mid_channel, mid_channel * multiplier, text_channel]
        adapter = MLP(
            adapter_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            num_layers=self.hparams.net.adapter_cfg.num_layers,
            last_norm_fn=self.hparams.net.adapter_cfg.last_norm,
        )

        binary_head = None
        if self.hparams.net.enable_binary_head:
            binary_head = BinaryHead(**self.hparams.net.binary_head_cfg)
            binary_head.register_hook_for_binary_head(backbone)

        task_head = TextSegHead(**self.hparams.net.task_head_cfg)
        caption_head = CaptionHead(**self.hparams.net.caption_head_cfg)

        self.net = nn.ModuleDict(
            {
                "backbone_3d": backbone,
                "adapter": adapter,
                "binary_head": binary_head,
                "task_head": task_head,
                "caption_head": caption_head,
            }
        )

        # text encoder
        self.text_encoder = build_text_model(self.hparams.text_encoder)

    def setup(self, stage: str) -> None:
        val_dataloader = self.trainer.datamodule.val_dataloader()
        self.class_names = val_dataloader.dataset.class_names

        self.val_confmat = MulticlassConfusionMatrix(
            num_classes=len(self.class_names),
            ignore_index=val_dataloader.dataset.ignore_label,
        )

    def training_step(self, batch, batch_idx):
        caption_infos = caption_utils.get_caption_batch_refactor(
            batch["caption_data"], self.text_encoder, local_rank=self.local_rank
        )
        batch.update(caption_infos)

        point = Point(batch)
        point.sparsify(pad=128)

        sparse_tensor = point.sparse_conv_feat
        backbone_feat = self.net.backbone_3d(sparse_tensor)
        adapter_feat = self.net.adapter(backbone_feat.features)

        sparse_tensor = sparse_tensor.replace_feature(adapter_feat)
        point.sparse_conv_feat = sparse_tensor

        binary_head_output = None
        if self.net.binary_head is not None:
            binary_head_output = self.net.binary_head(point)

        task_head_output = self.net.task_head(point, binary_head_output)
        caption_head_output = self.net.caption_head(point)

        # loss
        binary_loss = self.net.binary_head.get_loss(
            binary_head_output["binary_scores"], point.binary
        )
        seg_loss = self.net.task_head.get_loss(task_head_output["seg_scores"], point.segment)
        caption_loss = self.net.caption_head.get_loss(caption_head_output)
        loss = binary_loss + seg_loss + caption_loss

        log_metrics = dict(
            loss=loss,
            binary_loss=binary_loss,
            caption_loss=caption_loss,
            seg_loss=seg_loss,
        )
        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        point = Point(batch)
        point.grid_size = 0.02
        point.sparsify(pad=128)

        sparse_tensor = point.sparse_conv_feat
        backbone_feat = self.net.backbone_3d(sparse_tensor)
        adapter_feat = self.net.adapter(backbone_feat.features)
        sparse_tensor = sparse_tensor.replace_feature(adapter_feat)
        point.sparse_conv_feat = sparse_tensor

        binary_head_output = None
        if self.net.binary_head is not None:
            binary_head_output = self.net.binary_head(point)

        task_head_output = self.net.task_head(point, binary_head_output)
        preds = task_head_output["seg_preds"]
        labels = batch["segment"]

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

    def test_step(self, batch, batch_idx):
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
