from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

from src.models.components.misc import offset2batch
from src.models.regionplc_refactor import build_text_model, load_data_to_gpu
from src.models.regionplc_refactor.text_networks import load_text_embedding_from_path
from src.models.regionplc_refactor.utils import caption_utils
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class RegionPLCLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        text_encoder: Dict,
        text_embed_path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        # load text embed
        self.text_embed = load_text_embedding_from_path(text_embed_path, log)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_confmat = None
        self.val_miou_best = MaxMetric()

    def configure_model(self) -> None:
        if self.net is not None:
            return

        self.net = self.hparams.net(dataset=self.data_cfg.train_dataset)
        self.net.task_head.set_cls_head_with_text_embed(self.text_embed)
        self.text_encoder = build_text_model(self.hparams.text_encoder)

    def setup(self, stage: str) -> None:
        val_dataloader = self.trainer.datamodule.val_dataloader()
        self.class_names = val_dataloader.dataset.class_names

        self.val_confmat = MulticlassConfusionMatrix(
            num_classes=len(self.class_names),
            ignore_index=val_dataloader.dataset.ignore_label,
        )

    def forward(self, batch) -> torch.Tensor:
        # batch_ids = offset2batch(batch["offset"])
        # batch["voxel_coords"] = torch.cat(
        #     [batch_ids.unsqueeze(-1).int(), batch["grid_coord"].int()], dim=1
        # ).contiguous()
        # batch["v2p_map"] = batch["inverse"].long()
        # batch["voxel_features"] = batch["feat"]
        # batch["spatial_shape"] = torch.clip(
        #     batch["grid_coord"].max(0).values + 1, 128, None
        # )
        # batch["batch_size"] = len(batch["offset"])
        # batch["labels"] = batch["segment"]
        # batch["binary_labels"] = batch["binary"]
        # batch["batch_idxs"] = offset2batch(batch["offset_origin"])
        load_data_to_gpu(batch)
        return self.net(batch)

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
        log_kwargs = dict(
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["batch_size"],
        )
        self.log("train/loss", self.train_loss, **log_kwargs)
        self.log("train/loss_segment", tb_dict["loss_seg"], **log_kwargs)
        self.log("train/loss_caption", tb_dict["caption_view"], **log_kwargs)
        self.log("train/loss_binary", tb_dict["binary_loss"], **log_kwargs)
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

        miou = np.array(list(class_ious.values())).mean()
        macc = np.array(list(class_accs.values())).mean()
        allacc = np.diag(confmat).sum() / (confmat.sum() + 1e-10)

        for k, v in class_ious.items():
            self.log(f"val/iou_{k}", v, sync_dist=True, logger=True)

        self.val_miou_best.update(miou)
        self.log("val/best_miou", self.val_miou_best.compute(), sync_dist=True, logger=True)
        self.log("val/miou", miou, sync_dist=True, logger=True)
        self.log("val/macc", macc, sync_dist=True, logger=True)
        self.log("val/allacc", allacc, sync_dist=True, logger=True)

        self.val_confmat.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    # def setup(self, stage: str) -> None:
    #     if self.hparams.compile and stage == "fit":
    #         self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            if self.hparams.scheduler.func.__name__ == "OneCycleLR":
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


if __name__ == "__main__":
    _ = RegionPLCLitModule(None, None, None, None)
