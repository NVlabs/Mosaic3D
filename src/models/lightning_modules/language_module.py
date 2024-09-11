from typing import Any, Dict, List, Optional

from einops import rearrange, repeat
from torch_scatter import segment_csr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

import src.models.regionplc.utils.caption_utils as caption_utils
from src.models.lightning_modules.module_base import LitModuleBase
from src.models.losses.caption_loss import (
    CaptionLoss,
    DenseCaptionAlignmentLoss,
    CaptionAlignmentLoss,
)
from src.models.losses.clip_alignment_loss import (
    CLIPAlignmentLoss,
    compute_clip_image_alignment,
    compute_clip_text_cosine_similarity,
)
from src.models.regionplc.clip_models import build_clip_model
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class DenseLanguageLitModule(LitModuleBase):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        clip_encoder: Dict,
        compile: bool,
        loss_cfg: Dict,
        eval_cfg: Dict,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        # loss functions
        self.is_entity = loss_cfg["caption_loss"].get("is_entity", False)
        self.caption_loss_type = loss_cfg["caption_loss"].get("type", "contrastive")
        if self.caption_loss_type == "contrastive":
            self.caption_loss = CaptionLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "alignment":
            self.caption_loss = DenseCaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "region_alignment":
            self.caption_loss = CaptionAlignmentLoss(**loss_cfg["caption_loss"])
        else:
            raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")

        self.clip_alignment_loss = CLIPAlignmentLoss(**loss_cfg["seg_loss"])
        self.binary_loss = nn.BCEWithLogitsLoss() if loss_cfg.get("binary_loss", None) else None

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_confmat = None
        self.val_miou_best = MaxMetric()
        self.val_clip_text_score = MeanMetric()
        self.val_clip_image_score = MeanMetric()

        # Sync distributed metrics
        self.train_sync_dist = loss_cfg.get("sync_dist", False)

        # CLIP score for eval / train
        self.train_clip_text_alignment = loss_cfg.get("train_clip_text_alignment", False)
        self.train_clip_image_alignment = loss_cfg.get("train_clip_image_alignment", False)
        self.eval_clip_text_alignment = eval_cfg.get("eval_clip_text_alignment", False)
        self.eval_clip_image_alignment = eval_cfg.get("eval_clip_image_alignment", False)

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        self.net = self.hparams.net()
        # Print network on the first GPU
        if self.local_rank == 0:
            log.info(self.net)

        # clip encoder
        self.clip_encoder = build_clip_model(self.hparams.clip_encoder)
        # freeze clip encoder
        for params in self.clip_encoder.parameters():
            params.requires_grad = False

    def setup(self, stage: str) -> None:
        val_dataloader = self.trainer.datamodule.val_dataloader()
        self.class_names = val_dataloader.dataset.class_names

        self.val_confmat = MulticlassConfusionMatrix(
            num_classes=len(self.class_names),
            ignore_index=val_dataloader.dataset.ignore_label,
        )

        self.base_class_idx = val_dataloader.dataset.base_class_idx
        self.novel_class_idx = val_dataloader.dataset.novel_class_idx
        self.valid_class_idx = val_dataloader.dataset.valid_class_idx

    def forward(self, batch: Any) -> Dict[str, Any]:
        point = self.net(batch)
        out_dict = self._output_to_dict(point, batch)
        return out_dict

    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def match_labels(self, batch: Dict[str, Any], pred_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Match the output of the network to the labels in the batch."""
        return batch

    def training_step(self, batch, batch_idx):
        # Prepare caption data in bf16
        with torch.cuda.amp.autocast(enabled=True) and torch.inference_mode():
            batched_captions: List[List[str]] = batch["caption_data"]["caption"]

            caption_fn_kargs = {
                "batched_captions": batched_captions,
                "clip_encoder": self.clip_encoder,
                "is_entity": self.is_entity,
            }
            if self.caption_loss_type == "contrastive":
                caption_embeds, caption_targets = caption_utils.get_unique_caption_batch(
                    **caption_fn_kargs
                )
            elif (
                self.caption_loss_type == "alignment"
                or self.caption_loss_type == "region_alignment"
            ):
                caption_embeds = caption_utils.get_caption_batch(**caption_fn_kargs)
            else:
                raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")

        # copy for backward
        caption_embeds = (
            caption_embeds.clone() if isinstance(caption_embeds, torch.Tensor) else caption_embeds
        )

        # Forward
        out_dict = self(batch)

        # Match labels
        batch = self.match_labels(batch=batch, pred_dict=out_dict)

        # loss
        binary_loss, seg_loss, caption_loss = 0, 0, 0
        clip_image_alignment_loss = 0

        if self.binary_loss is not None:
            binary_labels = batch["binary"]
            binary_scores = out_dict["binary_scores"]
            valid_idx = binary_labels != -100
            binary_loss = (
                self.binary_loss(
                    binary_scores.view(-1)[valid_idx],
                    binary_labels.view(-1)[valid_idx].to(binary_scores),
                )
                * self.hparams.loss_cfg.binary_loss_weight
            )

        clip_feat = out_dict["clip_feat"]
        if not self.clip_alignment_loss.eval_only:
            seg_loss = (
                self.clip_alignment_loss.loss(clip_feat, batch["segment"])
                * self.hparams.loss_cfg.seg_loss_weight
            )

        caption_loss_kargs = {
            "batched_list_of_point_indices": batch["caption_data"]["idx"],
            "input_batch_offsets": batch["offset"],
            "valid_mask": out_dict.get("mapping_valid_mask", None),
        }
        if self.caption_loss_type == "contrastive":
            caption_loss_kargs.update(
                {
                    "unique_caption_embeds": caption_embeds,
                    "caption_targets": caption_targets,
                }
            )
        elif self.caption_loss_type == "alignment" or self.caption_loss_type == "region_alignment":
            caption_loss_kargs["caption_embeddings"] = caption_embeds
        caption_loss = (
            self.caption_loss.loss(clip_feat, **caption_loss_kargs)
            * self.hparams.loss_cfg.caption_loss_weight
        )

        # CLIP image loss
        if self.train_clip_image_alignment:
            clip_image_alignment_loss = (
                compute_clip_image_alignment(
                    clip_encoder=self.clip_encoder,
                    clip_processed_image=batch["clip_processed_image"],
                    point_feat=out_dict["clip_feat"],
                    clip_point_indices=batch["clip_point_indices"],
                    clip_indices_image_to_point=batch["clip_indices_image_to_point"],
                    is_loss=True,
                )
                * self.hparams.loss_cfg.clip_image_loss_weight
            )

        loss = binary_loss + seg_loss + caption_loss + clip_image_alignment_loss

        log_metrics = dict(
            loss=loss,
            caption_loss=caption_loss,
        )
        if self.binary_loss is not None:
            log_metrics["binary_loss"] = binary_loss
        if not self.clip_alignment_loss.eval_only:
            log_metrics["seg_loss"] = seg_loss
        if self.train_clip_image_alignment:
            log_metrics["clip_image_alignment_loss"] = clip_image_alignment_loss

        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=self.train_sync_dist,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        out_dict = self(batch)
        batch = self.match_labels(batch=batch, pred_dict=out_dict)
        logits = out_dict["logits"]

        new_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        new_logits[..., self.valid_class_idx] = logits[..., self.valid_class_idx]

        if self.binary_loss is not None:
            base_scores = new_logits[..., self.base_class_idx].softmax(dim=-1)
            novel_scores = new_logits[..., self.novel_class_idx].softmax(dim=-1)
            scores = new_logits.clone().float()
            scores[:] = 0.0
            scores[..., self.base_class_idx] = base_scores
            scores[..., self.novel_class_idx] = novel_scores

            weights = torch.sigmoid(out_dict["binary_scores"])
            weights = weights.repeat(1, scores.shape[-1])
            weights[..., self.novel_class_idx] = 1 - weights[..., self.novel_class_idx]

            scores = scores * weights
            scores /= scores.sum(-1, keepdim=True)
            preds = scores.max(1)[1]
        else:
            preds = new_logits.max(1)[1]

        if self.eval_clip_image_alignment:
            clip_scores = compute_clip_image_alignment(
                clip_encoder=self.clip_encoder,
                clip_processed_image=batch["clip_processed_image"],
                point_feat=out_dict["clip_feat"],
                clip_point_indices=batch["clip_point_indices"],
                clip_indices_image_to_point=batch["clip_indices_image_to_point"],
                is_loss=False,
            )
            clip_avg_score = clip_scores.mean().cpu().numpy()
            self.val_clip_image_score.update(clip_avg_score)

        if self.eval_clip_text_alignment:
            clip_avg_score = compute_clip_text_cosine_similarity(
                clip_encoder=self.clip_encoder,
                clip_tokenized_text=batch["clip_tokenized_text"],
                point_feat=out_dict["clip_feat"],
                offset=batch["offset"],
                point_indices_to_caption=batch["caption_data"]["idx"],
            )
            self.val_clip_text_score.update(clip_avg_score)

        # update and log metrics
        self.val_confmat(preds, batch["segment"])

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

        valid_class_idx = self.valid_class_idx
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
            hasattr(self, "base_class_idx")
            and self.base_class_idx is not None
            and len(self.base_class_idx) > 0
        ):
            base_class_idx = self.base_class_idx
            novel_class_idx = self.novel_class_idx
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

        if self.eval_clip_text_alignment:
            log_metrics.update({"val/clip_text_score": self.val_clip_text_score.compute()})
        if self.eval_clip_image_alignment:
            log_metrics.update(
                {
                    "val/clip_image_score": self.val_clip_image_score.compute(),
                }
            )

        self.log_dict(log_metrics, sync_dist=True, logger=True)

        self.val_confmat.reset()
        self.val_clip_text_score.reset()
        self.val_clip_image_score.reset()

    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
