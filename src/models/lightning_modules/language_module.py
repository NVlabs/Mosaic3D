import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

import src.utils.caption_utils as caption_utils
from src.models.components.clip_models import build_clip_model, download_clip_model
from src.models.components.evaluator import InstanceSegmentationEvaluator
from src.models.lightning_modules.module_base import LitModuleBase
from src.models.losses.caption_loss import (
    CaptionAlignmentLoss,
    CaptionCLIPLoss,
    CaptionLoss,
    CaptionSigLIPLoss,
    DenseCaptionAlignmentLoss,
)
from src.models.losses.clip_alignment_loss import (
    CLIPAlignmentEval,
    CLIPAlignmentLoss,
    compute_clip_image_alignment,
    compute_clip_text_cosine_similarity,
)
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
        eval_cfg: Optional[Dict] = None,
        use_prompt: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        # loss functions
        self.caption_loss_type = loss_cfg["caption_loss"].get("type", "contrastive")
        if self.caption_loss_type == "contrastive":
            self.caption_loss = CaptionLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "alignment":
            self.caption_loss = DenseCaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "region_alignment":
            self.caption_loss = CaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "clip":
            self.caption_loss = CaptionCLIPLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "siglip":
            self.caption_loss = CaptionSigLIPLoss(**loss_cfg["caption_loss"])
        else:
            raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")

        self.clip_alignment_loss = (
            CLIPAlignmentLoss(**loss_cfg["seg_loss"])
            if not loss_cfg["seg_loss"]["eval_only"]
            else None
        )
        self.binary_loss = nn.BCEWithLogitsLoss() if loss_cfg.get("binary_loss", None) else None

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metrics = nn.ModuleList()
        self.val_class_info = []

        # Sync distributed metrics
        self.train_sync_dist = loss_cfg.get("sync_dist", False)

        # CLIP score for eval / train
        if eval_cfg is not None:
            self.train_clip_image_alignment = loss_cfg.get("train_clip_image_alignment", False)
            self.eval_clip_text_alignment = eval_cfg.get("eval_clip_text_alignment", False)
            self.eval_clip_image_alignment = eval_cfg.get("eval_clip_image_alignment", False)
            self.ignore_background = eval_cfg.get("ignore_background", False)
            self.ignore_class_prob = eval_cfg.get("ignore_class_prob", False)
        else:
            self.train_clip_image_alignment = False
            self.eval_clip_text_alignment = False
            self.eval_clip_image_alignment = False
            self.ignore_background = False
            self.ignore_class_prob = False

    def prepare_data(self) -> None:
        # download clip model on rank 0
        ckpt_path = download_clip_model(self.hparams.clip_encoder)
        log.info(f"Downloaded CLIP model to {ckpt_path}")

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        self.net = self.hparams.net()
        # Print network on the first GPU
        if self.local_rank == 0:
            log.info(self.net)

        # clip encoder
        self.clip_encoder = build_clip_model(self.hparams.clip_encoder, device=self.device)

        # freeze clip encoder
        for params in self.clip_encoder.parameters():
            params.requires_grad = False

    def setup(self, stage: str) -> None:
        val_dataloaders = self.trainer.datamodule.val_dataloader()
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]

        for val_dataloader in val_dataloaders:
            dataset = val_dataloader.dataset
            class_names = dataset.CLASS_LABELS
            postfix = dataset.log_postfix
            assert postfix is not None, "log_postfix is required for clarity"

            # semantic segmentation metrics (default)
            val_metric = nn.ModuleDict(
                {
                    "confmat": MulticlassConfusionMatrix(
                        num_classes=len(class_names),
                        ignore_index=dataset.ignore_label,
                    ),
                    "confmat_all": MulticlassConfusionMatrix(
                        num_classes=len(class_names),
                        ignore_index=dataset.ignore_label,
                    ),
                    "miou_all_best": MaxMetric(),
                    "clip_text_score": MeanMetric(),
                    "clip_image_score": MeanMetric(),
                }
            )
            # instance segmentation metrics (optional)
            if dataset.mask_dir is not None:
                val_metric["mAP_evaluator"] = InstanceSegmentationEvaluator(
                    class_names=class_names,
                    segment_ignore_index=dataset.instance_ignore_class_idx
                    + [dataset.ignore_label],
                    instance_ignore_index=dataset.ignore_label,
                    subset_mapper=dataset.subset_mapper,
                )
            # dataset class info
            val_class_info = dict(
                postfix=postfix,
                class_names=class_names,
                base_class_idx=dataset.base_class_idx,
                novel_class_idx=dataset.novel_class_idx,
                fg_class_idx=dataset.fg_class_idx,
                bg_class_idx=dataset.bg_class_idx,
                ignore_label=dataset.ignore_label,
                instance_ignore_class_idx=dataset.instance_ignore_class_idx,
            )
            self.val_metrics.append(val_metric)
            self.val_class_info.append(val_class_info)

        self.clip_alignment_eval = nn.ModuleList(
            [
                CLIPAlignmentEval(**self.hparams.eval_cfg.seg_eval)
                for _ in range(len(self.val_metrics))
            ]
        )

    def forward(self, batch: Any) -> Dict[str, Any]:
        point = self.net(batch)
        out_dict = self._output_to_dict(point, batch)
        return out_dict

    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        raise NotImplementedError

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
                * self.hparams.loss_cfg.weights.binary_loss
            )

        clip_feat = out_dict["clip_feat"]
        if self.clip_alignment_loss is not None:
            seg_loss = (
                self.clip_alignment_loss.loss(clip_feat, batch["segment"])
                * self.hparams.loss_cfg.weights.seg_loss
            )

        caption_loss_kargs = {
            "captions": batch["caption_data"]["caption"],
            "point_indices": batch["caption_data"]["point_indices"],
            "caption_offsets": batch["caption_data"]["caption_offsets"],
            "num_points_per_caption": batch["caption_data"]["num_points_per_caption"],
            "clip_encoder": self.clip_encoder,
        }
        caption_loss = (
            self.caption_loss.loss(clip_feat, **caption_loss_kargs)
            * self.hparams.loss_cfg.weights.caption_loss
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
                * self.hparams.loss_cfg.weights.clip_image_loss
            )

        loss = binary_loss + seg_loss + caption_loss + clip_image_alignment_loss
        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        lr = self.optimizers().param_groups[0]["lr"]
        log_metrics = dict(loss=loss, caption_loss=caption_loss, lr=lr)
        if self.binary_loss is not None:
            log_metrics["binary_loss"] = binary_loss
        if self.clip_alignment_loss is not None:
            log_metrics["seg_loss"] = seg_loss
        if self.train_clip_image_alignment:
            log_metrics["clip_image_alignment_loss"] = clip_image_alignment_loss

        # useful metadata
        bs = len(batch["offset"]) - 1
        log_metrics["num_points"] = batch["coord"].shape[0] / bs
        log_metrics["num_objects"] = np.mean(
            [len(captions) for captions in batch["caption_data"]["caption"]]
        )

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
        for class_info, eval_module in zip(self.val_class_info, self.clip_alignment_eval):
            class_names = class_info["class_names"]

            if eval_module.emb_target is None:
                if self.hparams.use_prompt:
                    class_names = [
                        f"a {c} in a scene" if "other" not in c else "other" for c in class_names
                    ]  # OpenScene setting
                text_embedding = caption_utils.forward_text_encoder(
                    class_names,
                    self.clip_encoder,
                    normalize=True,
                    device=self.clip_encoder.text_projection.device,
                )
                eval_module.set_target_embedding(text_embedding.to(self.device))
            else:
                if eval_module.emb_target.device != self.device:
                    eval_module.emb_target = eval_module.emb_target.to(self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.val_metrics[dataloader_idx]
        class_info = self.val_class_info[dataloader_idx]

        out_dict = self(batch)
        logits = self.clip_alignment_eval[dataloader_idx].predict(
            out_dict["clip_feat"], return_logit=True
        )

        # 1. semantic segmentation
        preds_all = logits.max(1)[1]
        metrics["confmat_all"](preds_all, batch["segment"])

        logits_fg = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits_fg[..., class_info["fg_class_idx"]] = logits[..., class_info["fg_class_idx"]]

        if self.binary_loss is not None:
            base_scores = logits_fg[..., self.base_class_idx].softmax(dim=-1)
            novel_scores = logits_fg[..., self.novel_class_idx].softmax(dim=-1)
            scores = logits_fg.clone().float()
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
            preds = logits_fg.max(1)[1]

        segment_fg = batch["segment"].clone()
        for i in class_info["bg_class_idx"]:
            segment_fg[segment_fg == i] = class_info["ignore_label"]  # Set background classes to 0

        # update and log metrics
        metrics["confmat"](preds, segment_fg)

        # 2. instance segmentation (optional)
        if "mAP_evaluator" in metrics:
            self._update_instance_segmentation_metrics(batch, logits, metrics, class_info)

        # 3. CLIP image alignment (optional)
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
            metrics["clip_image_score"].update(clip_avg_score)

        if self.eval_clip_text_alignment:
            clip_avg_score = compute_clip_text_cosine_similarity(
                clip_encoder=self.clip_encoder,
                clip_tokenized_text=batch["clip_tokenized_text"],
                point_feat=out_dict["clip_feat"],
                offset=batch["offset"],
                point_indices_to_caption=batch["caption_data"]["idx"],
            )
            metrics["clip_text_score"].update(clip_avg_score)

    def _update_instance_segmentation_metrics(self, batch, logits, metrics, class_info):
        offset = batch["offset"]
        batch_size = len(offset) - 1
        ignore_class_idx = class_info["instance_ignore_class_idx"]
        for i in range(batch_size):
            gt_classes = batch["segment"][offset[i] : offset[i + 1]]
            gt_instances = batch["instance"][offset[i] : offset[i + 1]]
            pred_logits = logits[offset[i] : offset[i + 1]]
            pred_masks = batch["masks_binary"][i]

            # mask logits (voting)
            pred_logits_fg = pred_logits.clone()

            if self.ignore_background:
                pred_logits_fg[..., ignore_class_idx] = torch.finfo(pred_logits.dtype).min

            pred_logits_fg = torch.nn.functional.softmax(pred_logits_fg, dim=-1)
            pred_logits_fg = torch.stack([pred_logits_fg[mask].mean(dim=0) for mask in pred_masks])
            pred_scores, pred_classes = torch.max(pred_logits_fg, dim=1)

            if self.ignore_class_prob:
                pred_scores = torch.ones_like(pred_scores)

            metrics["mAP_evaluator"].update(
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                pred_masks=pred_masks,
                gt_segment=gt_classes,
                gt_instance=gt_instances,
            )

    def on_validation_epoch_end(self) -> None:
        def compute_classwise_metrics(confmat, class_names):
            computed_confmat = confmat.compute().cpu().numpy()
            class_ious = {}
            class_accs = {}
            for i, class_name in enumerate(class_names):
                tp = computed_confmat[i, i]
                fp = computed_confmat[:, i].sum() - tp
                fn = computed_confmat[i, :].sum() - tp

                class_ious[class_name] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                class_accs[class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0

            return class_ious, class_accs

        for idx, metrics in enumerate(self.val_metrics):
            class_info = self.val_class_info[idx]
            class_names = class_info["class_names"]
            postfix = class_info["postfix"]

            # 1. semantic segmentation
            class_ious, class_accs = compute_classwise_metrics(metrics["confmat"], class_names)
            class_ious_all, class_accs_all = compute_classwise_metrics(
                metrics["confmat_all"], class_names
            )

            miou = np.nanmean([class_ious[class_names[i]] for i in class_info["fg_class_idx"]])
            macc = np.nanmean([class_accs[class_names[i]] for i in class_info["fg_class_idx"]])
            miou_all = np.nanmean([class_ious_all[c] for c in class_names])
            macc_all = np.nanmean([class_accs_all[c] for c in class_names])
            metrics["miou_all_best"].update(miou_all)

            val_section = f"val_{postfix}"
            log_metrics = {f"{val_section}/iou_{k}": v for k, v in class_ious_all.items()}
            log_metrics.update(
                {
                    f"{val_section}/best_miou_all": metrics["miou_all_best"].compute(),
                    f"{val_section}/miou": miou,
                    f"{val_section}/macc": macc,
                    f"{val_section}/miou_all": miou_all,
                    f"{val_section}/macc_all": macc_all,
                }
            )

            if (
                hasattr(class_info, "base_class_idx")
                and class_info["base_class_idx"] is not None
                and len(class_info["base_class_idx"]) > 0
            ):
                base_class_idx = class_info["base_class_idx"]
                novel_class_idx = class_info["novel_class_idx"]
                miou_base = np.nanmean([class_ious[class_names[i]] for i in base_class_idx])
                miou_novel = np.nanmean([class_ious[class_names[i]] for i in novel_class_idx])
                hiou = 2 * miou_base * miou_novel / (miou_base + miou_novel + 1e-8)
                log_metrics.update(
                    {
                        f"{val_section}/miou_base": miou_base,
                        f"{val_section}/miou_novel": miou_novel,
                        f"{val_section}/hiou": hiou,
                    }
                )

            # 2. instance segmentation (optional)
            if "mAP_evaluator" in metrics:
                instance_metrics = metrics["mAP_evaluator"].compute()
                classwise_aps = {}
                for class_name, classwise_metrics in instance_metrics["classes"].items():
                    for metric_name, metric_value in classwise_metrics.items():
                        classwise_aps[f"{val_section}/{metric_name}_{class_name}"] = metric_value
                log_metrics.update(classwise_aps)
                instance_metrics.pop("classes")
                log_metrics.update({f"{val_section}/{k}": v for k, v in instance_metrics.items()})

            # 3. CLIP alignment (optional)
            if self.eval_clip_text_alignment:
                log_metrics.update(
                    {f"{val_section}/clip_text_score": metrics["clip_text_score"].compute()}
                )
            if self.eval_clip_image_alignment:
                log_metrics.update(
                    {
                        f"{val_section}/clip_image_score": metrics["clip_image_score"].compute(),
                    }
                )

            # log metrics only if not sanity checking
            if not self.trainer.sanity_checking:
                self.log_dict(log_metrics, sync_dist=True, logger=True)

            metrics["confmat"].reset()
            metrics["confmat_all"].reset()
            metrics["clip_text_score"].reset()
            metrics["clip_image_score"].reset()
            if "mAP_evaluator" in metrics:
                metrics["mAP_evaluator"].reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.validation_step(batch, batch_idx, dataloader_idx)

    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
