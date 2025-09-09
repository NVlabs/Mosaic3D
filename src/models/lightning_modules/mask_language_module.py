import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from cuml.cluster import DBSCAN
from einops import repeat
from torchmetrics import MaxMetric, MeanMetric

import src.utils.caption_utils as caption_utils
from src.models.lightning_modules.module_base import LitModuleBase
from src.models.losses.clip_alignment_loss import CLIPAlignmentEval
from src.models.losses.mask_caption_loss import (
    MaskCaptionAlignmentLoss,
    MaskCaptionCLIPLoss,
    MaskCaptionLoss,
    MaskCaptionSigLIPLoss,
)
from src.models.utils.clip_models import build_clip_model, download_clip_model
from src.models.utils.evaluator import InstanceSegmentationEvaluator
from src.utils import RankedLogger
from src.utils.dist_utils import all_gather, all_gather_different_shapes

log = RankedLogger(__file__, rank_zero_only=True)


class MaskLanguageLitModule(LitModuleBase):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        clip_encoder: Dict,
        compile: bool,
        loss_cfg: Dict,
        best_metric: str,
        eval_cfg: Optional[Dict] = None,
        use_prompt: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        # segmentation loss
        self.seg_loss = loss_cfg["seg_loss"]
        self.seg_loss_weights = dict(
            loss_ce=self.seg_loss.matcher.cost_class,
            loss_mask=self.seg_loss.matcher.cost_mask,
            loss_dice=self.seg_loss.matcher.cost_dice,
        )

        # caption loss
        self.caption_loss_type = loss_cfg["caption_loss"]["type"]
        if self.caption_loss_type == "contrastive":
            self.caption_loss = MaskCaptionLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "alignment":
            self.caption_loss = MaskCaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "region_alignment":
            self.caption_loss = MaskCaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "clip":
            self.caption_loss = MaskCaptionCLIPLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "siglip":
            self.caption_loss = MaskCaptionSigLIPLoss(**loss_cfg["caption_loss"])
        else:
            raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metrics = nn.ModuleDict()
        self.val_class_info = dict()
        self.val_dataset_names = dict()
        self.val_best_metric = MaxMetric()

        # Sync distributed metrics
        self.train_sync_dist = loss_cfg.get("sync_dist", False)

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

        for i, val_dataloader in enumerate(val_dataloaders):
            dataset = val_dataloader.dataset
            dataset_name = dataset.dataset_name
            class_names = dataset.CLASS_LABELS
            postfix = dataset.log_postfix
            assert postfix is not None, "log_postfix is required for clarity"

            metric = nn.ModuleDict(
                {
                    "map_evaluator": InstanceSegmentationEvaluator(
                        class_names=class_names,
                        segment_ignore_index=dataset.instance_ignore_class_idx
                        + [dataset.ignore_label],
                        instance_ignore_index=dataset.ignore_label,
                        subset_mapper=dataset.subset_mapper,
                        sync_on_compute=False,
                    ),
                }
            )

            class_info = dict(
                postfix=postfix,
                class_names=class_names,
                fg_class_idx=dataset.fg_class_idx,
                bg_class_idx=dataset.bg_class_idx,
                ignore_label=dataset.ignore_label,
                instance_ignore_class_idx=dataset.instance_ignore_class_idx,
                subset_mapper=dataset.subset_mapper if hasattr(dataset, "subset_mapper") else None,
            )
            self.val_metrics[postfix] = metric
            self.val_class_info[postfix] = class_info
            self.val_dataset_names[i] = postfix

        self.clip_alignment_eval = nn.ModuleDict(
            {
                postfix: CLIPAlignmentEval(**self.hparams.eval_cfg.seg_eval)
                for postfix in self.val_metrics.keys()
            }
        )

    def forward(self, batch: Any, num_queries: Optional[int] = None) -> Dict[str, Any]:
        output = self.net(batch, num_queries)
        out_dict = self._output_to_dict(output, batch)
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
        seg_loss, caption_loss = 0, 0

        # segmentation loss with Hungarian matching
        caption_data = batch["caption_data"]
        seg_losses, mapping = self.seg_loss(out_dict, caption_data, return_indices=True)
        seg_loss = (
            sum(seg_losses[k] * self.seg_loss_weights[k] for k in seg_losses.keys())
            * self.hparams.loss_cfg.weights.seg_loss
        )

        # caption loss
        matched_mask_features = []
        matched_captions = [] if "caption" in caption_data[0] else None
        matched_embeddings = [] if "embedding" in caption_data[0] else None

        for mask_features, caption_datum, indices in zip(
            out_dict["clip_feat"], caption_data, mapping
        ):
            src_idx, trg_idx = indices
            matched_mask_features.append(mask_features[src_idx])
            if "caption" in caption_datum:
                matched_captions.append([caption_datum["caption"][i] for i in trg_idx])
            if "embedding" in caption_datum:
                matched_embeddings.append([caption_datum["embedding"][i] for i in trg_idx])

        matched_mask_features = torch.cat(matched_mask_features)
        caption_loss = (
            self.caption_loss.loss(
                matched_mask_features,
                matched_captions,
                self.clip_encoder,
                matched_embeddings,
            )
            * self.hparams.loss_cfg.weights.caption_loss
        )

        # total loss
        loss = seg_loss + caption_loss
        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        lr = self.optimizers().param_groups[0]["lr"]
        log_metrics = dict(loss=loss, seg_loss=seg_loss, caption_loss=caption_loss, lr=lr)

        # useful metadata
        bs = len(batch["offset"]) - 1
        log_metrics["num_points"] = batch["coord"].shape[0] / bs
        log_metrics["num_objects"] = np.mean([x["num_captions"] for x in caption_data])

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
        self.clip_encoder = self.clip_encoder.to(self.device)
        for postfix in self.val_class_info.keys():
            class_info = self.val_class_info[postfix]
            eval_module = self.clip_alignment_eval[postfix]
            class_names = class_info["class_names"]

            if eval_module.emb_target is None:
                if self.hparams.use_prompt:
                    class_names = [
                        f"a {c} in a scene" if "other" not in c else "other" for c in class_names
                    ]  # OpenScene setting
                text_embedding = caption_utils.forward_text_encoder(
                    class_names, self.clip_encoder, normalize=True, device=self.device
                )
                eval_module.set_target_embedding(text_embedding.to(self.device))
            else:
                if eval_module.emb_target.device != self.device:
                    eval_module.emb_target = eval_module.emb_target.to(self.device)

            # reset metrics
            metrics = self.val_metrics[postfix]
            for key in metrics.keys():
                metrics[key].reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        postfix = self.val_dataset_names[dataloader_idx]
        metrics = self.val_metrics[postfix]
        clip_evaluator = self.clip_alignment_eval[postfix]

        # Inference
        out_dict = self(batch)
        batch_binary_logits = out_dict["logit"]  # [B, Q, 2]
        batch_masks = out_dict["mask"]  # List[Tensor[N, Q]]
        batch_clip_feats = out_dict["clip_feat"]  # [B, Q, D]

        offset = batch["offset"]
        batch_size = len(offset) - 1
        for i in range(batch_size):
            gt_classes = batch["segment"][offset[i] : offset[i + 1]]  # [N]
            gt_instances = batch["instance"][offset[i] : offset[i + 1]]  # [N]

            clip_feats = batch_clip_feats[i]  # [Q, D]
            mask_binary_probs = batch_binary_logits[i].softmax(dim=-1)[:, :1]  # [Q, 1]
            mask_logits = clip_evaluator.predict(clip_feats, return_logit=True)  # [Q, C]
            mask_probs = nn.functional.softmax(mask_logits, dim=-1)  # [Q, C]
            mask_probs = mask_binary_probs * mask_probs  # [Q, C]
            masks = batch_masks[i]  # [N, Q]

            heatmap = masks.sigmoid()  # [N, Q]
            masks = (masks.T > 0).float()  # [Q, N]
            mask_scores_per_image = (heatmap.T * masks).sum(1, keepdim=True) / (
                masks.sum(1, keepdim=True) + 1e-6
            )  # [Q, 1]
            mask_probs = mask_scores_per_image * mask_probs  # [Q, C]
            scores, classes = mask_probs.max(1)  # [Q]

            if dist.is_initialized() and dist.get_world_size() > 1:
                gathered_classes = all_gather(classes)
                gathered_scores = all_gather(scores)
                gathered_masks = all_gather_different_shapes(masks)
                gathered_gt_classes = all_gather_different_shapes(gt_classes)
                gathered_gt_instances = all_gather_different_shapes(gt_instances)
            else:
                gathered_classes = [classes]
                gathered_scores = [scores]
                gathered_masks = [masks]
                gathered_gt_classes = [gt_classes]
                gathered_gt_instances = [gt_instances]

            if not self.trainer.is_global_zero:
                continue

            for classes, scores, masks, gt_classes, gt_instances in zip(
                gathered_classes,
                gathered_scores,
                gathered_masks,
                gathered_gt_classes,
                gathered_gt_instances,
            ):
                metrics["map_evaluator"].update(
                    pred_classes=classes.cpu(),
                    pred_scores=scores.cpu(),
                    pred_masks=masks.cpu(),
                    gt_segment=gt_classes.cpu(),
                    gt_instance=gt_instances.cpu(),
                )

    def on_validation_epoch_end(self) -> None:
        log_metrics = {}
        for postfix, metrics in self.val_metrics.items():
            val_section = f"val_{postfix}"
            logging_keys = [
                "map",
                "map50",
                "map25",
                "map_head",
                "map_common",
                "map_tail",
            ]

            # Only compute metrics on rank 0
            ap_tensor = torch.zeros(len(logging_keys), device=self.device)
            if self.trainer.is_global_zero:
                ap_results = metrics["map_evaluator"].compute()
                for i, key in enumerate(logging_keys):
                    ap_tensor[i] = ap_results.get(key, 0.0)

            # Broadcast ap_results to all other ranks
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast(ap_tensor, src=0)

            log_metrics.update(
                {f"{val_section}/{key}": ap_tensor[i].item() for i, key in enumerate(logging_keys)}
            )

        # Update best metric
        self.val_best_metric.update(log_metrics[self.hparams.best_metric])
        log_metrics[f"{self.hparams.best_metric}_best"] = self.val_best_metric.compute()

        # Log metrics if not in sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(log_metrics, sync_dist=True, logger=True)

    def _apply_dbscan(self, mask_probs: torch.Tensor, masks: torch.Tensor, coord: torch.Tensor):
        # DBSCAN parameters
        eps = self.hparams.eval_cfg.post_processing.dbscan_eps
        min_points = self.hparams.eval_cfg.post_processing.dbscan_min_points
        remove_small_group = self.hparams.eval_cfg.post_processing.remove_small_group

        new_mask_probs = []
        new_masks = []
        for query_idx in range(masks.shape[1]):
            mask = masks[:, query_idx]  # [N]
            binary_mask = mask > 0
            points = coord[binary_mask]

            if len(points) == 0:
                continue

            # Run DBSCAN clustering on points
            clusters = DBSCAN(eps=eps, min_samples=min_points, verbose=2).fit(points).labels_
            clusters = torch.tensor(clusters.get(), dtype=torch.long, device=masks.device)

            # Map clusters back to original point indices
            cluster_map = torch.zeros_like(binary_mask, dtype=torch.long)
            cluster_map[binary_mask] = clusters + 1

            # Process each cluster
            for cluster_id in torch.unique(clusters):
                if cluster_id == -1:
                    continue

                cluster_mask = cluster_map == cluster_id + 1
                if cluster_mask.sum() > remove_small_group:
                    new_mask_probs.append(mask_probs[query_idx])
                    new_masks.append(mask * cluster_mask)

        if not new_masks:
            return mask_probs, masks

        return torch.stack(new_mask_probs), torch.stack(new_masks, dim=1)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        postfix = self.val_dataset_names[dataloader_idx]
        metrics = self.val_metrics[postfix]
        clip_evaluator = self.clip_alignment_eval[postfix]

        # Inference
        out_dict = self(batch, self.hparams.eval_cfg.num_queries)
        batch_binary_logits = out_dict["logit"]  # [B, Q, 2]
        batch_masks = out_dict["mask"]  # List[Tensor[N, Q]]
        batch_clip_feats = out_dict["clip_feat"]  # [B, Q, D]

        offset = batch["offset"]
        coord = batch["coord"]
        batch_size = len(offset) - 1
        for i in range(batch_size):
            gt_classes = batch["segment"][offset[i] : offset[i + 1]]  # [N]
            gt_instances = batch["instance"][offset[i] : offset[i + 1]]  # [N]

            clip_feats = batch_clip_feats[i]  # [Q, D]
            masks = batch_masks[i]  # [N, Q]
            mask_binary_probs = batch_binary_logits[i].softmax(dim=-1)[:, :1]  # [Q, 1]
            mask_logits = clip_evaluator.predict(clip_feats, return_logit=True)  # [Q, C]
            mask_probs = nn.functional.softmax(mask_logits, dim=-1)  # [Q, C]
            mask_probs = mask_binary_probs * mask_probs  # [Q, C]

            # DBSCAN post-processing
            if self.hparams.eval_cfg.post_processing.use_dbscan:
                mask_probs, masks = self._apply_dbscan(
                    mask_probs, masks, coord[offset[i] : offset[i + 1]]
                )

            heatmap = masks.sigmoid()
            masks = (masks.T > 0).float()
            mask_scores_per_image = (heatmap.T * masks).sum(1) / (masks.sum(1) + 1e-6)  # [Q]

            if self.hparams.eval_cfg.post_processing.topk_per_image > 0:
                num_queries, num_classes = mask_probs.shape
                classes = repeat(
                    torch.arange(num_classes, device=mask_probs.device),
                    "c -> (q c)",
                    q=num_queries,
                )
                scores, topk_indices = mask_probs.flatten().topk(
                    self.hparams.eval_cfg.post_processing.topk_per_image, sorted=True
                )
                topk_indices = topk_indices // num_classes
                masks = masks[topk_indices]
                classes = classes[topk_indices]
                scores = mask_scores_per_image[topk_indices] * scores
            else:
                scores, classes = mask_probs.max(1)
                scores = mask_scores_per_image * scores

            metrics["map_evaluator"].update(
                pred_classes=classes.cpu(),
                pred_scores=scores.cpu(),
                pred_masks=masks.cpu(),
                gt_segment=gt_classes.cpu(),
                gt_instance=gt_instances.cpu(),
            )

    def on_test_epoch_end(self) -> None:
        log_metrics = {}
        for postfix, metrics in self.val_metrics.items():
            test_section = f"test_{postfix}"
            ap_results = metrics["map_evaluator"].compute()
            # Log class-wise metrics
            for class_name, class_metrics in ap_results["classes"].items():
                log_metrics.update(
                    {f"{test_section}/{k}_{class_name}": v for k, v in class_metrics.items()}
                )

            # Log overall metrics
            ap_results.pop("classes")
            log_metrics.update({f"{test_section}/{k}": v for k, v in ap_results.items()})

        # Log metrics if not in sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(log_metrics, logger=True)

    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
