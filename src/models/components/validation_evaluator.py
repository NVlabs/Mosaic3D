from typing import List, Dict, Callable, Any, Optional
from jaxtyping import Float

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CLIPTextClassification:
    def __init__(
        self,
        class_names: List[str],
        device: torch.device,
        use_prompt: Optional[bool] = False,
    ):
        self.class_names = class_names
        self.device = device
        self.use_prompt = use_prompt

    @torch.inference_mode()
    def setup(self, clip_encoder: "CLIPTextEncoderInterface"):  # noqa: F821
        if self.use_prompt:
            self.class_names = [f"a {c} in a scene" for c in self.class_names]
        # Setup the clip text embedding M x C
        self.emb = clip_encoder(self.class_names)
        self.emb /= self.emb.norm(dim=-1, keepdim=True)
        self.emb = self.emb.t()  # C x M transpose to make the matmul efficient

    @torch.inference_mode()
    def __call__(self, x: Float[Tensor, "N C"]) -> Float[Tensor, "N M"]:  # noqa: F722,F821
        pred = x.to(self.emb) @ self.emb  # N x M
        return pred


class ValidationEvaluator:
    """Encapsulates the validation logic for semantic segmentation for a SINGLE dataset."""

    def __init__(
        self,
        # class_info: Dict, # Info for the specific dataset - Replaced by individual args
        use_prompt: bool,
        log_func: Callable,
        trainer: Any,
        device: torch.device,
        postfix: str,
        # Unpacked arguments from class_info:
        class_names: List[str],
        ignore_label: int,
        fg_class_idx: List[int],
        bg_class_idx: List[int],
        base_class_idx: Optional[List[int]],
        novel_class_idx: Optional[List[int]],
        subset_mapper: Optional[Dict],
        instance_ignore_class_idx: List[int],  # Keep even if unused for now
    ):
        super().__init__()
        # Store unpacked arguments
        self.clip_text_classifier = CLIPTextClassification(
            class_names=class_names,
            device=device,
            use_prompt=use_prompt,
        )
        self.log_func = log_func
        self.trainer = trainer
        self.device = device
        self.postfix = postfix
        self.class_names = class_names
        self.ignore_label = ignore_label
        self.fg_class_idx = fg_class_idx
        self.bg_class_idx = bg_class_idx
        self.base_class_idx = base_class_idx
        self.novel_class_idx = novel_class_idx
        self.subset_mapper = subset_mapper
        self.instance_ignore_class_idx = instance_ignore_class_idx

        # self.class_info = class_info # Removed
        self.val_section = f"val_{self.postfix}"  # Precompute log prefix

        self.metrics = nn.ModuleDict()
        self._setup_metrics()

    def setup(self, clip_encoder: "CLIPTextEncoderInterface"):  # noqa: F821
        self.clip_text_classifier.setup(clip_encoder)

    def _setup_metrics(self):
        # Operates directly on self attributes
        # class_names = self.class_info["class_names"]
        # ignore_label = self.class_info["ignore_label"]
        num_classes = len(self.class_names)

        # Semantic segmentation metrics (default)
        metric_dict = nn.ModuleDict(
            {
                "confmat": MulticlassConfusionMatrix(
                    num_classes=num_classes,
                    ignore_index=self.ignore_label,  # Use self.ignore_label
                ).to(self.device),
                "confmat_all": MulticlassConfusionMatrix(
                    num_classes=num_classes,
                    ignore_index=self.ignore_label,  # Use self.ignore_label
                ).to(self.device),
            }
        )

        # Instance segmentation metrics (optional - adapt if needed)
        # if self.subset_mapper is not None: # Example check based on available attributes
        #     metric_dict["mAP_evaluator"] = InstanceSegmentationEvaluator(...)

        self.metrics = metric_dict

    def reset(self):
        """Resets all metrics for this evaluator."""
        # No loop needed, operates on self.metrics directly
        for key in self.metrics.keys():
            self.metrics[key].reset()

    def update(self, batch: Dict, out_dict: Dict):
        """Updates metrics with results from a validation batch for this dataset."""
        # No dataloader_idx or postfix lookup needed
        metrics = self.metrics
        # class_info = self.class_info # Removed
        classifier = self.clip_text_classifier

        # Ensure classifier components are on the right device if needed
        # (Assuming Lightning handles device placement for nn.Modules)

        clip_feat = out_dict["clip_feat"]
        logits = classifier(clip_feat.to(self.device))

        segment_gt = batch["segment"].to(self.device)

        # 1. Semantic Segmentation (All Classes)
        preds_all = logits.max(1)[1]
        metrics["confmat_all"](preds_all, segment_gt)

        # 2. Semantic Segmentation (Foreground Classes Only)
        logits_fg = torch.full_like(logits, torch.finfo(logits.dtype).min)
        # fg_class_idx = class_info.get("fg_class_idx") # Use self.fg_class_idx
        fg_class_idx = self.fg_class_idx

        if fg_class_idx is not None and len(fg_class_idx) > 0:
            mask = torch.zeros_like(logits, dtype=torch.bool, device=self.device)
            fg_indices_tensor = torch.tensor(fg_class_idx, device=self.device, dtype=torch.long)
            # Handle potential empty fg_indices_tensor if fg_class_idx was empty after filtering
            if fg_indices_tensor.numel() > 0:
                mask.scatter_(1, fg_indices_tensor.unsqueeze(0).expand(logits.size(0), -1), True)
                logits_fg = torch.where(mask, logits, logits_fg)
                preds = logits_fg.max(1)[1]

                segment_fg = segment_gt.clone()
                # bg_class_idx = class_info.get("bg_class_idx", []) # Use self.bg_class_idx
                # ignore_label = class_info.get("ignore_label", -100) # Use self.ignore_label
                bg_class_idx = self.bg_class_idx
                ignore_label = self.ignore_label
                if ignore_label is None:
                    ignore_label = -100

                for i in bg_class_idx:
                    segment_fg[segment_fg == i] = ignore_label

                metrics["confmat"](preds, segment_fg)
            else:
                log.warning(
                    f"[{self.postfix}] fg_class_idx resulted in empty tensor. Skipping foreground metric update."
                )

        else:
            log.warning(
                f"[{self.postfix}] fg_class_idx not found or empty. Skipping foreground metric calculation."
            )

    def _compute_classwise_metrics(self, confmat, class_names):
        """Computes IoU and Accuracy per class from a confusion matrix."""
        computed_confmat = confmat.compute()
        if computed_confmat.is_cuda:
            computed_confmat = computed_confmat.cpu()
        computed_confmat = computed_confmat.numpy()

        class_ious = {}
        class_accs = {}
        num_classes = len(class_names)

        if computed_confmat.shape != (num_classes, num_classes):
            log.error(
                f"[{self.postfix}] Computed confusion matrix shape {computed_confmat.shape} does not match num_classes {num_classes}. Skipping metric calculation."
            )
            return {}, {}

        for i, class_name in enumerate(class_names):
            tp = computed_confmat[i, i]
            fp = computed_confmat[:, i].sum() - tp
            fn = computed_confmat[i, :].sum() - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_ious[class_name] = float(np.nan_to_num(iou))
            class_accs[class_name] = float(np.nan_to_num(acc))

        return class_ious, class_accs

    def compute(self) -> Dict[str, float]:
        """Computes final validation metrics for THIS dataset and returns them."""
        # No loop over postfix needed
        # val_section prefix is now precomputed in __init__
        # info = self.class_info # Removed
        # class_names = info["class_names"] # Use self.class_names
        class_names = self.class_names
        metrics_for_postfix = self.metrics
        current_log_metrics = {}  # Metrics for this specific dataset/postfix

        # --- Semantic Segmentation Metrics ---
        class_ious = {}
        class_accs = {}
        if "confmat" in metrics_for_postfix:
            class_ious, class_accs = self._compute_classwise_metrics(
                metrics_for_postfix["confmat"], class_names
            )

        class_ious_all, class_accs_all = self._compute_classwise_metrics(
            metrics_for_postfix["confmat_all"], class_names
        )

        # fg_class_idx = info.get("fg_class_idx", []) # Use self.fg_class_idx
        fg_class_idx = self.fg_class_idx
        miou = np.nanmean(
            [
                class_ious[class_names[i]]
                for i in fg_class_idx
                if i < len(class_names) and class_names[i] in class_ious
            ]
            + [np.nan]
        )
        macc = np.nanmean(
            [
                class_accs[class_names[i]]
                for i in fg_class_idx
                if i < len(class_names) and class_names[i] in class_accs
            ]
            + [np.nan]
        )
        miou_all = np.nanmean(
            [class_ious_all[c] for c in class_names if c in class_ious_all] + [np.nan]
        )
        macc_all = np.nanmean(
            [class_accs_all[c] for c in class_names if c in class_accs_all] + [np.nan]
        )

        miou = float(np.nan_to_num(miou))
        macc = float(np.nan_to_num(macc))
        miou_all = float(np.nan_to_num(miou_all))
        macc_all = float(np.nan_to_num(macc_all))

        current_log_metrics.update(
            {f"{self.val_section}/iou_{k}": v for k, v in class_ious_all.items()}
        )
        current_log_metrics.update(
            {
                f"{self.val_section}/miou": miou,
                f"{self.val_section}/macc": macc,
                f"{self.val_section}/miou_all": miou_all,
                f"{self.val_section}/macc_all": macc_all,
            }
        )

        # subset_mapper = info.get("subset_mapper") # Use self.subset_mapper
        subset_mapper = self.subset_mapper
        if subset_mapper is not None:
            subset_names = subset_mapper.get("subset_names", [])
            for subset_name in subset_names:
                subset_class_indices = [
                    i
                    for i, name in enumerate(class_names)
                    if subset_mapper.get(name) == subset_name
                ]
                subset_miou = np.nanmean(
                    [
                        class_ious[class_names[i]]
                        for i in subset_class_indices
                        if i < len(class_names) and class_names[i] in class_ious
                    ]
                    + [np.nan]
                )
                subset_macc = np.nanmean(
                    [
                        class_accs[class_names[i]]
                        for i in subset_class_indices
                        if i < len(class_names) and class_names[i] in class_accs
                    ]
                    + [np.nan]
                )
                current_log_metrics[f"{self.val_section}/miou_{subset_name}"] = float(
                    np.nan_to_num(subset_miou)
                )
                current_log_metrics[f"{self.val_section}/macc_{subset_name}"] = float(
                    np.nan_to_num(subset_macc)
                )

        # base_class_idx = info.get("base_class_idx") # Use self.base_class_idx
        # novel_class_idx = info.get("novel_class_idx") # Use self.novel_class_idx
        base_class_idx = self.base_class_idx
        novel_class_idx = self.novel_class_idx
        if (
            base_class_idx is not None
            and novel_class_idx is not None
            and len(base_class_idx) > 0
            and len(novel_class_idx) > 0
        ):
            miou_base = np.nanmean(
                [
                    class_ious[class_names[i]]
                    for i in base_class_idx
                    if i < len(class_names) and class_names[i] in class_ious
                ]
                + [np.nan]
            )
            miou_novel = np.nanmean(
                [
                    class_ious[class_names[i]]
                    for i in novel_class_idx
                    if i < len(class_names) and class_names[i] in class_ious
                ]
                + [np.nan]
            )
            miou_base = float(np.nan_to_num(miou_base))
            miou_novel = float(np.nan_to_num(miou_novel))
            hiou = (
                (2 * miou_base * miou_novel / (miou_base + miou_novel + 1e-8))
                if (miou_base + miou_novel) > 0
                else 0.0
            )

            current_log_metrics.update(
                {
                    f"{self.val_section}/miou_base": miou_base,
                    f"{self.val_section}/miou_novel": miou_novel,
                    f"{self.val_section}/hiou": hiou,
                }
            )

        # --- Instance Segmentation Metrics (if applicable) ---
        # if "mAP_evaluator" in metrics_for_postfix:
        #     instance_metrics = metrics_for_postfix["mAP_evaluator"].compute()
        #     # Process and add instance metrics...

        # --- Final Logging and Best Metric Update --- Removed
        # This evaluator doesn't log directly or track best metric anymore

        # Return the computed metrics for this dataset
        # Ensure tensors are converted to floats/primitives before returning
        metrics_serializable = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in current_log_metrics.items()
        }
        return metrics_serializable
