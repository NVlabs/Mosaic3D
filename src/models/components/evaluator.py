from typing import Dict, List

import numpy as np
import torch
from torchmetrics import Metric

from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class InstanceSegmentationEvaluator(Metric):
    def __init__(
        self,
        class_names: List[str],
        segment_ignore_index: List[int],
        instance_ignore_index: int,
        min_region_size: int = 100,
        distance_thresh: float = float("inf"),
        distance_conf: float = -float("inf"),
        subset_mapper: Dict[str, str] = None,
    ):
        super().__init__()
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.valid_class_names = [
            class_names[i] for i in range(self.num_classes) if i not in segment_ignore_index
        ]
        # Multiple IoU thresholds (0.5~0.9, 0.25)
        self.overlaps = [round(o, 2) for o in list(np.arange(0.5, 0.95, 0.05)) + [0.25]]
        self.min_region_size = min_region_size
        self.distance_thresh = distance_thresh
        self.distance_conf = distance_conf
        self.subset_mapper = subset_mapper

        # Accumulated states: use flat lists for each class and threshold
        self.add_state(
            "num_scenes", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum"
        )
        for class_name in self.valid_class_names:
            for ov in self.overlaps:
                # For scores
                self.add_state(f"scores_{class_name}_{ov}", default=[], dist_reduce_fx="cat")
                # For labels (TP/FP)
                self.add_state(f"labels_{class_name}_{ov}", default=[], dist_reduce_fx="cat")

            # For GT counts
            self.add_state(
                f"gt_count_{class_name}",
                default=torch.tensor(0, dtype=torch.int),
                dist_reduce_fx="sum",
            )

    def update(
        self,
        pred_classes: torch.Tensor,  # shape: [num_preds]
        pred_scores: torch.Tensor,  # shape: [num_preds]
        pred_masks: torch.Tensor,  # shape: [num_preds, num_vertices]
        gt_segment: torch.Tensor,  # shape: [num_vertices]
        gt_instance: torch.Tensor,  # shape: [num_vertices]
    ):
        """
        For a single scene, extract GT/predicted instances and accumulate
        (confidence, TP/FP) results and GT counts for each class.
        """
        device = gt_segment.device

        # --- Process GT instances ---
        # Extract unique GT instance ids with their indices and counts
        np_instance = gt_instance.cpu().numpy()
        np_instance_ids, np_indices, np_counts = np.unique(
            np_instance, return_index=True, return_counts=True
        )
        instance_ids = torch.from_numpy(np_instance_ids).to(device)
        indices = torch.from_numpy(np_indices).to(device)
        counts = torch.from_numpy(np_counts).to(device)
        gt_instance_info = []
        for i, inst_id in enumerate(instance_ids):
            if inst_id.item() == self.instance_ignore_index:
                continue
            seg_id = gt_segment[indices[i]].item()
            if seg_id in self.segment_ignore_index:
                continue
            # Create binary mask for this instance
            mask = gt_instance == inst_id
            vert_count = counts[i].item()
            if vert_count < self.min_region_size:
                continue
            # Distance-related conditions pass by default
            gt_instance_info.append(
                {
                    "instance_id": inst_id.item(),
                    "segment_id": seg_id,
                    "mask": mask,  # boolean tensor
                    "vert_count": vert_count,
                }
            )
        # Group GT by class
        gt_by_class = {c: [] for c in self.valid_class_names}
        for gt in gt_instance_info:
            class_name = self.class_names[gt["segment_id"]]
            if class_name in gt_by_class:
                gt_by_class[class_name].append(gt)

        # Accumulate GT counts by class
        for class_name in self.valid_class_names:
            gt_count = len(gt_by_class[class_name])
            gt_count_attr = getattr(self, f"gt_count_{class_name}")
            gt_count_attr += torch.tensor(gt_count, device=device)
            setattr(self, f"gt_count_{class_name}", gt_count_attr)

        # --- Process predicted instances ---
        pred_instance_info = []
        num_preds = pred_classes.shape[0]
        for i in range(num_preds):
            cls = pred_classes[i].item()
            if cls in self.segment_ignore_index:
                continue
            mask = pred_masks[i] != 0
            vert_count = torch.count_nonzero(mask).item()
            if vert_count < self.min_region_size:
                continue
            confidence = pred_scores[i].item()
            pred_instance_info.append(
                {
                    "instance_id": i,
                    "segment_id": cls,
                    "mask": mask,  # boolean tensor
                    "vert_count": vert_count,
                    "confidence": confidence,
                }
            )
        # Group predictions by class
        preds_by_class = {c: [] for c in self.valid_class_names}
        for pred in pred_instance_info:
            class_name = self.class_names[pred["segment_id"]]
            if class_name in preds_by_class:
                preds_by_class[class_name].append(pred)

        # --- Calculate IoU matrix for each class and perform greedy matching for each threshold ---
        for class_name in self.valid_class_names:
            gt_list = gt_by_class[class_name]
            pred_list = preds_by_class[class_name]

            # If there are no GTs for this class, all predictions are FPs
            if len(gt_list) == 0:
                for ov in self.overlaps:
                    if len(pred_list) > 0:
                        scores = torch.tensor(
                            [pred["confidence"] for pred in pred_list], device=device
                        )
                        labels = torch.zeros(len(pred_list), dtype=torch.int, device=device)

                        # Update state variables
                        scores_attr = getattr(self, f"scores_{class_name}_{ov}")
                        scores_attr.append(scores)
                        setattr(self, f"scores_{class_name}_{ov}", scores_attr)

                        labels_attr = getattr(self, f"labels_{class_name}_{ov}")
                        labels_attr.append(labels)
                        setattr(self, f"labels_{class_name}_{ov}", labels_attr)
                continue

            num_preds_cls = len(pred_list)
            num_gt_cls = len(gt_list)
            # IoU matrix: [num_preds, num_gt]
            iou_matrix = torch.zeros((num_preds_cls, num_gt_cls), device=device)
            for p_idx, pred in enumerate(pred_list):
                pred_mask = pred["mask"]
                for g_idx, gt in enumerate(gt_list):
                    gt_mask = gt["mask"]
                    intersection = torch.count_nonzero(pred_mask & gt_mask).float()
                    union = torch.count_nonzero(pred_mask | gt_mask).float()
                    iou = (intersection / union).item() if union > 0 else 0.0
                    iou_matrix[p_idx, g_idx] = iou

            # Perform matching for each overlap threshold
            for ov in self.overlaps:
                # Sort by confidence in descending order
                sorted_indices = sorted(
                    range(num_preds_cls), key=lambda i: pred_list[i]["confidence"], reverse=True
                )
                gt_matched = [False] * num_gt_cls

                # Collect scores and labels for this class and threshold
                scores_list = []
                labels_list = []

                for idx in sorted_indices:
                    score = pred_list[idx]["confidence"]
                    scores_list.append(score)

                    # IoU values between this prediction and all GTs
                    ious = iou_matrix[idx, :]
                    max_iou, max_idx = torch.max(ious, dim=0)
                    if max_iou >= ov and not gt_matched[max_idx]:
                        # True Positive case
                        labels_list.append(1)
                        gt_matched[max_idx] = True
                    else:
                        # False Positive case
                        labels_list.append(0)

                # Convert to tensors and update state
                if scores_list:
                    scores = torch.tensor(scores_list, device=device)
                    labels = torch.tensor(labels_list, dtype=torch.int, device=device)

                    # Update state variables
                    scores_attr = getattr(self, f"scores_{class_name}_{ov}")
                    scores_attr.append(scores)
                    setattr(self, f"scores_{class_name}_{ov}", scores_attr)

                    labels_attr = getattr(self, f"labels_{class_name}_{ov}")
                    labels_attr.append(labels)
                    setattr(self, f"labels_{class_name}_{ov}", labels_attr)

        self.num_scenes += torch.tensor(1, device=device)

    def compute(self):
        """
        Calculate class-wise and overall AP (mAP) based on accumulated detection results.
        (Using simple trapezoidal integration method)
        """
        log.info(f"Evaluating instance segmentation results for {self.num_scenes} scenes...")
        ap_table = {class_name: {} for class_name in self.valid_class_names}

        for class_name in self.valid_class_names:
            gt_count = getattr(self, f"gt_count_{class_name}").item()

            for ov in self.overlaps:
                scores = getattr(self, f"scores_{class_name}_{ov}")
                labels = getattr(self, f"labels_{class_name}_{ov}")

                if len(scores) == 0:
                    ap = float("nan") if gt_count == 0 else 0.0
                    ap_table[class_name][ov] = ap
                    continue

                sorted_indices = torch.argsort(scores, descending=True)
                labels_sorted = labels[sorted_indices]

                tp = torch.cumsum(labels_sorted, dim=0)
                fp = torch.cumsum(1 - labels_sorted, dim=0)
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (gt_count + 1e-6)

                # Add sentinel values
                precision = torch.cat(
                    [
                        torch.tensor([0.0], device=self.device),
                        precision,
                        torch.tensor([0.0], device=self.device),
                    ]
                )
                recall = torch.cat(
                    [
                        torch.tensor([0.0], device=self.device),
                        recall,
                        torch.tensor([1.0], device=self.device),
                    ]
                )

                # Make precision non-increasing
                for i in range(len(precision) - 1, 0, -1):
                    precision[i - 1] = max(precision[i - 1], precision[i])

                # Calculate AP (trapezoidal integration)
                ap = 0.0
                for i in range(len(precision) - 1):
                    ap += (recall[i + 1] - recall[i]) * precision[i + 1]
                ap_table[class_name][ov] = ap

        # Class-wise AP (separating threshold 0.25 and 0.5, averaging the rest)
        ap_values = []
        ap50_values = []
        ap25_values = []
        class_ap = {}
        for class_name in self.valid_class_names:
            aps = []
            aps50 = None
            aps25 = None
            for ov in self.overlaps:
                if ov == 0.25:
                    aps25 = ap_table[class_name][ov]
                else:
                    aps.append(ap_table[class_name][ov])
                    if ov == 0.5:
                        aps50 = ap_table[class_name][ov]
            mean_ap = sum(aps) / len(aps) if len(aps) > 0 else float("nan")
            ap_values.append(mean_ap)
            ap50_values.append(aps50 if aps50 is not None else float("nan"))
            ap25_values.append(aps25 if aps25 is not None else float("nan"))
            class_ap[class_name] = {"ap": mean_ap, "ap50": aps50, "ap25": aps25}

        map_all = sum(ap_values) / len(ap_values) if len(ap_values) > 0 else float("nan")
        map50_all = sum(ap50_values) / len(ap50_values) if len(ap50_values) > 0 else float("nan")
        map25_all = sum(ap25_values) / len(ap25_values) if len(ap25_values) > 0 else float("nan")
        result = {"map": map_all, "map50": map50_all, "map25": map25_all, "classes": class_ap}

        if self.subset_mapper is not None:
            for subset_name in self.subset_mapper["subset_names"]:
                result[f"map_{subset_name}"] = []
            for class_name in self.valid_class_names:
                subset_name = self.subset_mapper[class_name]
                result[f"map_{subset_name}"].append(class_ap[class_name]["ap"])
            for subset_name in self.subset_mapper["subset_names"]:
                result[f"map_{subset_name}"] = sum(result[f"map_{subset_name}"]) / len(
                    result[f"map_{subset_name}"]
                )
        return result
