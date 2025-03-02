# TorchMetrics implementation of the ScanNet instance segmentation evaluation metric
# https://github.com/Pointcept/Pointcept/blob/main/pointcept/engines/hooks/evaluator.py#L205

from typing import Dict, List
from uuid import uuid4

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
        sync_on_compute: bool = True,
    ):
        super().__init__(sync_on_compute=sync_on_compute)
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.valid_class_names = [
            class_names[i] for i in range(self.num_classes) if i not in segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_size = min_region_size
        self.distance_thresh = distance_thresh
        self.distance_conf = distance_conf
        self.subset_mapper = subset_mapper

        self.add_state("pred_classes", default=[])
        self.add_state("pred_scores", default=[])
        self.add_state("pred_masks", default=[])
        self.add_state("gt_segment", default=[])
        self.add_state("gt_instance", default=[])

    def update(
        self,
        pred_classes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_segment: torch.Tensor,
        gt_instance: torch.Tensor,
    ):
        self.pred_classes.append(pred_classes)
        self.pred_scores.append(pred_scores)
        self.pred_masks.append(pred_masks)
        self.gt_segment.append(gt_segment)
        self.gt_instance.append(gt_instance)

    def compute(self):
        log.info(
            f"Computing instance segmentation evaluation with {len(self.pred_classes)} predictions"
        )
        scenes = []
        for i in range(len(self.pred_classes)):
            gt_instances, pred_instances = self.associate_instances(
                {
                    "pred_classes": self.pred_classes[i].cpu().numpy(),
                    "pred_scores": self.pred_scores[i].cpu().numpy(),
                    "pred_masks": self.pred_masks[i].cpu().numpy(),
                },
                self.gt_segment[i].cpu().numpy(),
                self.gt_instance[i].cpu().numpy(),
            )
            scenes.append({"gt": gt_instances, "pred": pred_instances})
        results = self.evaluate_matches(scenes)
        log.info(f">>> mAP: {results['map']}")
        return results

    def associate_instances(
        self, pred: Dict[str, np.ndarray], segment: np.ndarray, instance: np.ndarray
    ):
        void_mask = np.isin(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]

        gt_instances = {name: [] for name in self.valid_class_names}
        instance_ids, idx, counts = np.unique(instance, return_index=True, return_counts=True)
        segment_ids = segment[idx]

        for i in range(len(instance_ids)):
            if (
                instance_ids[i] == self.instance_ignore_index
                or segment_ids[i] in self.segment_ignore_index
            ):
                continue
            gt_inst = {
                "instance_id": instance_ids[i],
                "segment_id": segment_ids[i],
                "dist_conf": 0.0,
                "med_dist": -1.0,
                "vert_count": counts[i],
                "matched_pred": [],
            }
            gt_instances[self.class_names[segment_ids[i]]].append(gt_inst)

        pred_instances = {name: [] for name in self.valid_class_names}
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = {
                "uuid": uuid4(),
                "instance_id": i,
                "segment_id": pred["pred_classes"][i],
                "confidence": pred["pred_scores"][i],
                "mask": pred["pred_masks"][i] != 0,
                "vert_count": np.count_nonzero(pred["pred_masks"][i] != 0),
                "void_intersection": np.count_nonzero(
                    np.logical_and(void_mask, pred["pred_masks"][i] != 0)
                ),
            }
            if pred_inst["vert_count"] < self.min_region_size:
                continue
            segment_name = self.class_names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_inst in gt_instances[segment_name]:
                intersection = np.count_nonzero(
                    np.logical_and(instance == gt_inst["instance_id"], pred_inst["mask"])
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)

        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_size]
        dist_threshes = [self.distance_thresh]
        dist_confs = [self.distance_conf]

        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )

        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {
                    p["uuid"]: False
                    for scene in scenes
                    for label in scene["pred"]
                    for p in scene["pred"][label]
                }

                for li, label_name in enumerate(self.valid_class_names):
                    y_true = []
                    y_score = []
                    hard_false_negatives = 0
                    has_gt = has_pred = False

                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]

                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]

                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)

                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"] + pred["vert_count"] - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        y_true.append(0)
                                        y_score.append(min_score)
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1

                        y_true.extend(cur_true[cur_match])
                        y_score.extend(cur_score[cur_match])

                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"] + pred["vert_count"] - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = float(num_ignore) / pred["vert_count"]
                                if proportion_ignore <= overlap_th:
                                    y_true.append(0)
                                    y_score.append(pred["confidence"])

                    if has_gt and has_pred:
                        y_true = np.array(y_true)
                        y_score = np.array(y_score)
                        sorted_indices = np.argsort(y_score)[::-1]
                        y_true = y_true[sorted_indices]
                        y_score = y_score[sorted_indices]

                        tp = np.cumsum(y_true)
                        fp = np.cumsum(1 - y_true)
                        fn = np.sum(y_true) - tp

                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn + hard_false_negatives)

                        ap = self.compute_ap(precision, recall)
                    elif has_gt:
                        ap = 0.0
                    else:
                        ap = float("nan")

                    ap_table[di, li, oi] = ap

        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))

        ap_scores = {
            "map": np.nanmean(ap_table[d_inf, :, oAllBut25]),
            "map50": np.nanmean(ap_table[d_inf, :, o50]),
            "map25": np.nanmean(ap_table[d_inf, :, o25]),
            "classes": {},
        }

        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {
                "ap": np.average(ap_table[d_inf, li, oAllBut25]),
                "ap50": np.average(ap_table[d_inf, li, o50]),
                "ap25": np.average(ap_table[d_inf, li, o25]),
            }

        if self.subset_mapper is not None:
            for subset_name in self.subset_mapper["subset_names"]:
                ap_scores[f"map_{subset_name}"] = []

            for class_name in self.valid_class_names:
                subset_name = self.subset_mapper[class_name]
                ap_scores[f"map_{subset_name}"].append(ap_scores["classes"][class_name]["ap"])

            for subset_name in self.subset_mapper["subset_names"]:
                ap_scores[f"map_{subset_name}"] = np.nanmean(ap_scores[f"map_{subset_name}"])

        return ap_scores

    @staticmethod
    def compute_ap(precision, recall):
        recall = np.concatenate([[0.0], recall, [1.0]])
        precision = np.concatenate([[0.0], precision, [0.0]])

        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        ap = 0.0
        for i in range(precision.size - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        return ap
