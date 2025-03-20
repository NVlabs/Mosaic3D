# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


class BinarySegmentationLoss(nn.Module):
    """This class computes the loss for OpenSegment3D.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth masks and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher: nn.Module,
        losses: List[str],
        num_points: int = -1,
        oversample_ratio: Optional[float] = None,
        importance_sample_ratio: Optional[float] = None,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices):
        """Binary classification loss"""
        assert "logit" in outputs
        src_logits = outputs["logit"]  # [B, Q, 2]

        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            1,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = 0  # if the query is matched to a target, set the target class to 0

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "mask" containing a tensor of dim [nb_target_boxes, num_points]
        """
        assert "mask" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["mask"][batch_id][:, map_id].T
            target_mask = targets[batch_id]["mask"][target_id]

            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1], device=target_mask.device)[
                    : int(self.num_points * target_mask.shape[1])
                ]
            else:
                # sample all points
                point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))

        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices)

    def forward(self, outputs, targets, return_indices: bool = False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        return (losses, indices) if return_indices else losses

    def __repr__(self):
        head = "Binary Segmentation Loss " + self.__class__.__name__
        body = [
            f"matcher: {self.matcher.__repr__(_repr_indent=8)}",
            f"losses: {self.losses}",
            f"num_points: {self.num_points}",
            f"oversample_ratio: {self.oversample_ratio}",
            f"importance_sample_ratio: {self.importance_sample_ratio}",
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
