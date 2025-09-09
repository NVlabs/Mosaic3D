import random
from collections.abc import Mapping, Sequence
from typing import List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from src.models.utils.misc import offset2bincount


def convert_list_list_tensor_to_tensor(
    batched_list_of_point_indices: List[List[Int[Tensor, "N"]]],  # noqa: F722, F821
    batch_offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722, F821
    valid_mask: Optional[Bool[Tensor, "L"]] = None,  # noqa: F722, F821
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "M + 1"], Int[Tensor, "M"]]:  # noqa: F722, F821
    """Convert List[List[Tensor]] to concatenated indices, offsets, and counts."""
    # Get the counts of inner lists
    num_points_per_cap = [
        len(tensor) for sublist in batched_list_of_point_indices for tensor in sublist
    ]

    # Concatenate inner lists first and generate offsets for the inner lists
    batched_flat_point_indices = [
        torch.cat(sublist, dim=0) for sublist in batched_list_of_point_indices
    ]

    # Add batch offset if provided
    if batch_offsets is not None:
        if isinstance(batch_offsets, torch.Tensor):
            batch_offsets = batch_offsets.tolist()
        batched_flat_point_indices = [
            l + batch_offsets[i] for i, l in enumerate(batched_flat_point_indices)
        ]
    else:
        assert (
            len(batched_flat_point_indices) == 1
        ), "batch_offset must be provided if len(list_tensor) > 1"

    # Concatenate all lists and generate offsets for the outer lists
    point_indices = torch.cat(batched_flat_point_indices, 0)
    offsets = np.cumsum(num_points_per_cap)
    offsets = torch.tensor([0] + offsets.tolist())
    counts = torch.tensor(num_points_per_cap).to(point_indices.device)

    # Apply valid mask
    if valid_mask is not None:
        valid_point_indices = valid_mask[point_indices]
        point_indices = point_indices[valid_point_indices]

        # Use cumsum to efficiently compute valid counts
        cumulative_valid = torch.cumsum(valid_point_indices, dim=0)

        # Compute new offsets directly from cumulative_valid
        new_offsets = torch.zeros_like(offsets)
        new_offsets[1:] = cumulative_valid[offsets[1:] - 1]

        # Compute valid counts
        valid_counts = new_offsets[1:] - new_offsets[:-1]

        offsets = new_offsets
        counts = valid_counts

        # Convert point_indices to compacted indices
        compacted_indices = torch.cumsum(valid_mask, dim=0)
        point_indices = compacted_indices[point_indices] - 1

    return point_indices, offsets, counts


def collate_fn(batch):
    """Collate function for point cloud which support dict and list, 'coord' is necessary to
    determine 'offset'."""
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is an instance of Sequence, so need to check it first
        return list(batch)
    elif isinstance(batch[0], Sequence) and isinstance(batch[0][0], str):
        # Do not collate str list to distinguish from batch
        # batch = [item for sublist in batch for item in sublist]
        return batch
    elif isinstance(batch[0], Sequence) and isinstance(batch[0][0], torch.Tensor):
        return batch
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
                batch[key] = torch.cat((torch.zeros(1, dtype=torch.int32), batch[key]))
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, grid_size, mix_prob=0, drop_feat: bool = False):
    batch = [b for b in batch if b is not None]  # filter out None
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)

    if "view1_caption_data" in batch.keys():
        # caption loss on view1
        batch["caption_data"] = batch["view1_caption_data"]
        batch["offset"] = batch["view1_offset"]
        batch.pop("view1_caption_data")

    if "caption_data" in batch.keys():
        (
            point_indices,
            caption_offsets,
            num_points_per_caption,
        ) = convert_list_list_tensor_to_tensor(
            batched_list_of_point_indices=batch["caption_data"]["idx"],
            batch_offsets=batch["offset"],
        )
        batch["caption_data"]["point_indices"] = point_indices
        batch["caption_data"]["caption_offsets"] = caption_offsets
        batch["caption_data"]["num_points_per_caption"] = num_points_per_caption

    if "clip_point_indices" in batch.keys():
        batch_size = len(batch["clip_point_offset"]) - 1
        for idx_batch in range(1, batch_size):  # exclude idx_batch==0
            idx_start = batch["clip_point_offset"][idx_batch]
            idx_end = batch["clip_point_offset"][idx_batch + 1]
            batch["clip_point_indices"][idx_start:idx_end] += batch["offset"][idx_batch]

    if "clip_point_offset" in batch.keys():
        num_pts = batch["clip_point_offset"][-1]
        clip_indices_image_to_point = torch.zeros(
            (num_pts,),
            dtype=torch.int64,
        )
        for idx_batch, (idx_start, idx_end) in enumerate(
            zip(batch["clip_point_offset"][:-1], batch["clip_point_offset"][1:])
        ):
            clip_indices_image_to_point[idx_start:idx_end] += idx_batch
        batch["clip_indices_image_to_point"] = clip_indices_image_to_point

    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )

    if drop_feat:
        batch["feat"] = torch.ones_like(batch["feat"])

    batch["grid_size"] = grid_size
    return batch


def point_collate_fn_with_masks(batch, grid_size, mix_prob=0, drop_feat: bool = False):
    batch = [b for b in batch if b is not None]  # filter out None
    assert isinstance(batch[0], Mapping)

    batch_masks_binary = [
        sample.pop("masks_binary") for sample in batch if "masks_binary" in sample
    ]

    batch = point_collate_fn(batch, grid_size, mix_prob, drop_feat)

    if batch_masks_binary:
        batch["masks_binary"] = batch_masks_binary

    return batch


def point_collate_fn_with_captioned_masks(batch, grid_size, mix_prob=0, drop_feat: bool = False):
    batch = [b for b in batch if b is not None]
    assert isinstance(batch[0], Mapping)

    # Extract caption data before collating
    batch_captioned_masks = [x.pop("caption_data") for x in batch if "caption_data" in x]
    batch = point_collate_fn(batch, grid_size, mix_prob, drop_feat)

    if batch_captioned_masks:
        # Create binary masks for each caption's point indices
        collated_masks = []
        for num_points, caption_data in zip(
            offset2bincount(batch["offset"]), batch_captioned_masks
        ):
            has_embedding = "embedding" in caption_data
            key = "embedding" if has_embedding else "caption"
            num_captions = len(caption_data[key])

            # Create masks and set points to True
            masks = torch.zeros(
                [num_captions, num_points],
                dtype=torch.bool,
                device=batch["feat"].device,
            )
            for i, idx in enumerate(caption_data["idx"]):
                masks[i, idx] = True

            # Build collated mask with appropriate data
            collated_mask = {"mask": masks, key: caption_data[key], "num_captions": num_captions}
            collated_masks.append(collated_mask)

        batch["caption_data"] = collated_masks

    return batch
