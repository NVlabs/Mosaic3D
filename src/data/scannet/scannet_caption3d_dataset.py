import os
from glob import glob
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Tuple

import fire
import numpy as np
import torch
from clip import clip
from PIL import Image
from tqdm import tqdm
import json

from src.data.dataset_base import DatasetBase
from src.data.metadata.scannet import (
    CLASS_LABELS_20,
    CLASS_LABELS_200,
    COMMON_CLASSES_200,
    HEAD_CLASSES_200,
    TAIL_CLASSES_200,
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from src.data.utils.geometry import project_3d_point_to_image
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays
from src.data.scannet.dataset import ScanNetDataset

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetCaption3DDataset(DatasetBase):
    """
    ScanNet Dataset to load point clouds and 3D object captions generated from multi-view images.
    Inherits from DatasetBase and implements the required abstract methods.
    Captions are expected in 'multi_view_captions.json' within each scene directory under caption_dir.
    """

    # Define class labels and IDs required by DatasetBase logic if used (e.g., mappers)
    CLASS_LABELS = CLASS_LABELS_20
    CLASS_IDS = VALID_CLASS_IDS_20

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        caption_dir: str,  # Expecting a single path to the root containing scene folders with json files
        caption_subset: Optional[str] = None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[str] = None,
        object_num_max: Optional[int] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
        # Removed: caption_subset, segment_dir, segment_subset, mask_dir, load_embeddings, embedding_filename, etc.
    ):
        assert caption_dir is not None, "caption_dir must be provided"
        assert segment_dir is not None, "segment_dir must be provided"
        if isinstance(caption_subset, list):
            assert len(caption_subset) == 1, "Only one caption_subset is supported"
            caption_subset = caption_subset[0]
        if isinstance(segment_subset, list):
            assert len(segment_subset) == 1, "Only one segment_subset is supported"
            segment_subset = segment_subset[0]
        super().__init__(
            dataset_name="scannetv2",  # Assuming ScanNet v2 split files are used
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            # Pass dummy values or None for base class parameters we don't directly use here
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_num_max=object_num_max,
            object_sample_ratio=None,
            base_class_idx=base_class_idx,
            novel_class_idx=novel_class_idx,
            ignore_class_idx=ignore_class_idx,
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
            mask_dir=None,  # Not used
            load_embeddings=False,  # Explicitly false
        )
        self.caption_root_path = Path(caption_dir) / caption_subset
        assert (
            self.caption_root_path.exists()
        ), f"Caption directory {self.caption_root_path} does not exist."
        # Store caption dir as a single Path for internal use
        # self.caption_dir from base class is now potentially a list, ignore it internally.
        log.info(
            f"Initialized ScanNetCaption3DDataset for split '{split}'. Loading captions from '{self.caption_root_path}/<scene_name>/multi_view_captions.json'."
        )

    def load_point_cloud(self, scene_name: str):
        """Loads point cloud data (coords, colors, segments, instances) for a given scene."""
        scene_dir = self.data_dir / self.split / scene_name
        assert scene_dir.exists(), f"Scene directory {scene_dir} not found."

        coord_path = scene_dir / "coord.npy"
        color_path = scene_dir / "color.npy"
        segment_path = scene_dir / "segment20.npy"  # Default to segment20
        instance_path = scene_dir / "instance.npy"

        assert coord_path.exists(), f"File not found: {coord_path}"
        assert color_path.exists(), f"File not found: {color_path}"
        assert segment_path.exists(), f"File not found: {segment_path}"
        assert instance_path.exists(), f"File not found: {instance_path}"

        coord = np.load(coord_path)
        color = np.load(color_path)
        segment = np.load(segment_path)
        instance = np.load(instance_path)

        return coord, color, segment, instance

    def _load_3d_captions(self, scene_name: str) -> Optional[Dict[int, str]]:
        """Load captions generated per 3D mask from multiple viewpoints.

        Args:
            scene_name: The name of the scene to load captions for.

        Returns:
            A dictionary mapping object instance IDs (int) to string captions,
            or None if the caption file is not found. Returns {} if file is empty or invalid.
        """
        caption_path = self.caption_dir[0] / scene_name / "multi_view_captions.json"

        # Assert file exists before trying to open
        assert caption_path.is_file(), f"Caption path is not a file: {caption_path}"

        with open(caption_path) as f:
            try:
                captions = json.load(f)
            except json.JSONDecodeError:
                # If JSON is invalid, treat as empty/unusable captions for this scene
                log.error(f"Error decoding JSON from {caption_path}. Returning empty captions.")
                return {}  # Return empty dict if file is invalid JSON

        # Ensure keys are integers and values are strings, filter out invalid entries
        # Using dict comprehension with checks instead of try-except for ValueError
        valid_captions = {}
        for k, v in captions.items():
            if k is None or v is None:
                continue
            try:
                obj_id = int(k)
                caption_text = str(v)
                valid_captions[obj_id] = caption_text
            except (ValueError, TypeError):
                log.warning(f"Skipping invalid entry in {caption_path}: key={k}, value={v}")
                continue  # Skip this entry if conversion fails

        return valid_captions

    def load_caption(self, scene_name: str) -> Tuple[List[torch.Tensor], List[str]]:
        """Implementation of abstract method. Loads captions from multi_view_captions.json.

        Retrieves object instance IDs and their associated captions from the JSON file,
        then finds the corresponding point indices for each object instance in the point cloud.

        Args:
            scene_name: The name of the scene.

        Returns:
            A tuple containing:
            - A list of PyTorch tensors, where each tensor holds the point indices for an object instance.
            - A list of strings, where each string is the caption for the corresponding object instance.
            Returns ([], []) if captions cannot be loaded or no matches are found.
        """
        captions_dict: Dict[int, str] = self._load_3d_captions(scene_name)

        assert captions_dict, f"No captions found for scene {scene_name}"
        assert len(captions_dict) > 0, f"No captions found for scene {scene_name}"

        # Load point_indices
        indices_path = self.segment_dir[0] / scene_name / "point_indices.npz"
        point_indices_all = unpack_list_of_np_arrays(indices_path)

        assert len(point_indices_all) == len(
            captions_dict
        ), f"Number of point indices and captions must be the same. len(point_indices_all): {len(point_indices_all)}, len(captions_dict): {len(captions_dict)}"

        all_point_indices = []
        all_captions = []
        for i, point_indices in enumerate(point_indices_all):
            all_point_indices.append(torch.from_numpy(point_indices).int())
            all_captions.append(captions_dict[i])

        return all_point_indices, all_captions

    # Implement load_embedding as required by DatasetBase, even if unused
    def load_embedding(self, scene_name: str):
        log.warning(
            f"load_embedding called on {self.__class__.__name__}, but it is not implemented/supported. Returning empty lists."
        )
        return [], []

    def __getitem__(self, idx_original):
        """Implementation of abstract method. Gets a data sample."""
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        # Load pcd data - Assertion happens within load_point_cloud
        coord, color, segment, instance = self.load_point_cloud(scene_name)

        # --- Class Mapping (Leveraging DatasetBase logic) ---
        # Binary label for base/novel distinction if configured
        if hasattr(self, "base_class_mapper") and self.base_class_idx and self.novel_class_idx:
            binary_label = self.binary_class_mapper[segment.astype(np.int64)].astype(np.float32)
        else:
            # Default to all points being 'base' (1) or handle as needed
            binary_label = np.ones_like(segment, dtype=np.float32)

        # Map segment labels based on configuration (valid classes or base classes)
        if self.use_base_class_mapper:  # Property from DatasetBase
            segment = self.base_class_mapper[segment.astype(np.int64)]
        else:
            segment = self.valid_class_mapper[segment.astype(np.int64)]

        # Apply ignore label to instances corresponding to ignored segments
        instance[segment == self.ignore_label] = self.ignore_label
        # --- End Class Mapping ---

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,  # Mapped segment labels
            binary=binary_label,  # Binary base/novel labels
            origin_idx=np.arange(coord.shape[0]).astype(np.int64),
            scene_name=scene_name,
        )
        # Include instance map only if not training (common practice)
        if self.split != "train":
            data_dict["instance"] = instance

        # Load captions and apply sampling explicitly for training split
        if self.split == "train":
            point_indices, captions = self.load_caption(scene_name)  # Call directly

            # Apply sampling if needed
            if (
                point_indices
                and self.object_num_max is not None
                and self.object_num_max < len(point_indices)
            ):
                sel = np.random.choice(len(point_indices), self.object_num_max, replace=False)
                point_indices = [point_indices[i] for i in sel]
                captions = [captions[i] for i in sel]

            # Add caption data if any captions were loaded and survived sampling
            if point_indices:
                data_dict["caption_data"] = {"idx": point_indices, "caption": captions}
            else:
                log.debug(
                    f"No captions loaded or survived sampling for training scene {scene_name}"
                )
                # Optionally add empty caption data: data_dict["caption_data"] = {"idx": [], "caption": []}

        # Apply transforms
        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict


if __name__ == "__main__":
    dataset = ScanNetCaption3DDataset(
        data_dir="/datasets/scannet_hf",
        split="train",
        transforms=None,
        object_num_max=100,
        caption_dir="/datasets/mosaic3d++",
        caption_subset="caption-mc-mv.gemma3.scannet-250k",
        segment_dir="/datasets/mosaic3d++",
        segment_subset="mask_clustering.cropformer.scannet-250k+combined",
    )
    print(len(dataset))
    # Load the 3D caption and point indices
    scene_name = "scene0000_00"
    point_indices, captions = dataset.load_caption(scene_name)
    # Check length of point_indices and captions
    assert len(point_indices) == len(
        captions
    ), "Length of point_indices and captions must be the same"
    # Randomply sample 10 point indices and captions
    print(f"Number of objects in scene {scene_name}: {len(point_indices)}")
    sample_idx = np.random.choice(len(point_indices), 10, replace=False)
    for i in sample_idx:
        print(f"Point indices length: {len(point_indices[i])}")
        print(f"Caption: {captions[i]}")

    # Test the dataset indexing
    data_dict = dataset[0]
    print(data_dict.keys())

    # Check the caption_data key
    print(data_dict["caption_data"].keys())
    assert "idx" in data_dict["caption_data"]
    assert "caption" in data_dict["caption_data"]
    # Assert the length of idx and caption are the same
    assert len(data_dict["caption_data"]["idx"]) == len(
        data_dict["caption_data"]["caption"]
    ), "Length of idx and caption must be the same"
    print(
        f"Number of objects in scene {data_dict['scene_name']}: {len(data_dict['caption_data']['idx'])}"
    )
