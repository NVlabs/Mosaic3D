import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import json

import numpy as np
import torch
from src.data.dataset_base import DatasetBase
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class ARKitScenesCaption3DDataset(DatasetBase):
    """
    ARKitScenes Dataset to load point clouds and 3D object captions generated from multi-view images.
    Inherits from DatasetBase and implements the required abstract methods.
    Captions are expected in 'multi_view_captions.json' within each scene directory under caption_dir/caption_subset.
    Segments (point indices) are expected in 'point_indices.npz' within each scene directory under segment_dir/segment_subset.
    """

    # ARKitScenes does not have predefined semantic labels or class IDs
    CLASS_LABELS = []
    CLASS_IDS = []

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        caption_dir: str,  # Expecting a single path to the root containing caption_subset folders
        caption_subset: str,  # Expecting the specific subset folder name (e.g., 'caption-mc-mv.gemma3.arkit')
        segment_dir: str,  # Expecting a single path to the root containing segment_subset folders
        segment_subset: str,  # Expecting the specific subset folder name (e.g., 'mask_clustering.cropformer.arkit')
        object_num_max: Optional[int] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
        # Removed ScanNet specific params: base_class_idx, novel_class_idx, ignore_class_idx
        # Removed unused general params: object_sample_ratio, load_embeddings
    ):
        # Ensure single string paths are provided for dirs and subsets
        assert isinstance(caption_dir, str), "caption_dir must be a string path"
        assert isinstance(caption_subset, str), "caption_subset must be a string name"
        assert isinstance(segment_dir, str), "segment_dir must be a string path"
        assert isinstance(segment_subset, str), "segment_subset must be a string name"

        super().__init__(
            dataset_name="arkitscenes",  # Use arkitscenes splits/metadata handling
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,  # Pass single dir path
            caption_subset=caption_subset,  # Pass single subset name
            segment_dir=segment_dir,  # Pass single dir path
            segment_subset=segment_subset,  # Pass single subset name
            object_num_max=object_num_max,
            object_sample_ratio=None,  # Not used for this dataset type
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
            mask_dir=None,  # Not used
            load_embeddings=False,  # Explicitly false
        )
        assert self.caption_dir[
            0
        ].exists(), f"Caption directory {self.caption_dir[0]} does not exist."
        assert self.segment_dir[
            0
        ].exists(), f"Segment directory {self.segment_dir[0]} does not exist."

        log.info(
            f"Initialized ARKitScenesCaption3DDataset for split '{split}'. "
            f"Loading captions from '{self.caption_dir[0]}/<scene_name>/multi_view_captions.json'. "
            f"Loading segments from '{self.segment_dir[0]}/<scene_name>/point_indices.npz'."
        )
        # Scan the caption directory and remove scenes that do not have a caption file
        print(f"Dataset has {len(self.scene_names)} scenes.")
        invalid_scene_names = []
        for scene_name in self.scene_names:
            caption_file = self.caption_dir[0] / scene_name / "multi_view_captions.json"
            if not (self.caption_dir[0] / scene_name).exists() or not caption_file.is_file():
                invalid_scene_names.append(scene_name)
        # remove invalid scene names
        self.scene_names = list(set(self.scene_names) - set(invalid_scene_names))
        print(
            f"Dataset has {len(self.scene_names)} scenes after removing scenes without captions."
        )

    def load_point_cloud(self, scene_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads point cloud coordinates and colors for a given ARKit scene."""
        split_folder = "Training" if self.split == "train" else "Validation"
        scene_dir = Path(self.data_dir) / split_folder / scene_name  # Use Path object
        assert scene_dir.exists(), f"Scene directory {scene_dir} not found."

        coord_path = scene_dir / "coord.npy"
        color_path = scene_dir / "color.npy"

        # Check file existence before loading
        assert coord_path.exists(), f"File not found: {coord_path}"
        assert color_path.exists(), f"File not found: {color_path}"

        coord = np.load(coord_path).astype(np.float32)
        color = np.load(color_path).astype(np.float32)  # Ensure color is float32
        # ARKit doesn't have semantic segment or instance annotations
        return coord, color

    def _load_3d_captions(self, scene_name: str) -> Dict[int, str]:
        """Load captions generated per 3D mask from multiple viewpoints.

        Args:
            scene_name: The name of the scene to load captions for.

        Returns:
            A dictionary mapping object instance indices (int, starting from 0) to string captions.
            Returns {} if the caption file is not found, empty, or invalid.
        """
        caption_file = self.caption_dir[0] / scene_name / "multi_view_captions.json"

        assert caption_file.is_file(), f"Caption file not found: {caption_file}"

        try:
            with open(caption_file) as f:
                captions = json.load(f)
        except json.JSONDecodeError:
            log.error(f"Error decoding JSON from {caption_file}. Returning empty captions.")
            return {}
        except Exception as e:
            log.error(
                f"Error loading captions from {caption_file}: {e}. Returning empty captions."
            )
            return {}

        # Ensure keys are integers (representing 0-based index) and values are strings
        valid_captions = {}
        if isinstance(captions, dict):  # Check if it's a dict first
            for k, v in captions.items():
                if k is None or v is None:
                    continue
                try:
                    # Keys in the json should correspond to the 0-based index
                    # of the object mask/point_indices array
                    obj_idx = int(k)
                    caption_text = str(v)
                    valid_captions[obj_idx] = caption_text
                except (ValueError, TypeError):
                    log.warning(f"Skipping invalid entry in {caption_file}: key={k}, value={v}")
                    continue
        else:
            log.error(
                f"Caption file {caption_file} does not contain a dictionary. Returning empty captions."
            )
            return {}

        return valid_captions

    def load_caption(self, scene_name: str) -> Tuple[List[torch.Tensor], List[str]]:
        """Loads captions and corresponding point indices for object instances.

        Args:
            scene_name: The name of the scene.

        Returns:
            A tuple containing:
            - A list of PyTorch tensors, where each tensor holds the point indices for an object instance.
            - A list of strings, where each string is the caption for the corresponding object instance.
            Returns ([], []) if captions or point indices cannot be loaded or are mismatched.
        """
        captions_dict: Dict[int, str] = self._load_3d_captions(scene_name)

        indices_path = self.segment_dir[0] / scene_name / "point_indices.npz"

        if not indices_path.is_file():
            log.warning(
                f"Point indices file not found: {indices_path}. Cannot pair with captions."
            )
            return [], []

        try:
            point_indices_all = unpack_list_of_np_arrays(indices_path)
        except Exception as e:
            log.error(f"Error loading point indices from {indices_path}: {e}")
            return [], []

        if not captions_dict or not point_indices_all:
            log.debug(
                f"No captions or point indices found for scene {scene_name}. Returning empty."
            )
            return [], []

        if len(point_indices_all) != len(captions_dict):
            log.warning(
                f"Mismatch between number of point index arrays ({len(point_indices_all)}) "
                f"and captions ({len(captions_dict)}) for scene {scene_name}. "
                f"Indices path: {indices_path}. Caption path: {self.caption_dir[0] / scene_name / 'multi_view_captions.json'} "
                f"Attempting to match based on keys..."
            )
            # Attempt to salvage by matching keys/indices if possible
            matched_indices = []
            matched_captions = []
            for i, indices in enumerate(point_indices_all):
                if i in captions_dict:
                    matched_indices.append(torch.from_numpy(indices).int())
                    matched_captions.append(captions_dict[i])
                else:
                    log.warning(
                        f"  - No caption found for index {i} in scene {scene_name}. Skipping this object."
                    )
            if not matched_indices:
                log.error(
                    f"Could not match any captions to point indices for {scene_name}. Returning empty."
                )
                return [], []
            log.warning(f"  - Successfully matched {len(matched_indices)} objects.")
            return matched_indices, matched_captions

        # If lengths match, proceed as before
        all_point_indices = []
        all_captions = []
        # Assuming keys in captions_dict are 0, 1, ..., N-1 matching the list order
        for i, point_indices in enumerate(point_indices_all):
            if i in captions_dict:
                all_point_indices.append(torch.from_numpy(point_indices).int())
                all_captions.append(captions_dict[i])
            else:
                # This case should ideally not happen if lengths matched, but added for safety
                log.error(
                    f"Index {i} missing in captions_dict for scene {scene_name} despite matching lengths. This indicates an issue with caption keys."
                )
                return [], []  # Return empty if keys are inconsistent

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

        # Load ARKit point cloud data (coord, color only)
        # load_point_cloud now handles assertions for file existence
        try:
            coord, color = self.load_point_cloud(scene_name)
        except AssertionError as e:
            log.error(f"Failed to load point cloud for scene {scene_name}: {e}")
            # Handle error: return None or raise, or return a dummy dict
            # Returning a dummy dict might mask issues downstream. Consider raising.
            # For now, let's re-raise to make the error explicit.
            raise e
        except Exception as e:
            log.error(f"Unexpected error loading point cloud for scene {scene_name}: {e}")
            raise e

        # ARKitScenes doesn't have semantic/instance segmentations in the base dataset
        # Create placeholder segment/binary arrays as done in original ARKitScenesDataset
        segment = np.ones(coord.shape[0], dtype=np.int64) * self.ignore_label
        binary_label = np.ones(
            coord.shape[0], dtype=np.float32
        )  # Treat all points as '1' (no base/novel concept here)
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,  # Placeholder, all ignored
            binary=binary_label,  # Placeholder, all 1
            origin_idx=origin_idx,
            scene_name=scene_name,
        )
        # ARKitScenes doesn't have instance labels, so don't add 'instance' key

        # Load captions and apply sampling explicitly for training split
        point_indices, captions = self.load_caption(scene_name)

        # Apply sampling if object_num_max is set and we have more objects
        if (
            point_indices  # Ensure list is not empty
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
            # It's valid for a scene to have no captioned objects, provide empty lists
            log.debug(f"No captions loaded or survived sampling for training scene {scene_name}")
            data_dict["caption_data"] = {"idx": [], "caption": []}

        # Apply transforms
        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict


if __name__ == "__main__":
    from rich.progress import Progress

    dataset = ARKitScenesCaption3DDataset(
        data_dir="/datasets/arkitscenes/3dod",
        split="train",
        transforms=None,
        object_num_max=50,
        caption_dir="/datasets/mosaic3d++",
        caption_subset="caption-mc-mv.gemma3.arkitscenes",
        segment_dir="/datasets/mosaic3d++",
        segment_subset="mask_clustering.cropformer.arkitscenes+combined",
    )
    progress = Progress()
    task = progress.add_task("Processing scenes", total=len(dataset.scene_names))
    for i in range(len(dataset)):
        dataset[i]
        progress.update(task, advance=1)
