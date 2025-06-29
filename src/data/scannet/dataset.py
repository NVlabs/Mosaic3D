import os
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(DatasetBase):
    CLASS_LABELS = CLASS_LABELS_20
    CLASS_IDS = VALID_CLASS_IDS_20

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[Union[str, List[str]]] = None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[Union[str, List[str]]] = None,
        object_num_max: Optional[int] = None,
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
        mask_dir: Optional[str] = None,
        load_embeddings: bool = False,
        embedding_filename: Optional[str] = None,
        use_location_augmentation: bool = False,
        # unused
        image_root_path: Optional[str] = None,
        clip_text_alignment: bool = False,
        clip_image_alignment: bool = False,
        clip_input_resolution: int = 224,
    ):
        super().__init__(
            dataset_name="scannetv2",
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_num_max=object_num_max,
            object_sample_ratio=object_sample_ratio,
            base_class_idx=base_class_idx,
            novel_class_idx=novel_class_idx,
            ignore_class_idx=ignore_class_idx,
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
            mask_dir=mask_dir,
            load_embeddings=load_embeddings,
        )
        self.embedding_filename = embedding_filename
        self.use_location_augmentation = use_location_augmentation

    def load_point_cloud(self, scene_name: str):
        scene_dir = self.data_dir / self.split / scene_name
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        segment = np.load(scene_dir / "segment20.npy")
        instance = np.load(scene_dir / "instance.npy")
        return coord, color, segment, instance

    def load_embedding(self, scene_name):
        all_point_indices = []
        all_embeddings = []

        for i, caption_dir in enumerate(self.caption_dir):
            segment_dir = self.segment_dir[i]
            indices_path = segment_dir / scene_name / "point_indices.npz"
            embeddings_path = caption_dir / scene_name / f"{self.embedding_filename}.npz"

            if not indices_path.exists() or not embeddings_path.exists():
                return [], []

            point_indices_all = unpack_list_of_np_arrays(indices_path)
            embeddings_all = unpack_list_of_np_arrays(embeddings_path)

            num_embeddings_per_object = np.array([len(e) for e in embeddings_all])
            # filter out empty embeddings
            point_indices_all = [
                indices
                for indices, num_embeddings in zip(point_indices_all, num_embeddings_per_object)
                if num_embeddings > 0
            ]
            embeddings_all = [
                embeddings
                for embeddings, num_embeddings in zip(embeddings_all, num_embeddings_per_object)
                if num_embeddings > 0
            ]
            num_embeddings_per_object = np.array([len(e) for e in embeddings_all])
            idx_select_embedding = np.cumsum(
                np.insert(num_embeddings_per_object, 0, 0)[0:-1]
            ) + np.random.randint(0, num_embeddings_per_object, len(num_embeddings_per_object))

            # flatten the list of list
            point_indices = [torch.from_numpy(indices).int() for indices in point_indices_all]
            embeddings = np.concatenate(embeddings_all, axis=0)
            embeddings = torch.from_numpy(embeddings[idx_select_embedding]).float()
            all_point_indices.extend(point_indices)
            all_embeddings.extend(embeddings)

        return all_point_indices, all_embeddings

    def load_caption(self, scene_name):
        all_point_indices = []
        all_captions = []

        # Check all caption directories
        for i, caption_dir in enumerate(self.caption_dir):
            # legacy version
            if (caption_dir / f"{scene_name}.npz").exists():
                point_indices, captions = self._load_caption_legacy(scene_name, caption_dir)
            elif (
                (caption_dir / scene_name).is_dir()
                and (
                    (caption_dir / scene_name / "captions.npz").exists()
                    or (caption_dir / scene_name / "captions-gathered.npz").exists()
                )
                and hasattr(self, "segment_dir")
                and (self.segment_dir[i] / scene_name / "point_indices.npz").exists()
            ):
                if "mosaic3d" in str(caption_dir):
                    point_indices, captions = self._load_caption_mosaic3dpp(
                        scene_name, caption_dir, self.segment_dir[i]
                    )

                elif "gathered" in str(caption_dir):
                    point_indices, captions = self._load_caption_gathered(
                        scene_name, caption_dir, self.segment_dir[i]
                    )
                else:
                    point_indices, captions = self._load_caption(
                        scene_name, caption_dir, self.segment_dir[i]
                    )
            else:
                raise FileNotFoundError(
                    f"No caption data found for scene {scene_name} in any of the provided directories."
                )

            all_point_indices.extend(point_indices)
            all_captions.extend(captions)

        return all_point_indices, all_captions

    def _load_caption_mosaic3dpp(self, scene_name, caption_dir, segment_dir):
        indices_path = segment_dir / scene_name / "point_indices.npz"
        caption_path = caption_dir / scene_name / "captions-gathered.npz"

        if not indices_path.exists() or not caption_path.exists():
            return [], []

        point_indices_all = unpack_list_of_np_arrays(indices_path)
        captions_all = unpack_list_of_np_arrays(caption_path)

        num_captions_per_object = np.array([len(c) for c in captions_all])
        # filter out empty captions
        point_indices_all = [
            indices
            for indices, num_captions in zip(point_indices_all, num_captions_per_object)
            if num_captions > 0
        ]
        captions_all = [
            captions
            for captions, num_captions in zip(captions_all, num_captions_per_object)
            if num_captions > 0
        ]
        num_captions_per_object = np.array([len(c) for c in captions_all])
        idx_select_caption = np.cumsum(
            np.insert(num_captions_per_object, 0, 0)[0:-1]
        ) + np.random.randint(0, num_captions_per_object, len(num_captions_per_object))

        # flatten the list of list
        point_indices = [torch.from_numpy(indices).int() for indices in point_indices_all]
        captions = [item for sublist in captions_all for item in sublist]
        captions = [captions[i] for i in idx_select_caption]

        return point_indices, captions

    def _load_caption_legacy(self, scene_name, caption_dir):
        filepath = os.path.join(caption_dir, f"{scene_name}.npz")
        data = np.load(filepath)

        object_ids = data["object_ids"]
        captions = data["captions"]
        point_indices_flatten = data["point_indices"]
        num_points = data["num_points"]
        num_captions = data.get("num_captions", None)

        if num_captions is not None:
            idx_select_caption = (
                np.cumsum(np.insert(num_captions, 0, 0)[0:-1])
                + np.random.randint(0, num_captions.max(), num_captions.size) % num_captions
            )
            captions = np.array(captions)[idx_select_caption]

        cumsum = np.cumsum(num_points)[:-1]
        point_indices = np.split(point_indices_flatten, cumsum)
        point_indices = [torch.from_numpy(indices).int() for indices in point_indices]

        captions = list(captions)
        return point_indices, captions

    def _load_caption_gathered(self, scene_name, caption_dir, segment_dir):
        indices_path = segment_dir / scene_name / "point_indices.npz"
        caption_path = caption_dir / scene_name / "captions.npz"
        if not indices_path.exists() or not caption_path.exists():
            return [], []

        point_indices = unpack_list_of_np_arrays(indices_path)
        captions = unpack_list_of_np_arrays(caption_path)

        num_captions_per_object = [len(c) for c in captions]
        # randomly select one caption per object
        idx_select_caption = np.cumsum(
            np.insert(num_captions_per_object, 0, 0)[0:-1]
        ) + np.random.randint(0, num_captions_per_object, len(num_captions_per_object))

        # flatten the list of list
        point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
        captions = [item for sublist in captions for item in sublist]
        captions = [captions[i] for i in idx_select_caption]

        return point_indices, captions

    def _load_caption(self, scene_name, caption_dir, segment_dir):
        indices_path = segment_dir / scene_name / "point_indices.npz"
        caption_path = caption_dir / scene_name / "captions.npz"
        if not indices_path.exists() or not caption_path.exists():
            return [], []

        point_indices = unpack_list_of_np_arrays(indices_path)
        captions = unpack_list_of_np_arrays(caption_path)

        # flatten the list of list
        point_indices = [item for sublist in point_indices for item in sublist]
        point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
        captions = [item for sublist in captions for item in sublist]

        return point_indices, captions

    def load_clip_point_indices(
        self,
        scene_name: str,
        image_path: str,
        coord: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Load point indices that visible to the image."""
        depth_path = image_path.replace("color", "depth").replace(".jpg", ".png")
        pose_path = image_path.replace("color", "pose").replace(".jpg", ".txt")

        # load pcd
        if coord is None:
            scene_dir = self.data_dir / self.split / scene_name
            coord = np.load(scene_dir / "coord.npy")

        # load depth
        depth = np.array(Image.open(depth_path), dtype="uint16") / 1000.0
        depth_intrinsic = np.loadtxt(
            os.path.join(self.image_root_path, scene_name, "intrinsics_depth.txt")
        )
        depth_hw = depth.shape[:2]  # (height, width)
        image_hw = Image.open(image_path).size[::-1]

        # load pose
        pose = np.loadtxt(pose_path)

        # project point to image
        point_indices, pixel_coords = project_3d_point_to_image(
            coord, pose, depth, depth_hw, depth_intrinsic
        )

        return torch.LongTensor(point_indices)

    def load_clip_processed_image(
        self,
        scene_name: str,
    ) -> torch.Tensor:
        image_paths = sorted(glob(os.path.join(self.image_root_path, scene_name, "color/*.jpg")))
        if self.split == "train":
            idx_image = np.random.randint(len(image_paths))
        else:
            idx_image = 0
        image_path = image_paths[idx_image]
        assert os.path.exists(image_path), f"no image at {image_path:s}"
        return self.preprocess_image(Image.open(image_path)).unsqueeze(0), image_path

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        # load pcd data
        coord, color, segment, instance = self.load_point_cloud(scene_name)

        # class mapping
        # base / novel label
        if hasattr(self, "base_class_mapper"):
            binary_label = self.binary_class_mapper[segment.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(segment)

        if self.use_base_class_mapper:
            segment = self.base_class_mapper[segment.astype(np.int64)]
        else:
            segment = self.valid_class_mapper[segment.astype(np.int64)]
        instance[segment == self.ignore_label] = self.ignore_label

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            binary=binary_label,
            origin_idx=np.arange(coord.shape[0]).astype(np.int64),
        )
        if self.split != "train":
            data_dict["instance"] = instance

        # load mask data
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"{scene_name}.npz")
            mask_data = np.load(mask_path)
            masks_binary = mask_data["masks_binary"]
            data_dict["masks_binary"] = masks_binary

        # load captions
        if self.split == "train":
            if self.load_embeddings:
                point_indices, embeddings = self.load_embedding_and_sample(scene_name)
                data_dict["caption_data"] = {"idx": point_indices, "embedding": embeddings}
            else:
                point_indices, captions = self.load_caption_and_sample(scene_name)
                data_dict["caption_data"] = {"idx": point_indices, "caption": captions}

        data_dict = self.transforms(data_dict)
        return data_dict


class ScanNet200Dataset(ScanNetDataset):
    CLASS_LABELS = CLASS_LABELS_200
    CLASS_IDS = VALID_CLASS_IDS_200

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[Union[str, List[str]]] = None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[Union[str, List[str]]] = None,
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        mask_dir: Optional[str] = None,
        log_postfix: Optional[str] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_sample_ratio=object_sample_ratio,
            base_class_idx=base_class_idx,
            novel_class_idx=novel_class_idx,
            ignore_class_idx=ignore_class_idx,
            ignore_label=ignore_label,
            repeat=repeat,
            mask_dir=mask_dir,
            log_postfix=log_postfix,
        )

    def build_subset_mapper(self):
        mapper = {}
        mapper["subset_names"] = ["head", "common", "tail"]
        for name in self.CLASS_LABELS:
            if name in HEAD_CLASSES_200:
                mapper[name] = "head"
            elif name in COMMON_CLASSES_200:
                mapper[name] = "common"
            elif name in TAIL_CLASSES_200:
                mapper[name] = "tail"
            else:
                raise ValueError(f"Unknown class name: {name}")

        return mapper

    def load_point_cloud(self, scene_name: str):
        scene_dir = self.data_dir / self.split / scene_name
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        segment = np.load(scene_dir / "segment200.npy")
        instance = np.load(scene_dir / "instance.npy")
        return coord, color, segment, instance


def check_scannet_data(
    data_dir: str = "/datasets/scannet_hf",
    image_root_path: str = "/datasets/scannet_frames/tasks/scannet_frames_25k",
    pcd_root_path: str = "/datasets/scannet_frames/tasks/scannet_frames_25k",
    caption_dir: str = "/datasets/scannet_captions",
    caption_subset: str = "ours_sam_video",
):
    data_dir = Path(data_dir)
    caption_dir = Path(caption_dir) / caption_subset

    erroneous_files = []

    # Check if directories exist
    if not data_dir.exists():
        erroneous_files.append(f"Data directory {data_dir} does not exist.")
    if not caption_dir.exists():
        erroneous_files.append(f"Caption directory {caption_dir} does not exist.")

    # Read scene names for all splits
    splits = ["train", "val"]
    scene_names = []
    for split in splits:
        split_file = f"src/data/metadata/split_files/scannetv2_{split}.txt"
        if not os.path.exists(split_file):
            erroneous_files.append(f"Split file {split_file} does not exist.")
            continue
        with open(split_file) as f:
            scene_names.extend([(split, line.strip()) for line in f.readlines()])

    print(f"Checking {len(scene_names)} scenes...")

    for split, scene_name in tqdm(scene_names):
        # Check point cloud data
        scene_dir = data_dir / split / scene_name
        if not scene_dir.exists():
            erroneous_files.append(f"Scene directory {scene_dir} does not exist.")
            continue

        files_to_check = ["coord.npy", "color.npy"]
        if split != "test":
            files_to_check.extend(["segment20.npy", "instance.npy"])

        for file in files_to_check:
            if not (scene_dir / file).exists():
                erroneous_files.append(f"{file} does not exist for scene {split}/{scene_name}.")

        # Check caption data (only for train and val sets)
        if split != "test":
            caption_file = caption_dir / f"{scene_name}.npz"
            if not caption_file.exists():
                erroneous_files.append(f"Caption file {caption_file} does not exist.")
            else:
                try:
                    caption_data = np.load(caption_file)
                    required_keys = ["object_ids", "captions", "point_indices", "num_points"]
                    for key in required_keys:
                        if key not in caption_data:
                            erroneous_files.append(
                                f"Key '{key}' missing in caption data for scene {split}/{scene_name}."
                            )

                    num_points = caption_data["num_points"]
                    if any(num_points == 0):
                        erroneous_files.append(
                            f"num_points has zero for scene {split}/{scene_name}."
                        )

                    if len(caption_data["point_indices"]) != num_points.sum():
                        erroneous_files.append(
                            f"point_indices has wrong length for scene {split}/{scene_name}."
                        )

                    # Check the point_indices: list[array] is not empty
                    if len(caption_data["point_indices"]) == 0:
                        erroneous_files.append(
                            f"point_indices is empty for scene {split}/{scene_name}."
                        )

                except Exception as e:
                    erroneous_files.append(
                        f"Error loading caption data for scene {split}/{scene_name}: {str(e)}"
                    )

        # Check image data
        image_scene_dir = os.path.join(image_root_path, scene_name)
        if not os.path.exists(image_scene_dir):
            erroneous_files.append(f"Image directory {image_scene_dir} does not exist.")
            continue

        color_images = glob(os.path.join(image_scene_dir, "color/*.jpg"))
        depth_images = glob(os.path.join(image_scene_dir, "depth/*.png"))
        pose_files = glob(os.path.join(image_scene_dir, "pose/*.txt"))

        if len(color_images) == 0:
            erroneous_files.append(f"No color images found for scene {split}/{scene_name}.")
        if len(depth_images) == 0:
            erroneous_files.append(f"No depth images found for scene {split}/{scene_name}.")
        if len(pose_files) == 0:
            erroneous_files.append(f"No pose files found for scene {split}/{scene_name}.")
        if len(color_images) != len(depth_images) or len(color_images) != len(pose_files):
            erroneous_files.append(
                f"Mismatch in number of color, depth, and pose files for scene {split}/{scene_name}."
            )

        # Check intrinsics file
        intrinsics_file = os.path.join(image_scene_dir, "intrinsics_depth.txt")
        if not os.path.exists(intrinsics_file):
            erroneous_files.append(f"Intrinsics file missing for scene {split}/{scene_name}.")

    if erroneous_files:
        print("The following errors were found:")
        for error in erroneous_files:
            print(error)
        print(f"Total number of errors: {len(erroneous_files)}")
    else:
        print("All data checks passed successfully!")

    return erroneous_files


def convert_scannet200_gt_format(
    data_dir: str = "/datasets/scannet_hf",
    output_dir: str = "/datasets/openvocab-3d-captions",
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    segment_dir = output_dir / "segment.gt.scannet200"
    caption_dir = output_dir / "caption.gt.scannet200"  # w/o prompt engineering

    # Read scene names for all splits
    splits = ["train", "val"]
    scene_names = []
    for split in splits:
        split_file = f"src/data/metadata/split_files/scannetv2_{split}.txt"
        with open(split_file) as f:
            scene_names.extend([(split, line.strip()) for line in f.readlines()])

    for split, scene_name in tqdm(scene_names):
        scene_dir = data_dir / split / scene_name
        instance = np.load(scene_dir / "instance.npy")
        segment = np.load(scene_dir / "segment200.npy")

        # Filter out invalid instance ids
        valid_instance_mask = instance != -1
        valid_segment_mask = segment != -1
        valid_mask = valid_instance_mask & valid_segment_mask
        valid_instance = instance[valid_mask]
        valid_segment = segment[valid_mask]

        # Get unique instances
        unique_instances, unique_idx = np.unique(valid_instance, return_index=True)
        unique_segment = valid_segment[unique_idx]

        # Convert instance to point_indices (flattened)
        point_indices_all = []
        captions_all = []
        inner_lengths_all = []
        for unique_instance, unique_segment in zip(unique_instances, unique_segment):
            point_indices = np.where(instance == unique_instance)[0].tolist()
            point_indices_all.extend(point_indices)
            inner_lengths_all.append(len(point_indices))
            captions_all.append(CLASS_LABELS_200[unique_segment])

        point_indices_all = np.array(point_indices_all).astype(np.int32)
        inner_lengths_all = np.array(inner_lengths_all).astype(np.int64)
        outer_lengths_all = np.array([len(inner_lengths_all)]).astype(
            np.int64
        )  # for compatibility with Mosaic3D
        captions_all = np.array(captions_all).astype(np.str_)

        # Save segment
        segment_output_dir = segment_dir / f"{scene_name}"
        segment_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            segment_output_dir / "point_indices.npz",
            packed=point_indices_all,
            outer_lengths=outer_lengths_all,
            inner_lengths=inner_lengths_all,
        )

        # Save caption
        caption_output_dir = caption_dir / f"{scene_name}"
        caption_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            caption_output_dir / "captions.npz",
            packed=captions_all,
            lengths=outer_lengths_all,
        )


if __name__ == "__main__":
    # fire.Fire(check_scannet_data)
    # fire.Fire(convert_scannet200_gt_format)
    dataset = ScanNetDataset(
        data_dir="/datasets/scannet_hf",
        split="train",
        transforms=None,
        caption_dir="/datasets/mosaic3d++",
        caption_subset="caption-mc.osprey.scannet-125k.cyclic-check",
        segment_dir="/datasets/mosaic3d++",
        segment_subset="mask_clustering.cropformer.scannet-125k",
        object_num_max=300,
        ignore_label=-100,
        load_embeddings=False,
    )

    for i in range(dataset.__len__()):
        print(i)
        sample = dataset[i]
