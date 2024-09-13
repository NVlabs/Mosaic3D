import os
import fire
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from glob import glob

import numpy as np
from PIL import Image
from natsort import natsorted
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
from clip import clip

from src.data.metadata.scannet import (
    CLASS_LABELS_20,
    CLASS_LABELS_200,
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from src.data.transform import Compose
from src.data.utils.geometry import project_3d_point_to_image
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ScanNetDataset(Dataset):
    CLASS_LABELS = CLASS_LABELS_20
    CLASS_IDS = VALID_CLASS_IDS_20

    def __init__(
        self,
        data_dir: str,
        image_root_path: str,
        pcd_root_path: str,
        split: str,
        transforms: None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[str] = None,
        object_sample_ratio: Optional[float] = None,
        base_class_idx: Optional[List[int]] = None,
        novel_class_idx: Optional[List[int]] = None,
        ignore_class_idx: Optional[List[int]] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        eval_clip_text_alignment: bool = True,
        eval_clip_image_alignment: bool = False,
        train_clip_text_alignment: bool = False,
        train_clip_image_alignment: bool = False,
        clip_input_resolution: int = 224,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_root_path = image_root_path
        self.pcd_root_path = pcd_root_path
        self.split = split
        self.object_sample_ratio = object_sample_ratio
        self.class_names = self.CLASS_LABELS
        self.repeat = repeat
        self.eval_clip_text_alignment = eval_clip_text_alignment
        self.eval_clip_image_alignment = eval_clip_image_alignment
        self.train_clip_text_alignment = train_clip_text_alignment
        self.train_clip_image_alignment = train_clip_image_alignment

        # set caption dir for train dataset
        self.caption_dir = Path(caption_dir) / caption_subset
        assert self.caption_dir.exists(), f"{self.caption_dir} not exist."

        # read scene names for split
        with open(f"src/data/metadata/split_files/scannetv2_{self.split}.txt") as f:
            self.scene_names = natsorted([line.strip() for line in f.readlines()])

        # class label mappers
        self.base_class_idx = base_class_idx
        self.novel_class_idx = novel_class_idx
        self.ignore_class_idx = ignore_class_idx
        self.valid_class_idx = np.arange(len(self.CLASS_LABELS)).tolist()
        if base_class_idx is not None and len(base_class_idx):
            self.base_class_mapper = self.build_class_mapper(base_class_idx, ignore_label)
            self.binary_class_mapper = self.build_binary_class_mapper(
                base_class_idx, novel_class_idx, ignore_class_idx, ignore_label
            )
        if ignore_class_idx is not None:
            for c in ignore_class_idx:
                self.valid_class_idx.remove(c)
        self.valid_class_mapper = self.build_class_mapper(self.valid_class_idx, ignore_label)
        self.ignore_label = ignore_label

        # data transform
        self.transforms = Compose(OmegaConf.to_container(transforms))

        log.info(f"Loaded {self.__len__()} samples in {self.split} set.")

        # backward compatibility for regionplc
        self.caption_cfg = dict(GATHER_CAPTION=False)

        # load clip preprocessing/tokenizer
        self.tokenize_text = clip.tokenize
        self.preprocess_image = clip._transform(clip_input_resolution)

    @property
    def use_base_class_mapper(self):
        return self.split == "train" and hasattr(self, "base_class_mapper")

    @staticmethod
    def build_class_mapper(class_idx, ignore_label, squeeze_label=False):
        remapper = np.ones(256, dtype=np.int64) * ignore_label
        for i, x in enumerate(class_idx):
            if squeeze_label:
                remapper[x] = i
            else:
                remapper[x] = x
        return remapper

    @staticmethod
    def build_binary_class_mapper(base_class_idx, novel_class_idx, ignore_class_idx, ignore_label):
        remapper = np.ones(256, dtype=np.int64) * ignore_label  # base: 1, novel: 0
        for _, x in enumerate(base_class_idx):
            remapper[x] = 1
        for _, x in enumerate(novel_class_idx):
            remapper[x] = 0
        # ignored categories are mapped to novel
        for _, x in enumerate(ignore_class_idx):
            remapper[x] = 0
        return remapper

    def __len__(self):
        length = len(self.scene_names) * (self.repeat if self.split == "train" else 1)
        return length

    def load_caption(self, scene_name):
        filepath = os.path.join(self.caption_dir, f"{scene_name}.npz")
        data = np.load(filepath)

        object_ids = data["object_ids"]
        captions = data["captions"]
        point_indices_flatten = data["point_indices"]
        num_points = data["num_points"]

        cumsum = np.cumsum(num_points)[:-1]
        point_indices = np.split(point_indices_flatten, cumsum)

        if self.object_sample_ratio < 1.0:
            sel = np.random.choice(
                np.arange(len(object_ids)),
                max(1, int(len(object_ids) * self.object_sample_ratio)),
                replace=False,
            )
            object_ids = object_ids[sel]
            captions = captions[sel]
            num_points = num_points[sel]
            point_indices = [point_indices[i] for i in sel]

        point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
        captions = list(captions)
        return point_indices, captions

    def load_clip_point_indices(
        self,
        scene_name: str,
        image_path: str,
        coord: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Load point indices that visible to the image."""
        # if image_path is None:
        #     idx_image = 0
        #     image_path = sorted(glob(os.path.join(
        #         self.image_root_path, scene_name, f"color/*.jpg"
        #     )))[idx_image]
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

        # scale_hw = np.array(image_hw) / np.array(depth_hw)
        # scale_wh = scale_hw[::-1]
        # pixel_coords = (pixel_coords * scale_wh).astype("int64")
        # return torch.LongTensor(pixel_coords)
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
        scene_dir = self.data_dir / self.split / scene_name

        # load pcd data
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        segment = np.load(scene_dir / "segment20.npy")
        instance = np.load(scene_dir / "instance.npy")

        # class mapping
        # base / novel label
        if hasattr(self, "base_class_mapper"):
            binary_label = self.binary_class_mapper[segment.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(segment)

        if self.use_base_class_mapper:
            segment = self.base_class_mapper[segment.astype(np.int64)]
        elif not self.use_base_class_mapper and hasattr(self, "ignore_class_idx"):
            segment = self.valid_class_mapper[segment.astype(np.int64)]
        instance[segment == self.ignore_label] = self.ignore_label

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            binary=binary_label,
            origin_idx=np.arange(coord.shape[0]).astype(np.int64),
        )

        # load captions
        point_indices, captions = self.load_caption(scene_name)
        data_dict["caption_data"] = {"idx": point_indices, "caption": captions}
        if self.eval_clip_text_alignment or self.train_clip_text_alignment:
            # NOTE: huggingface tokenization issue
            # ** is too long for context length.
            captions = [caption[:50] for caption in captions]
            data_dict["clip_tokenized_text"] = self.tokenize_text(captions)

        # load CLIP processed single image and corresponding point indices
        data_dict["clip_processed_image"], image_path = self.load_clip_processed_image(scene_name)
        data_dict["clip_point_indices"] = self.load_clip_point_indices(
            scene_name, image_path, coord
        )

        data_dict = self.transforms(data_dict)

        return data_dict


class ScanNet200Dataset(ScanNetDataset):
    CLASS_LABELS = CLASS_LABELS_200
    CLASS_IDS = VALID_CLASS_IDS_200

    def __init__(
        self,
        data_dir: str,
        caption_dir: str,
        split: str,
        transforms: None,
        base_class_idx: Optional[List[int]],
        novel_class_idx: Optional[List[int]],
        ignore_class_idx: Optional[List[int]],
        ignore_label: int = -100,
    ):
        super().__init__(
            data_dir,
            caption_dir,
            split,
            transforms,
            base_class_idx,
            novel_class_idx,
            ignore_class_idx,
            ignore_label,
        )


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
    splits = ["train", "val", "test"]
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


if __name__ == "__main__":
    fire.Fire(check_scannet_data)
