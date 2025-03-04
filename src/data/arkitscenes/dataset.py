from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from src.data.dataset_base import DatasetBase
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class ARKitScenesDataset(DatasetBase):
    CLASS_LABELS = []  # there is no GT semantic labels

    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: None,
        segment_dir: Optional[str] = None,
        segment_subset: Optional[Union[str, List[str]]] = None,
        caption_dir: Optional[str] = None,
        caption_subset: Optional[Union[str, List[str]]] = None,
        object_num_max: Optional[int] = None,
        object_sample_ratio: Optional[float] = None,
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
        load_embeddings: bool = False,
        embedding_filename: Optional[str] = None,
    ):
        super().__init__(
            dataset_name="arkitscenes",
            data_dir=data_dir,
            split=split,
            transforms=transforms,
            caption_dir=caption_dir,
            caption_subset=caption_subset,
            segment_dir=segment_dir,
            segment_subset=segment_subset,
            object_num_max=object_num_max,
            object_sample_ratio=object_sample_ratio,
            ignore_label=ignore_label,
            repeat=repeat,
            log_postfix=log_postfix,
            load_embeddings=load_embeddings,
        )
        self.embedding_filename = embedding_filename

    def load_point_cloud(self, scene_name: str):
        split = "Training" if self.split == "train" else "Validation"
        scene_dir = self.data_dir / split / scene_name
        coord = np.load(scene_dir / "coord.npy")
        color = np.load(scene_dir / "color.npy")
        coord = coord.astype(np.float32)
        return coord, color

    def load_caption(self, scene_name):
        all_point_indices = []
        all_captions = []

        for caption_dir, segment_dir in zip(self.caption_dir, self.segment_dir):
            indices_path = segment_dir / scene_name / "point_indices.npz"
            caption_path = caption_dir / scene_name / "captions.npz"
            if not indices_path.exists() or not caption_path.exists():
                continue

            point_indices = unpack_list_of_np_arrays(indices_path)
            captions = unpack_list_of_np_arrays(caption_path)

            # flatten the list of list
            point_indices = [item for sublist in point_indices for item in sublist]
            point_indices = [torch.from_numpy(indices).int() for indices in point_indices]
            captions = [item for sublist in captions for item in sublist]

            all_point_indices.extend(point_indices)
            all_captions.extend(captions)

        return all_point_indices, all_captions

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

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.scene_names)
        scene_name = self.scene_names[idx]

        # load pcd data
        coord, color = self.load_point_cloud(scene_name)
        segment = np.ones_like(coord[:, 0]) * self.ignore_label  # all are ignored
        binary_label = np.ones_like(segment)
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        data_dict = dict(
            coord=coord,
            color=color,
            segment=segment,
            binary=binary_label,
            origin_idx=origin_idx,
        )

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


if __name__ == "__main__":
    import os

    from natsort import natsorted
    from rich.progress import Progress

    datadir = "/datasets/arkitscenes/3dod"
    segment_dir = "/datasets/mosaic3d++/mask_clustering.cropformer.arkitscenes+combined"
    caption_dir = "/datasets/mosaic3d++/caption-mc.osprey.arkitscenes"

    scenes = os.listdir(segment_dir)

    bad_scenes = []
    with Progress() as progress:
        task = progress.add_task("Processing scenes", total=len(scenes))
        for scene in scenes:
            try:
                point_indices = unpack_list_of_np_arrays(
                    os.path.join(segment_dir, scene, "point_indices.npz")
                )
                num_point_indices = len(point_indices)
            except Exception as e:
                num_point_indices = 0
                bad_scenes.append((scene, "no point indices"))
            try:
                captions = unpack_list_of_np_arrays(
                    os.path.join(caption_dir, scene, "captions-gathered.npz")
                )
                num_captions = len(captions)
            except Exception as e:
                num_captions = 0
                bad_scenes.append((scene, "no captions"))

            if num_point_indices == 0:
                progress.console.print(
                    f"[red]scene: {scene}, len(point_indices): {num_point_indices:4d}, len(captions): {num_captions:4d}, match? {num_point_indices == num_captions}"
                )
            else:
                progress.console.print(
                    f"scene: {scene}, len(point_indices): {num_point_indices:4d}, len(captions): {num_captions:4d}, match? {num_point_indices == num_captions}"
                )
            progress.update(task, advance=1)

    for scene, reason in bad_scenes:
        print(f"{scene}: {reason}")
