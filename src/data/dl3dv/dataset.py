from typing import List, Optional, Union

import numpy as np
import torch

from src.data.dataset_base import DatasetBase
from src.utils import RankedLogger
from src.utils.io import unpack_list_of_np_arrays

log = RankedLogger(__name__, rank_zero_only=False)


class DL3DVDataset(DatasetBase):
    CLASS_LABELS = []  # there is no GT semantic labels

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
        ignore_label: int = -100,
        repeat: int = 1,
        log_postfix: Optional[str] = None,
        load_embeddings: bool = False,
        embedding_filename: Optional[str] = None,
    ):
        super().__init__(
            dataset_name="dl3dv",
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
        scene_dir = self.data_dir / scene_name
        coord = np.load(scene_dir / "pcd" / "coord_normalized.npy")
        color = np.load(scene_dir / "pcd" / "color_filtered.npy")
        return coord, color

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
        for caption_dir, segment_dir in zip(self.caption_dir, self.segment_dir):
            indices_path = segment_dir / scene_name / "point_indices.npz"
            caption_path = caption_dir / scene_name / "captions-gathered.npz"

            if not indices_path.exists() or not caption_path.exists():
                return [], []

            point_indices_all = unpack_list_of_np_arrays(indices_path)
            captions_all = unpack_list_of_np_arrays(caption_path)

            # filter out empty captions
            num_captions_per_object = np.array([sum(c != "cycle break") for c in captions_all])
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

            all_point_indices.extend(point_indices)
            all_captions.extend(captions)

        return all_point_indices, all_captions

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
    dataset = DL3DVDataset(
        data_dir="/datasets/dl3dv-10k/mast3r_sfm.dl3dv",
        split="train",
        transforms=None,
        caption_dir="/datasets/mosaic3d++",
        caption_subset="caption-mc.osprey.dl3dv",
        segment_dir="/datasets/mosaic3d++",
        segment_subset="mask_clustering.cropformer.dl3dv+combined",
        object_num_max=300,
        ignore_label=-100,
        load_embeddings=True,
        embedding_filename="embeddings-gathered.recap.npz",
    )

    for i in range(dataset.__len__()):
        print(i)
        sample = dataset[i]
