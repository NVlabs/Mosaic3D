from typing import Dict, List, Optional

from src.data.dataset_base import AnnotatedDataset
from src.data.metadata.matterport3d import (
    CLASS_LABELS_21,
    CLASS_LABELS_40,
    CLASS_LABELS_80,
    CLASS_LABELS_160,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class Matterport3DDataset(AnnotatedDataset):
    CLASS_LABELS = CLASS_LABELS_21
    SEGMENT_FILE = "segment.npy"
    LOG_POSTFIX = "matterport3d"

    def __init__(
        self,
        data_dir: str,
        split: str,
        ignore_label: int = -100,
        repeat: int = 1,
        transforms: Optional[List[Dict]] = None,
        num_masks: Optional[int] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            repeat=repeat,
            ignore_label=ignore_label,
            transforms=transforms,
            num_masks=num_masks,
        )


class Matterport3D40Dataset(Matterport3DDataset):
    CLASS_LABELS = CLASS_LABELS_40
    LOG_POSTFIX = "matterport3d40"


class Matterport3D80Dataset(Matterport3DDataset):
    CLASS_LABELS = CLASS_LABELS_80
    LOG_POSTFIX = "matterport3d80"


class Matterport3D160Dataset(Matterport3DDataset):
    CLASS_LABELS = CLASS_LABELS_160
    LOG_POSTFIX = "matterport3d160"


if __name__ == "__main__":
    dataset = Matterport3D160Dataset(
        data_dir="/datasets/mosaic3d/data/matterport3d",
        split="train",
        ignore_label=-100,
        repeat=1,
        transforms=None,
    )
