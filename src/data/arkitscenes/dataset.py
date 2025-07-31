from typing import Dict, List, Optional

from src.data.dataset_base import AnnotatedDataset
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ARKitScenesDataset(AnnotatedDataset):
    CLASS_LABELS = []  # there is no GT semantic labels
    LOG_POSTFIX = "arkitscenes"

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


if __name__ == "__main__":
    dataset = ARKitScenesDataset(
        data_dir="/datasets/mosaic3d/data/arkitscenes",
        split="train",
        ignore_label=-100,
        repeat=1,
        transforms=None,
    )
