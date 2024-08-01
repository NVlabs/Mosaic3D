from functools import partial
from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.collate import collate_regionplc
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class RegionPLCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_dataset,
        val_dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ["fit"] and not self.data_train:
            self.data_train = self.hparams.train_dataset()

        if self.data_val is None:
            self.data_val = self.hparams.val_dataset()

    def train_dataloader(self) -> DataLoader[Any]:
        num_data = len(self.data_train)
        num_batches = len(self.data_train) // (self.hparams.batch_size * self.trainer.world_size)
        log.info(f"num_data: {num_data}, num_batches: {num_batches}")
        if isinstance(self.data_train, Dataset):
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                drop_last=True,
                collate_fn=partial(
                    collate_regionplc,
                    ignore_label=self.data_train.ignore_label,
                    min_spatial_shape=self.data_train.min_spatial_shape,
                ),
            )

    def val_dataloader(self) -> DataLoader[Any]:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(
                collate_regionplc,
                ignore_label=self.data_val.ignore_label,
                min_spatial_shape=self.data_val.min_spatial_shape,
            ),
        )
