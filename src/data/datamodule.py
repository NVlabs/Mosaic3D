from typing import Any, Callable, List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.data.concat_dataset import ConcatDataset
from src.data.multi_dataloader import MultiDatasetDataloader
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_datasets,
        batch_size: int,
        num_workers: int,
        collate_fn: Callable,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[List[Dataset]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ["fit"] and not self.data_train:
            self.data_train = self.hparams.train_dataset()

        if self.data_val is None:
            self.data_val = [dataset() for dataset in self.hparams.val_datasets]

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise ValueError("No training dataset found. Please call `setup('fit')` first.")
        num_data = len(self.data_train)
        world_size = 1 if self.trainer is None else self.trainer.world_size
        num_batches = len(self.data_train) // (self.hparams.batch_size * world_size)
        log.info(f"num_data: {num_data}, num_batches: {num_batches}")
        if isinstance(self.data_train, Dataset):
            sampler = None
            if self.trainer.max_epochs < 0:
                sampler = RandomSampler(
                    self.data_train,
                    replacement=True,
                    num_samples=self.trainer.max_steps * self.hparams.batch_size,
                )
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=sampler is None,
                drop_last=True,
                collate_fn=self.hparams.collate_fn,
                sampler=sampler,
            )

    def val_dataloader(self) -> List[DataLoader[Any]]:
        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=self.hparams.collate_fn,
            )
            for dataset in self.data_val
        ]

    def test_dataloader(self) -> List[DataLoader[Any]]:
        return self.val_dataloader()


class MultiDataModule(DataModule):
    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise ValueError("No training dataset found. Please call `setup('fit')` first.")

        assert isinstance(self.data_train, ConcatDataset), "train_dataset must be a ConcatDataset"
        return MultiDatasetDataloader(
            self.data_train,
            self.hparams.batch_size,
            self.hparams.num_workers,
            self.hparams.collate_fn,
        )
