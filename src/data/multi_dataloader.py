import os
import random
import weakref
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset

import src.utils.dist_utils as dist_utils
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ConcatDataset(Dataset):
    def __init__(self, datasets, repeat: int = 1, *args, **kwargs):
        super().__init__()

        self.datasets = [dataset() for dataset in datasets]
        self.data_list = self.get_data_list()
        self.repeat = repeat

        dataset_names = " | ".join([f"{d.dataset_name} ({len(d)})" for d in self.datasets])
        log.info(f"Loaded {self.__len__()} samples from {dataset_names}")

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i]), dtype=int) * i, np.arange(len(self.datasets[i]))
                )
            )
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]


class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        if dist_utils.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    """
    Multiple Datasets Dataloader, batch data from a same dataset and mix up ratio determined by repeat of each sub dataset.
    The overall length is determined by the main dataset (first) and repeat of concat dataset.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_gpu: int,
        num_worker_per_gpu: int,
        collate_fn: Callable,
        mix_prob=0,
        seed=None,
    ):
        self.datasets = concat_dataset.datasets
        self.global_repeat = concat_dataset.repeat
        self.ratios = [dataset.repeat for dataset in self.datasets]
        # reset data repeat, original repeat serve as ratios
        for dataset in self.datasets:
            dataset.repeat = 1
        # determine union training epoch by main dataset
        self.datasets[0].repeat = concat_dataset.repeat
        # build sub-dataloaders
        num_workers = num_worker_per_gpu // len(self.datasets)
        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            if dist_utils.get_world_size() > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=dataset_id,
                    num_workers=num_workers,
                    num_datasets=len(self.datasets),
                    rank=dist_utils.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=(sampler is None),
                    num_workers=num_worker_per_gpu,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    worker_init_fn=init_fn,
                    drop_last=True,
                    persistent_workers=True,
                )
            )
        self.sampler = MultiDatasetDummySampler()
        self.sampler.dataloader = weakref.proxy(self)

    def __iter__(self):
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])
                    except StopIteration:
                        if i == 0:
                            return
                        else:
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch

    def __len__(self):
        main_data_loader_length = len(self.dataloaders[0])
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        worker_seed = (
            num_workers * num_datasets * rank + num_workers * dataset_id + worker_id + seed
        )
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(worker_seed)
