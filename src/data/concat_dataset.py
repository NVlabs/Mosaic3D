import numpy as np
from torch.utils.data import Dataset

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
