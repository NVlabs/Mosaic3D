from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor, nn


class NetworkBase(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, input: Any) -> Any:
        pass

    # Loss and metrics are defined in the LightningModule

    @abstractmethod
    def data_dict_to_input(self, data_dict: Dict) -> Any:
        pass


class NetworkBaseDict(NetworkBase):
    @abstractmethod
    def forward(self, data_dict: Dict) -> Dict:
        pass

    @abstractmethod
    def data_dict_to_input(self, data_dict: Dict) -> Dict:
        pass
