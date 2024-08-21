from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import torch
from torch import nn


class NetworkBase(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.device_indicator_param.device

    @abstractmethod
    def forward(self, input: Any) -> Any:
        pass

    # Loss and metrics are defined in the LightningModule

    @abstractmethod
    def data_dict_to_input(self, data_dict: Dict) -> Any:
        pass
