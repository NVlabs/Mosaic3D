from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig
from torch import Tensor


class LossBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def loss(
        self,
        pred: Dict[str, Any],
        target: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    def predict(
        self,
        pred: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        return pred

    def forward(
        self,
        pred: Dict[str, Any],
        target: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        if not self.is_training:
            return {}

        return self.loss(pred, target, *args, **kwargs)
