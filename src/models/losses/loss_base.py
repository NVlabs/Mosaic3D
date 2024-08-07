from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig
from torch import Tensor


class LossBase(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def compute_loss(
        self,
        pred: Dict[str, Any],
        target: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Float[Tensor, ()]:
        raise NotImplementedError

    def forward(
        self,
        pred: Dict[str, Any],
        target: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Float[Tensor, ()]]:
        if not self.is_training:
            return {}

        return self.compute_loss(pred, target, *args, **kwargs)
