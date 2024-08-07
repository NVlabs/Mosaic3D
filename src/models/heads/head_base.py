from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from omegaconf import DictConfig
from torch import Tensor


class BaseHead(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def loss(self, x: Union[Tensor, Tuple[Tensor]], data: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    def predict(
        self,
        x: Union[Tensor, Tuple[Tensor]],
    ) -> List[Tensor]:
        """Inference.

        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_img_metas (List[dict]): List of image information.

        Returns:
            list[Tensor]: prediction
        """
        return self.forward(x)["seg_preds"]


class BaseSemanticHead(BaseHead, metaclass=ABCMeta):
    pass
