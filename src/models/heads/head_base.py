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
    def forward(self, x: Union[Tensor, Tuple[Tensor]]) -> Dict[str, Tensor]:
        """Placeholder of forward function.

        Args:
            x (Tensor): Feature maps.

        Returns:
            Dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'seg_preds'
                and 'feats'.
        """
        pass

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
