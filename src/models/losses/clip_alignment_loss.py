from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from warp.convnet.geometry.point_collection import PointCollection

from src.models.losses.loss_base import LossBase
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CLIPAlignmentLoss(LossBase):
    """Given the embedding, compute inner product with the target embedding and compute loss."""

    def __init__(
        self,
        normalize_input: bool,
        text_clip_path: str,
        loss_type: Literal["cross_entropy", "contrastive"],
        ignore_label: int = -100,
        learnable_logit: bool = False,
        eval_only: bool = False,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.eval_only = eval_only

        # load pre-computed text embeddings (e.g. CLIP text embedding with shape NxC)
        assert Path(text_clip_path).exists(), f"Text embedding file not found: {text_clip_path}"
        text_embeddings = torch.load(text_clip_path, map_location="cpu").detach()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        log.info(f"=> loaded text embeddings from {text_clip_path}")

        # create embedding
        self.emb_target = nn.Parameter(text_embeddings.float(), requires_grad=False)

        # learable logit
        self.logit_scale = 1.0
        if learnable_logit:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        # loss type
        self.loss_type = loss_type
        if self.loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_label)
        elif self.loss_type == "contrastive":
            self.loss_fn = nn.CosineEmbeddingLoss(margin=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x: Tensor | PointCollection) -> Tensor:
        if isinstance(x, PointCollection):
            x = x.feature_tensor
        if self.normalize_input:
            return F.normalize(x, p=2, dim=1)
        return x

    def loss(
        self,
        x: Tensor | PointCollection,
        target: Int[Tensor, ("N")],  # noqa: F821, F722
    ) -> Tensor:
        logit = self.predict(x, return_logit=True)
        if self.loss_type == "cross_entropy":
            # target is the index of the correct class
            loss = self.loss_fn(logit, target)
        elif self.loss_type == "contrastive":
            raise NotImplementedError("Contrastive loss not implemented yet")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss

    def predict(
        self,
        x: Float[Tensor, "N C"],  # noqa: F821, F722
        return_logit: bool = False,
    ) -> Int[Tensor, "N"]:  # noqa: F821, F722
        pred = self.forward(x)
        logit_scale = self.logit_scale
        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = logit_scale.exp()
        logit = torch.matmul(pred, self.emb_target.t()) * logit_scale

        if return_logit:
            return logit

        return logit.argmax(dim=1)
