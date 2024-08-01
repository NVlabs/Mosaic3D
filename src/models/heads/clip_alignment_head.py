from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from src.models.heads.head_base import BaseHead
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FixedCLIPAlignmentHead(BaseHead):
    def __init__(
        self,
        normalize_input: bool,
        text_clip_path: str,
        loss_type: Literal["cross_entropy", "contrastive"],
        ignore_label: int = -100,
    ):
        super().__init__()
        self.normalize_input = normalize_input

        # load pre-computed text embeddings (e.g. CLIP text embedding with shape NxC)
        assert Path(text_clip_path).exists(), f"Text embedding file not found: {text_clip_path}"
        text_embeddings = torch.load(text_clip_path, map_location="cpu").detach()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        log.info(f"=> loaded text embeddings from {text_clip_path}")

        # create embedding
        self.emb_target = nn.Parameter(text_embeddings.float(), requires_grad=False)

        # loss type
        self.loss_type = loss_type
        if self.loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_label)
        elif self.loss_type == "contrastive":
            self.loss_fn = nn.CosineEmbeddingLoss(margin=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x: Tensor) -> Tensor:
        if self.normalize_input:
            return F.normalize(x, p=2, dim=1)
        return x

    def loss(
        self, x: Union[Tensor, Tuple[Tensor]], data: Int[Tensor, ("N C")]  # noqa: F821, F722
    ) -> Dict[str, Tensor]:
        pred = self.forward(x)
        logit = torch.matmul(pred, self.emb_target.t())
        if self.loss_type == "cross_entropy":
            # target is the index of the correct class
            loss = self.loss_fn(logit, data)
        elif self.loss_type == "contrastive":
            raise NotImplementedError("Contrastive loss not implemented yet")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return {"loss": loss}

    def predict(self, x: Float[Tensor, "N C"]) -> Int[Tensor, "N"]:  # noqa: F821, F722
        pred = self.forward(x)
        logit = torch.matmul(pred, self.emb_target.t())
        return logit.argmax(dim=1)
