from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float, Int
from torch import Tensor
from warpconvnet.geometry.types.points import Points

import src.utils.caption_utils as caption_utils
from src.models.losses.loss_base import LossBase
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CLIPAlignmentLoss(LossBase):
    """Given the embedding, compute inner product with the target embedding and compute loss."""

    def __init__(
        self,
        normalize_input: bool,
        loss_type: Literal["cross_entropy", "contrastive"],
        text_clip_path: Optional[str] = None,
        ignore_label: int = -100,
        learnable_logit: bool = False,
        eval_only: bool = False,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.eval_only = eval_only

        # load pre-computed text embeddings (e.g. CLIP text embedding with shape NxC)
        self.emb_target = None
        if text_clip_path is not None and Path(text_clip_path).exists():
            text_embeddings = torch.load(text_clip_path, map_location="cpu").detach()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            log.info(f"=> loaded text embeddings from {text_clip_path}")
            self.set_target_embedding(text_embeddings)
        else:
            log.warn(f"Text embedding file not found: {text_clip_path}")

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

    def set_target_embedding(self, text_embeddings: torch.Tensor):
        self.emb_target = text_embeddings.float()

    def forward(self, x: Tensor | Points) -> Tensor:
        if isinstance(x, Points):
            x = x.feature_tensor
        if self.normalize_input:
            return F.normalize(x, p=2, dim=1)
        return x

    def loss(
        self,
        x: Tensor | Points,
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
        assert self.emb_target is not None, "Text embedding is not loaded"

        pred = self.forward(x)
        logit_scale = self.logit_scale
        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = logit_scale.exp()
        logit = torch.matmul(pred, self.emb_target.t()) * logit_scale

        if return_logit:
            return logit

        return logit.argmax(dim=1)


class CLIPAlignmentEval(nn.Module):
    def __init__(self, normalize_input: bool, text_clip_path: Optional[str] = None):
        super().__init__()
        self.normalize_input = normalize_input

        # load pre-computed text embeddings (e.g. CLIP text embedding with shape NxC)
        self.emb_target = None
        if text_clip_path is not None and Path(text_clip_path).exists():
            text_embeddings = torch.load(text_clip_path, map_location="cpu").detach()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            log.info(f"=> loaded text embeddings from {text_clip_path}")
            self.set_target_embedding(text_embeddings)
        else:
            log.warn(f"Text embedding file not found: {text_clip_path}")

    def set_target_embedding(self, text_embeddings: torch.Tensor):
        self.emb_target = text_embeddings.float()

    def forward(self, x: Tensor | Points) -> Tensor:
        if isinstance(x, Points):
            x = x.feature_tensor
        if self.normalize_input:
            return F.normalize(x, p=2, dim=1)
        return x

    def loss(self, *args, **kwargs):
        raise NotImplementedError(
            "CLIPAlignmentEval is for evaluation only, not for computing loss."
        )

    def predict(self, x: Float[Tensor, "N C"], return_logit: bool = False):  # noqa: F821, F722
        assert self.emb_target is not None, "Text embedding is not loaded"

        pred = self.forward(x)
        logit = torch.matmul(pred, self.emb_target.t())

        if return_logit:
            return logit

        return logit.argmax(dim=1)


def compute_clip_image_alignment(
    clip_encoder,
    clip_processed_image: torch.Tensor,
    point_feat: torch.Tensor,
    clip_point_indices: torch.Tensor,
    clip_indices_image_to_point: torch.Tensor,
    is_loss: bool = True,
) -> torch.Tensor:
    """Compute clip_loss using images.

    args:
        clip_encoder:
        clip_processed_image: [b c h w]
        point_feat: [n_pts c]
        clip_point_indices: [n_pts_slice]
            These indices incicate the points visible to the given image.
            In collate.py, we already add offset.
        clip_indices_image_to_point: [n_pts_slice]
            These indices indicate the pixels corresponding to coord (3d_input_points).
    returns:
        clip_image_alignment_loss: []
        clip_scores: [n_pts_slice]
    """
    # prepare image data in bf16
    with torch.cuda.amp.autocast(enabled=True) and torch.inference_mode():
        image_feats = caption_utils.forward_image_encoder(
            clip_processed_image,
            clip_encoder,
        )  # [b c]

    # slice pointcloud visible to images.
    point_feat_slices = point_feat[clip_point_indices]  # [n_pts c]

    # unpool image_feats to per-point feats.
    image_feats_unpooled = image_feats[clip_indices_image_to_point, :]  # [n_pts c]

    # compute per-point cossine similarity between point_feat and image_feat
    # if len(point_feat_slices) != 0:
    target = torch.ones(
        len(point_feat_slices),
        device=point_feat_slices.device,
    )

    if is_loss:
        clip_image_alignment_loss = nn.CosineEmbeddingLoss()(
            point_feat_slices, image_feats_unpooled, target
        )
        return clip_image_alignment_loss
    else:
        clip_scores = F.cosine_similarity(point_feat_slices, image_feats_unpooled)
        return clip_scores


def compute_clip_text_cosine_similarity(
    clip_encoder,
    clip_tokenized_text,
    point_feat: torch.Tensor,
    offset: torch.Tensor,
    point_indices_to_caption: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between predictions and captions.

    args:
        clip_encoder:
        clip_tokenized_text:
            preprocessed text tokenziation following CLIP
        point_feat
        offset
        point_indices_to_caption: List[Tensor]
    returns:
        clip_avg_score: []
    """
    text_feats = clip_encoder.encode_text(clip_tokenized_text)  # [n_captions c]

    # slice pointcloud per caption

    point_feat_slices = [
        point_feat[point_indices_per_caption + offset[idx_batch], :]
        for idx_batch, point_indices_per_batch in enumerate(point_indices_to_caption)
        for point_indices_per_caption in point_indices_per_batch
    ]

    # compute per-point cossine similarity between point_feat and text_feat
    clip_score_sum = 0.0
    cnt = 0
    for point_feat_slice, text_feat in zip(point_feat_slices, text_feats):
        text_feat_unpooled = repeat(  # unpool text_feat to per-point feats
            text_feat, "c -> n_pts_per_caption c", n_pts_per_caption=len(point_feat_slice)
        )
        clip_scores_per_caption = F.cosine_similarity(point_feat_slice, text_feat_unpooled)
        clip_score_sum += clip_scores_per_caption.sum().cpu().numpy()
        cnt += len(clip_scores_per_caption)
    clip_avg_score = clip_score_sum / float(cnt)
    return clip_avg_score
