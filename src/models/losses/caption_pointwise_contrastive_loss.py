from typing import Optional, List, Literal
from jaxtyping import Float, Int

import math
import numpy as np

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch.distributed as dist

from src.cuda.load import CupyKernel, load_kernel
from src.utils import dist_utils

bsearch_kernel = CupyKernel("find_first_gt_bsearch.cu", "find_first_gt_bsearch_arange")


class CaptionPointwiseContrastiveLoss:
    def __init__(
        self,
        normalize: bool = True,
        use_logit_scale: Optional[bool] = False,
        **kwargs,
    ):
        self.normalize = normalize
        self.loss_func = nn.NLLLoss(reduction="none")
        self.use_logit_scale = use_logit_scale
        if use_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

        self.kwargs = kwargs

    @torch.no_grad()
    def offset_to_indices(
        self, caption_offsets: Int[Tensor, "M + 1"]  # noqa: F821
    ) -> Int[Tensor, "M"]:  # noqa: F821
        """
        Convert the caption offsets to indices.
        """
        N = caption_offsets[-1].item()

        # Launch parameters
        threads = 256
        blocks = math.ceil(N / threads)
        M = caption_offsets.shape[0]
        shared_mem_bytes = M * caption_offsets.element_size()  # M * sizeof(int)
        torch_out = torch.empty(N, dtype=torch.int32, device=caption_offsets.device)

        # Launch binary-search kernel
        bsearch_kernel(
            (blocks,),
            (threads,),
            (caption_offsets.int().data_ptr(), M, N, torch_out.data_ptr()),
            shared_mem=shared_mem_bytes,
        )
        # Subtract 1 from the output
        torch_out = torch_out - 1

        return torch_out

    def loss(
        self,
        point_features: Float[Tensor, "N C"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "M + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "M"],  # noqa: F821
        captions: List[List[str]],
        clip_encoder: "CLIPTextEncoderInterface",  # noqa: F821
        **kwargs,
    ) -> Tensor:
        """
        point_features: 3D features of the points
        point_indices: defines the point indices for objects
        caption_offsets: defines the offsets of point indices.
            point_indices[caption_offsets[i]:caption_offsets[i+1]]
            defines the point indices for the i-th object.
        num_points_per_caption: number of points in M objects
        captions: B batch of lists of captions. Total M captions.
        """
        # flatten captions
        flat_captions = [c for captions_batch in captions for c in captions_batch]
        # Clone the features to use it in autograd
        text_features: Float[Tensor, "C M"] = (  # noqa: 722
            clip_encoder(flat_captions).T.to(point_features).clone()
        )

        # normalize point features
        if self.normalize:
            point_features = nn.functional.normalize(point_features, dim=-1)

        # Define the logits for all possible point-caption pairs
        # Logits
        logits: Float[Tensor, "N M"] = point_features @ text_features  # noqa: F722
        if self.use_logit_scale:
            logits = self.logit_scale.exp() * logits
        # Scores
        scores = F.log_softmax(logits, dim=-1)

        # Generate labels for all logits
        # convert the caption_offsets to a tensor of shape L.
        labels_per_segment = self.offset_to_indices(caption_offsets)
        loss = self.loss_func(scores[point_indices], labels_per_segment.long())

        return loss.mean()


class AllGatherCaptionPointwiseContrastiveLoss(CaptionPointwiseContrastiveLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world_size = None
        self.curr_rank = None

    def loss(
        self,
        point_features: Float[Tensor, "N C"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "M + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "M"],  # noqa: F821
        captions: List[List[str]],
        clip_encoder: "CLIPTextEncoderInterface",  # noqa: F821
        **kwargs,
    ) -> Tensor:
        """
        point_features: 3D features of the points
        point_indices: defines the point indices for objects
        caption_offsets: defines the offsets of point indices.
            point_indices[caption_offsets[i]:caption_offsets[i+1]]
            defines the point indices for the i-th object.
        num_points_per_caption: number of points in M objects
        captions: B batch of lists of captions. Total M captions.
        """
        if self.world_size is None:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.curr_rank = dist.get_rank() if dist.is_initialized() else 0

        # Same loss, but we gather the text features and concatenate them at the end to make the contrastive loss more effective.
        # flatten captions
        flat_captions = [c for captions_batch in captions for c in captions_batch]
        # Clone the features to use it in autograd
        text_features: Float[Tensor, "C M"] = (  # noqa: 722
            clip_encoder(flat_captions).T.to(point_features).clone()
        )

        # normalize point features
        if self.normalize:
            point_features = nn.functional.normalize(point_features, dim=-1)

        # Gather the text features
        if self.world_size > 1:
            # Gather the text features
            all_text_features: List[Tensor] = dist_utils.differentiable_all_gather_varshapes(
                text_features, self.world_size, self.curr_rank
            )
            # Concatenate the text features.
            # reorder the text features to [Current rank, *all others]
            all_text_features.pop(self.curr_rank)
            all_text_features.insert(0, text_features)
            text_features: Float[Tensor, "C W*M"] = torch.cat(  # noqa: F722
                all_text_features, dim=-1
            )

        logits: Float[Tensor, "N M"] = point_features @ text_features  # noqa: F722
        if self.use_logit_scale:
            logits = self.logit_scale.exp() * logits

        # Scores
        scores = F.log_softmax(logits, dim=-1)

        # The labels are the same since the other ranks are concatenated at the end.
        labels_per_segment = self.offset_to_indices(caption_offsets)
        loss = self.loss_func(scores[point_indices], labels_per_segment.long())

        return loss.mean()
