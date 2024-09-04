from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


def get_caption_batch(
    batched_captions: List[List[str]], clip_encoder: nn.Module
) -> List[Float[Tensor, "N 512"]]:  # noqa: F821, F722
    # Get the size of each batch
    num_captions_per_batch = [len(captions) for captions in batched_captions]

    # Flatten the caption list
    flat_captions = [caption for sublist in batched_captions for caption in sublist]

    caption_embed = forward_text_encoder(flat_captions, clip_encoder)
    caption_embed = torch.nn.functional.normalize(caption_embed, dim=-1)

    # Split the caption_embed into the original batch size
    caption_embeds = torch.split(caption_embed, num_captions_per_batch)
    return caption_embeds


def get_unique_caption_batch(
    batched_captions: List[List[str]], clip_encoder: nn.Module
) -> Tuple[Float[Tensor, "N 512"], Int[Tensor, "M"]]:  # noqa: F821, F722
    # Flatten the caption list
    flat_captions = [caption for sublist in batched_captions for caption in sublist]
    flat_caption_hash = [hash(caption) for caption in flat_captions]

    # Get unique captions and their indices
    _, to_unique_indices, from_unique_indices = np.unique(
        flat_caption_hash, return_index=True, return_inverse=True
    )

    unique_captions = [flat_captions[i] for i in to_unique_indices]
    caption_embeds = forward_text_encoder(unique_captions, clip_encoder)
    caption_embeds = torch.nn.functional.normalize(caption_embeds, dim=-1)

    return (
        caption_embeds,  # embedding
        torch.tensor(from_unique_indices),  # target
    )


@torch.no_grad()
def forward_text_encoder(image_captions, clip_encoder):
    if len(image_captions) == 0:
        return torch.zeros((0, 512), dtype=torch.float32).cuda()

    text_tokens = clip_encoder.text_tokenizer(image_captions, truncate=True).cuda()
    text_embed = clip_encoder.encode_text(text_tokens).float()
    return text_embed


def forward_image_encoder(preprocessed_images, clip_encoder):
    ''' compute clip feature from images
    args:
        preprocessed_images: [b c h w]
        clip_encoder:
    return:
        image_features: [b c]
    '''
    image_features = clip_encoder.encode_image(preprocessed_images)
    return image_features
