import hashlib
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch_scatter import segment_csr


def get_caption_batch(
    batched_captions: List[List[str]],
    clip_encoder: nn.Module,
    is_entity: bool = False,
    interpolate: bool = False,
) -> List[Float[Tensor, "N 512"]]:  # noqa: F821, F722
    # Get the size of each batch
    num_captions_per_batch = [len(captions) for captions in batched_captions]

    # Flatten the caption list
    flat_captions = [caption for sublist in batched_captions for caption in sublist]

    if is_entity:
        caption_embed = forward_entity_text_encoder(flat_captions, clip_encoder, interpolate)
    else:
        caption_embed = forward_text_encoder(flat_captions, clip_encoder)

    caption_embed = torch.nn.functional.normalize(caption_embed, dim=-1)

    # Split the caption_embed into the original batch size
    caption_embeds = torch.split(caption_embed, num_captions_per_batch)
    return caption_embeds


# Use a deterministic hash function for strings
def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def get_unique_caption_batch(
    batched_captions: List[List[str]],
    clip_encoder: nn.Module,
    is_entity: bool = False,
    interpolate: bool = False,
) -> Tuple[Float[Tensor, "N 512"], Int[Tensor, "M"]]:  # noqa: F821, F722
    # Flatten the caption list
    flat_captions = [caption for sublist in batched_captions for caption in sublist]
    flat_caption_hash = [string_hash(caption) for caption in flat_captions]

    # Get unique captions and their indices
    _, to_unique_indices, from_unique_indices = np.unique(
        flat_caption_hash, return_index=True, return_inverse=True
    )

    unique_captions = [flat_captions[i] for i in to_unique_indices]

    if is_entity:
        caption_embeds = forward_entity_text_encoder(unique_captions, clip_encoder, interpolate)
    else:
        caption_embeds = forward_text_encoder(unique_captions, clip_encoder)

    caption_embeds = forward_text_encoder(unique_captions, clip_encoder)
    caption_embeds = torch.nn.functional.normalize(caption_embeds, dim=-1)

    return (
        caption_embeds,  # embedding
        torch.tensor(from_unique_indices),  # target
        torch.tensor(to_unique_indices),  # target
    )


@torch.no_grad()
def forward_text_encoder(
    image_captions,
    clip_encoder,
    normalize: bool = False,
    device: torch.device = torch.device("cuda"),
):
    if len(image_captions) == 0:
        # Get the channel size from the clip_encoder
        channel_size = clip_encoder.text_projection.shape[1]
        return torch.zeros((0, channel_size), dtype=torch.float32).to(device)

    text_tokens = clip_encoder.text_tokenizer(image_captions).to(device)
    text_embed = clip_encoder.encode_text(text_tokens).float()
    if normalize:
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
    return text_embed


@torch.no_grad()
def forward_entity_text_encoder(
    entity_texts,
    clip_encoder,
    interpolate: bool = False,
    device: torch.device = torch.device("cuda"),
):
    """EntityText = f"{entity_name}:{entity_description1},{entity_description2},..."."""

    if len(entity_texts) == 0:
        return torch.zeros((0, 512), dtype=torch.float32).to(device)

    def parse(text):
        name, descriptions = text.split(":")
        descriptions = descriptions.split(",")
        if interpolate:
            descriptions = [f"{name} {desc}" for desc in descriptions]
        return [name] + descriptions

    parsed_entity_texts = []
    offsets = [0]
    for entity_text in entity_texts:
        parsed_entity_text = parse(entity_text)
        parsed_entity_texts.extend(parsed_entity_text)
        offsets.append(offsets[-1] + len(parsed_entity_text))

    text_embed = forward_text_encoder(parsed_entity_texts, clip_encoder, device=device)
    text_embed = segment_csr(
        text_embed, torch.tensor(offsets, device=text_embed.device), reduce="mean"
    )

    return text_embed


def forward_image_encoder(preprocessed_images, clip_encoder):
    """compute clip feature from images
    args:
        preprocessed_images: [b c h w]
        clip_encoder:
    return:
        image_features: [b c]
    """
    image_features = clip_encoder.encode_image(preprocessed_images)
    return image_features
