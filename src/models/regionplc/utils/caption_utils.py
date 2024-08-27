from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor


def get_caption_batch(batched_captions: List[List[str]], text_encoder: nn.Module) -> List[Tensor]:
    # Get the size of each batch
    num_captions_per_batch = [len(captions) for captions in batched_captions]
    # Flatten the caption list
    flat_captions = [caption for sublist in batched_captions for caption in sublist]

    caption_embed = forward_text_encoder(flat_captions, text_encoder)
    caption_embed = torch.nn.functional.normalize(caption_embed, dim=-1)

    # Split the caption_embed into the original batch size
    caption_embeds = torch.split(caption_embed, num_captions_per_batch)
    return caption_embeds


@torch.no_grad()
def forward_text_encoder(image_captions, text_encoder):
    if len(image_captions) == 0:
        return torch.zeros((0, 512), dtype=torch.float32).cuda()

    text_tokens = text_encoder.tokenizer(image_captions, truncate=True).cuda()
    text_embed = text_encoder.encode_text(text_tokens).float()
    return text_embed
