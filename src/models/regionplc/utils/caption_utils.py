import torch


def get_caption_batch(caption_data, text_encoder, local_rank):
    caption, idx = caption_data["caption"], caption_data["idx"]

    # caption_embed: (K, 512), caption_idx: (N), (N > K)
    caption_embed, caption_idx = extract_caption_embed(caption, text_encoder, local_rank)
    caption_embed = torch.nn.functional.normalize(caption_embed, dim=-1)

    caption_infos = {
        "caption_embed": caption_embed,
        "caption_idx": caption_idx,
        "c2p_map": idx,  # caption 2 point mapping
    }
    return caption_infos


def extract_caption_embed(image_captions, text_encoder, rank):
    num_caption_list = [0] * 100
    num_caption_list[rank] = len(image_captions)
    caption_embed_all = forward_text_encoder(image_captions, text_encoder)

    num_caption_list = torch.LongTensor([0] + num_caption_list).cuda()
    idx = (
        torch.arange(num_caption_list[rank + 1]).long().cuda()
        + torch.cumsum(num_caption_list, 0)[rank]
    )
    caption_embeds, unique_indices = torch.unique(caption_embed_all, dim=0, return_inverse=True)
    caption_idx = unique_indices[idx]

    return caption_embeds, caption_idx


@torch.no_grad()
def forward_text_encoder(image_captions, text_encoder):
    if len(image_captions) == 0:
        return torch.zeros((0, 512), dtype=torch.float32).cuda()

    text_tokens = text_encoder.tokenizer(image_captions, truncate=True).cuda()
    text_embed = text_encoder.encode_text(text_tokens).float()
    return text_embed
