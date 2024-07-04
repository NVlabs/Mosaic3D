import os

import torch
from clip import clip

from src.models.regionplc.utils import commu_utils
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_clip_model(backbone_name, local_rank=0):
    url = clip._MODELS[backbone_name]
    if local_rank == 0:  # only download once at master node
        model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    else:
        model_path = _return_clip_path(url, os.path.expanduser("~/.cache/clip"))
    commu_utils.synchronize()

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict)
    return clip.tokenize, model


def _return_clip_path(url: str, root: str):
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    return download_target


def build_text_model(model_cfg):
    tokenizer, text_encoder = get_clip_model(model_cfg.backbone)

    text_encoder.tokenizer = tokenizer
    return text_encoder


def load_text_embedding_from_path(text_emb_path):
    text_embedding = torch.load(text_emb_path, map_location=torch.device("cpu")).detach()
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    log.info(f"=> loaded text embedding from path '{text_emb_path}'")
    return text_embedding


def is_bg_class(c):
    return (
        (c.lower() == "wall")
        or (c.lower() == "floor")
        or (c.lower() == "ceiling")
        or (c.lower() == "otherfurniture")
    )


# def build_text_token_from_class_names(model_cfg, class_names):
#     if model_cfg.TEMPLATE == "lseg":  # only instance classes are encoded with prompt
#         return [
#             template_meta[model_cfg.TEMPLATE][0].format(c) if not is_bg_class(c) else c
#             for c in class_names
#         ]
#     else:
#         return [template_meta[model_cfg.TEMPLATE][0].format(c) for c in class_names]
