import torch

from src.utils import RankedLogger

from . import text_models
from .prompt_template import template_meta

log = RankedLogger(__name__, rank_zero_only=True)


def build_text_model(model_cfg):
    tokenizer, text_encoder = getattr(text_models, f"get_{model_cfg.NAME.lower()}_model")(
        model_cfg.BACKBONE
    )

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


def build_text_token_from_class_names(model_cfg, class_names):
    if model_cfg.TEMPLATE == "lseg":  # only instance classes are encoded with prompt
        return [
            template_meta[model_cfg.TEMPLATE][0].format(c) if not is_bg_class(c) else c
            for c in class_names
        ]
    else:
        return [template_meta[model_cfg.TEMPLATE][0].format(c) for c in class_names]
