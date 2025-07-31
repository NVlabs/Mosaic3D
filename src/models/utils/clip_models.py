import os

import open_clip
import torch
from clip import clip
from open_clip.factory import download_pretrained_from_hf

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def download_clip_model(model_cfg):
    model_id = model_cfg.get("model_id", None)
    if model_id is None:
        model_id = model_cfg.backbone

    cache_dir = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/"))

    if model_id.startswith("hf-hub:"):
        ckpt_path = download_pretrained_from_hf(model_id[len("hf-hub:") :], cache_dir=cache_dir)
    else:
        url = clip._MODELS[model_id]
        ckpt_path = clip._download(url, root=cache_dir)

    return ckpt_path


def build_clip_model(model_cfg, device=None):
    model_id = model_cfg.get("model_id", None)
    # backward compatibility
    if model_id is None:
        model_id = model_cfg.backbone

    cache_dir = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/"))

    if model_id.startswith("hf-hub:"):
        model, preprocess = open_clip.create_model_from_pretrained(model_id, device=device)
        tokenizer = open_clip.get_tokenizer(model_id)
    else:
        model, preprocess = clip.load(model_id, device=device, download_root=cache_dir)
        tokenizer = lambda x: clip.tokenize(x, truncate=True)  # noqa: E731

    model.image_tokenizer = preprocess
    model.text_tokenizer = tokenizer
    return model
