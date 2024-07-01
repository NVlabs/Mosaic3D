import os

import torch
from clip import clip

# from ...utils import commu_utils


def get_clip_model(backbone_name, local_rank=0):
    url = clip._MODELS[backbone_name]
    if local_rank == 0:  # only download once at master node
        model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    else:
        model_path = _return_clip_path(url, os.path.expanduser("~/.cache/clip"))
    # commu_utils.synchronize()

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
