import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict as edict

from src.models.components.clip_models import build_clip_model
from src.utils.caption_utils import forward_text_encoder

CLIP_CFG = {
    "b16": dict(model_id="ViT-B/16", text_dim=512),
    "recapclip": dict(model_id="hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP", text_dim=768),
}

SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]


def main(
    scene_id: int,
    data_dir: str = "./eval_eth3d/spunet34c+ppt+all/pred",
    clip_name: str = "recapclip",
    prompt: str = None,
):
    device = torch.device("cuda")
    scene_name = SCENES[scene_id]

    text_prompt = prompt.split("@")
    print(f"Text prompt: {text_prompt}")

    pred_path = os.path.join(data_dir, f"pred_{scene_id}.pth")
    pred = torch.load(pred_path, map_location="cpu")
    coord = pred["coord"]
    pc_count = pred["pc_count"]
    print(f"Loaded {pred_path}, num points: {pc_count}")
    # normalize feat
    feat = pred["feat"].to(device)
    feat = feat / feat.norm(dim=-1, keepdim=True)

    # extract clip feat
    cfg = edict(CLIP_CFG[clip_name])
    model = build_clip_model(cfg, device=device)

    for i, p in enumerate(text_prompt):
        # get clip feat
        print(f"[{i}/{len(text_prompt)}] Getting clip feat for {p}")
        clip_feat = forward_text_encoder(p, model, normalize=True, device=device)

        # get correlation
        print(f"[{i}/{len(text_prompt)}] Getting correlation between feat and clip feat")
        corr = torch.matmul(feat, clip_feat.T)
        corr = corr.cpu().numpy()

        # color mapping with viridis color map
        corr = (corr - corr.min()) / (corr.max() - corr.min())
        corr = corr**3
        corr = plt.cm.viridis(corr.squeeze())[:, :3]

        # visualize
        print(f"[{i}/{len(text_prompt)}] Visualizing")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(corr)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.05)

        out_path = os.path.join(data_dir, f"{scene_id}_{scene_name}_{p.replace(' ', '_')}.ply")
        o3d.io.write_point_cloud(out_path, pcd_down)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
