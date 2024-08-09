import json
import os
import pickle

import fire
import numpy as np
from natsort import natsorted
from rich.progress import track

# results
# - scene_name # scenexxxx_xx
# - image_name # 00xxxx
# - image_path # /lustre/...../color/00xxxx.jpg
# - captions_raw [optional]
# - captions # (N)
# - point_indices # (M)
# - num_points # (N, )
# - bounding_boxes [optional]  # (N, 4)
# - masks [optional] # (N, H, W)


def main(image_corr_path: str, outdir: str):
    name = (
        os.path.basename(image_corr_path).replace("scannet_caption_idx_", "").replace(".pkl", "")
    )
    outdir = os.path.join(outdir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(image_corr_path, "rb") as f:
        image_corr_all = pickle.load(f)

    caption_path = (
        image_corr_path.replace("image_corr", "caption")
        .replace("scannet_caption_idx", "caption")
        .replace(".pkl", ".json")
    )

    with open(caption_path) as f:
        captions_all = json.load(f)

    for image_corr in track(image_corr_all):
        scene_name = image_corr["scene_name"]
        infos = image_corr["infos"]

        object_ids = natsorted(infos.keys())

        save_dict = dict(
            object_ids=object_ids,
            point_indices=np.concatenate([infos[id].numpy() for id in object_ids]),
            num_points=np.array([infos[id].shape[0] for id in object_ids]),
            captions=np.array([captions_all[scene_name][id] for id in object_ids]),
        )
        filepath = os.path.join(outdir, f"{scene_name}.npz")
        np.savez_compressed(filepath, **save_dict)


if __name__ == "__main__":
    fire.Fire(main)
