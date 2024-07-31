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

DATASET_DIR = "/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/"


def main(
    image_corr_path: str,
    caption_path: str,
    outdir: str,
):
    with open(image_corr_path, "rb") as f:
        image_corr_all = pickle.load(f)

    with open(caption_path) as f:
        captions_all = json.load(f)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for image_corr in track(image_corr_all):
        scene_name = image_corr["scene_name"]
        infos = image_corr["infos"]
        boxes = image_corr["boxes"]

        image_ids = natsorted(boxes.keys())
        image_view_ids = natsorted(infos.keys())
        for image_id in image_ids:
            filepath = os.path.join(outdir, f"{scene_name}@{int(image_id):06d}.npz")
            bounding_boxes = boxes[image_id]
            image_view_ids_sel = [i for i in image_view_ids if i.startswith(image_id + "_")]
            point_indices_list = [infos[i] for i in image_view_ids_sel]
            num_points = np.array([p.shape[0] for p in point_indices_list])
            point_indices = np.concatenate(point_indices_list)
            save_dict = {
                "scene_name": scene_name,
                "image_name": image_id,
                "image_path": os.path.join(
                    DATASET_DIR,
                    "/scannet_frames/tasks/scannet_frames_25k/",
                    scene_name,
                    "color",
                    f"{image_id}.jpg",
                ),
                "point_indices": point_indices.astype(int),
                "num_points": num_points.astype(int),
                "captions": np.array([captions_all[scene_name][i] for i in image_view_ids_sel]),
            }
            np.savez_compressed(filepath, **save_dict)


if __name__ == "__main__":
    fire.Fire(main)
