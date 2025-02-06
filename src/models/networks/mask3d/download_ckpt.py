import os
from pathlib import Path

import gdown
from natsort import natsorted
from tqdm import tqdm
import torch
import numpy as np


ID_CKPTS = {
    "mask3d-scannet200_val.ckpt": "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt",
    "mask3d-scannet20_val.ckpt": "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_val.ckpt",
}
MASK_DIR = Path("/datasets/scannet_masks")


def download_ckpt():
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts", exist_ok=True)

    for name, id_ckpt in ID_CKPTS.items():
        if os.path.exists(f"ckpts/{name}"):
            print(f"ckpt {name} already exists")
            continue

        os.system(f"wget {id_ckpt} -O ckpts/{name}")


def download_openins3d_masks():
    if not os.path.exists(MASK_DIR):
        MASK_DIR.mkdir(parents=True, exist_ok=True)

    gdown.download(
        "https://drive.google.com/uc?id=1NfnNBTvapKHi_ZrIrBiKrW8VSKzYc-V_",
        str(MASK_DIR / "openins3d.zip"),
        quiet=False,
    )

    # unzip
    os.system(f"unzip {MASK_DIR / 'openins3d.zip'} -d {MASK_DIR / 'openins3d'}")

    # remove all files except detected masks
    os.system(
        f"mv {MASK_DIR / 'openins3d' / 'masks' / 'detected'} {MASK_DIR / 'mask3d_scannet200_openins3d'}"
    )
    os.system(f"rm -r {MASK_DIR / 'openins3d'}")
    os.system(f"rm {MASK_DIR / 'openins3d.zip'}")


def convert_openins3d_masks():
    mask_dir = MASK_DIR / "mask3d_scannet200_openins3d"
    assert mask_dir.exists(), f"Mask directory {mask_dir} does not exist"

    out_dir = MASK_DIR / "mask3d_scannet200_openins3d_converted"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_files = natsorted(mask_dir.glob("*.pt"))
    for masks_file in tqdm(masks_files):
        scene_id = masks_file.stem
        scene_dir = out_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # sparse to dense
        masks_coo = torch.load(masks_file)
        masks = masks_coo.to_dense().numpy().T

        # convert to point_indices (flattened)
        point_indices_all = []
        lengths_all = []
        for mask in masks:
            point_indices = np.where(mask)[0].tolist()
            point_indices_all.extend(point_indices)
            lengths_all.append(len(point_indices))

        point_indices_all = np.array(point_indices_all).astype(np.int32)
        lengths_all = np.array(lengths_all).astype(np.int64)

        # save
        np.savez(scene_dir / "point_indices.npz", packed=point_indices_all, lengths=lengths_all)


if __name__ == "__main__":
    # download_ckpt()
    # download_openins3d_masks()
    convert_openins3d_masks()
