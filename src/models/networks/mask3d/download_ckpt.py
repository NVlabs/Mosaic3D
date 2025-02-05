import os
from pathlib import Path

import gdown

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


if __name__ == "__main__":
    download_ckpt()
    download_openins3d_masks()
