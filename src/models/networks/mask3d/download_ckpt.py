import os

ID_CKPTS = {
    "mask3d-scannet200_val.ckpt": "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt",
    "mask3d-scannet20_val.ckpt": "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_val.ckpt",
}


def download_ckpt():
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts", exist_ok=True)

    for name, id_ckpt in ID_CKPTS.items():
        if os.path.exists(f"ckpts/{name}"):
            print(f"ckpt {name} already exists")
            continue

        os.system(f"wget {id_ckpt} -O ckpts/{name}")


if __name__ == "__main__":
    download_ckpt()
