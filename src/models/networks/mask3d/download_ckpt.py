import os
import gdown

ID_CKPTS = {
    "mask3d.ckpt": "1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B",  # mask3d trained on ScanNet200 train set (scannet200_val.ckpt)
}


def download_ckpt():
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts", exist_ok=True)

    for name, id_ckpt in ID_CKPTS.items():
        if os.path.exists(f"ckpts/{name}"):
            print(f"ckpt {name} already exists")
            continue
        gdown.download(
            f"https://drive.google.com/uc?id={id_ckpt}",
            f"ckpts/{name}",
            quiet=False,
        )


if __name__ == "__main__":
    download_ckpt()
