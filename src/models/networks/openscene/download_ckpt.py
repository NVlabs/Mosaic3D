import os

import torch

ID_CKPTS = {
    "openscene_lseg.ckpt": "https://cvg-data.inf.ethz.ch/openscene/models/scannet_lseg.pth.tar",
    "openscene_openseg.ckpt": "https://cvg-data.inf.ethz.ch/openscene/models/scannet_openseg.pth.tar",
}


def download_ckpt():
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts", exist_ok=True)

    for name, download_url in ID_CKPTS.items():
        if os.path.exists(f"ckpts/{name}"):
            print(f"ckpt {name} already exists")
            continue

        os.system(f"wget {download_url} -O ckpts/{name}")
        preprocess_ckpt(f"ckpts/{name}")


def preprocess_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]

    # Create a new dictionary to store the updated key-value pairs
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("net3d"):
            new_k = k.replace("net3d.", "")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    # Replace the old state_dict with the new one
    ckpt["state_dict"] = new_state_dict

    torch.save(ckpt, ckpt_path)


if __name__ == "__main__":
    download_ckpt()
