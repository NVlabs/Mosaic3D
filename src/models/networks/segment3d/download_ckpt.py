import os
import gdown

import torch

ID_CKPTS = {
    "segment3d.ckpt": "1Swq9d7rjV2Q1lTuXiKh1z0OZPt9V4sgO",
}
ID_SAMPLES = [
    "1Ffe5jcItYmZHpb875Z3V48wsqsAAnDp9",
    "1pKmMo9zp9PUUdz2VO3fFNHNbdtu1nDcH",
]


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


def download_sample():
    if not os.path.exists("samples"):
        os.makedirs("samples", exist_ok=True)

    for id_sample in ID_SAMPLES:
        if os.path.exists(f"samples/{id_sample}"):
            print(f"sample {id_sample} already exists")
            continue
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{id_sample}",
            output=f"samples/{id_sample}",
            quiet=False,
            use_cookies=False,
        )


def preprocess_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]

    # Create a new dictionary to store the updated key-value pairs
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in [
            "criterion.empty_weight",
            "model.backbone.final.kernel",
            "model.backbone.final.bias",
        ]:
            continue

        if k.startswith("model."):
            new_k = k.replace("model.", "")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    ckpt["state_dict"] = new_state_dict
    new_ckpt_path = ckpt_path.replace(".ckpt", "_processed.ckpt")
    torch.save(ckpt, new_ckpt_path)


if __name__ == "__main__":
    # download_ckpt()
    # download_sample()
    preprocess_ckpt("ckpts/segment3d.ckpt")
