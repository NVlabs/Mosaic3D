import os

import gdown
import torch


ID_CKPTS = {
    "segment3d.ckpt": "1Swq9d7rjV2Q1lTuXiKh1z0OZPt9V4sgO",
}
SKIP_PARAMETERES = (
    "criterion.empty_weight",
    "model.backbone.final.kernel",
    "model.backbone.final.bias",
)
MODULE_MAPPING = {
    "mask_features_head": "decoder_proj",
    "query_projection": "query_proj",
    "mask_embed_head": "mask_head",
    "class_embed_head": "class_head",
    "ffn_attention": "ffn",
    "lin_squeeze": "linear",
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


def patch_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]

    # Create a new dictionary to store the updated key-value pairs
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in SKIP_PARAMETERES:
            continue

        if k.startswith("model."):
            k = k[len("model.") :]

            for before, after in MODULE_MAPPING.items():
                k = k.replace(before, after)

                if k.startswith("decoder_proj"):
                    v = v.squeeze(0)

                if k.startswith("query_proj"):
                    v = v.squeeze(-1)

            new_state_dict[k] = v

    new_ckpt_path = ckpt_path.replace(".ckpt", "_patched.ckpt")
    torch.save(new_state_dict, new_ckpt_path)


if __name__ == "__main__":
    download_ckpt()
    patch_ckpt("ckpts/segment3d.ckpt")
