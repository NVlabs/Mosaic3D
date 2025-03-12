import torch

from src.models.networks.segment3d.download_ckpt import download_ckpt


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

            new_state_dict[k] = v

    new_ckpt_path = ckpt_path.replace(".ckpt", "_patched.ckpt")
    torch.save(ckpt, new_ckpt_path)


if __name__ == "__main__":
    download_ckpt()
    patch_ckpt("ckpts/segment3d.ckpt")
