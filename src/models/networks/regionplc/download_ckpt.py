import os
from collections import OrderedDict

import torch

ID_CKPTS = {
    "regionplc_sparseunet16.ckpt": r"https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EWqYW58Q0GlKuhcAmWZYakUBYV0wyWfbxSarMHo0EZLfMg?e=1Ipfka\&download\=1",
    "regionplc_sparseunet32.ckpt": r"https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EdeusrJ9OnROsFYLd3E1vQMBdfydH5z5L7674K8tG4gYwQ?e=BTWn6j\&download\=1",
    "regionplc-openscene_sparseunet16.ckpt": r"https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EVdEJluknNdJhJ4H-oj8nrIBVIXBhZ7Wjw1m5nU68eM3AA?e=Y2V1pk\&download\=1",
    "regionplc-openscene_sparseunet32.ckpt": r"https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jhyang13_connect_hku_hk/EeleQjMKupFHnRDliBfCidIBLhC3xyewUJH4BSQTuh55HQ?e=ytbvVd\&download\=1",
}


def download_ckpt():
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts", exist_ok=True)

    for name, id_ckpt in ID_CKPTS.items():
        if os.path.exists(f"ckpts/{name}"):
            print(f"ckpt {name} already exists")
            continue

        os.system(f"wget {id_ckpt} -O ckpts/{name}")
        patch(f"ckpts/{name}")


def patch(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[f"net.{k.replace('backbone_3d.', 'backbone.')}"] = v

    torch.save(dict(state_dict=new_state_dict), ckpt_path.replace(".ckpt", "_patched.ckpt"))


if __name__ == "__main__":
    download_ckpt()
