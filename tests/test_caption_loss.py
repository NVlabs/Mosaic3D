import unittest

import hydra
import torch
import yaml
from lightning import LightningDataModule
from omegaconf import OmegaConf
from warp.convnet.geometry.point_collection import PointCollection

from src.models.losses.caption_loss import CaptionLoss
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc.utils.caption_utils import get_caption_batch_refactor

text_encoder_str = """text_encoder:
name: CLIP
backbone: ViT-B/16
"""


def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(sub_item, device) for sub_item in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    else:
        return item


class TestCaptionLoss(unittest.TestCase):
    def test_caption_loss(self):
        """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the
        necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch
        sizes correctly match.

        :param batch_size: Batch size of the data to be loaded by the dataloader.
        """
        with open("configs/data/regionplc_base15.yaml") as f:
            omega_config_dict = yaml.safe_load(f.read())
        cfg = OmegaConf.create(omega_config_dict)
        cfg.collate_fn._target_ = "src.data.collate.point_collate_warp_fn"

        cfg.val_dataset = cfg.train_dataset
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        caption_head = CaptionLoss(
            normalize_input=True,
        )

        device = torch.device("cuda:0")
        text_encoder_cfg = OmegaConf.create(text_encoder_str)
        text_encoder = build_text_model(text_encoder_cfg).to(device)

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()

        for i, batch_dict in enumerate(loader):
            assert isinstance(batch_dict, dict)
            batch_dict = to_device(batch_dict, device)
            caption_infos = get_caption_batch_refactor(
                batch_dict["caption_data"], text_encoder, local_rank=0
            )
            batch_dict.update(caption_infos)

            rand_feats = torch.randn(batch_dict["coord"].shape[0], 512)
            pc = PointCollection(
                batch_dict["coord"].cpu(), rand_feats, offsets=batch_dict["offset"].cpu()
            ).to(device)

            loss = caption_head.loss(pc, batch_dict)
            print(loss)
            if i == 0:
                break
