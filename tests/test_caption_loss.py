import unittest

import hydra
import torch
import warp as wp
import yaml
from lightning import LightningDataModule
from omegaconf import OmegaConf
from warp.convnet.geometry.point_collection import PointCollection

from src.models.losses.caption_loss import CaptionAlignmentLoss, CaptionLoss
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc.utils.caption_utils import (
    get_caption_batch,
    get_unique_caption_batch,
)

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
    def setUp(self):
        with open("configs/data/regionplc_base15.yaml") as f:
            omega_config_dict = yaml.safe_load(f.read())
        cfg = OmegaConf.create(omega_config_dict)
        cfg.val_dataset = cfg.train_dataset
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        self.device = torch.device("cuda:0")
        text_encoder_cfg = OmegaConf.create(text_encoder_str)
        text_encoder = build_text_model(text_encoder_cfg).to(self.device)

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()

        self.loader = loader
        self.text_encoder = text_encoder

    def test_caption_loss(self):
        """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the
        necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch
        sizes correctly match.

        :param batch_size: Batch size of the data to be loaded by the dataloader.
        """

        caption_head = CaptionLoss()

        for i, batch_dict in enumerate(self.loader):
            assert isinstance(batch_dict, dict)
            batch_dict = to_device(batch_dict, self.device)
            captions = batch_dict["caption_data"]["caption"]
            unique_caption_embed, caption_target = get_unique_caption_batch(
                captions, self.text_encoder
            )

            rand_feats = torch.randn(batch_dict["coord"].shape[0], 512)
            pc = PointCollection(
                batch_dict["coord"].cpu(),
                rand_feats,
                offsets=batch_dict["offset"].cpu(),
            ).to(self.device)

            loss = caption_head.loss(
                pc.feature_tensor,
                unique_caption_embeds=unique_caption_embed,
                caption_targets=caption_target,
                batched_list_of_point_indices=batch_dict["caption_data"]["idx"],
                input_batch_offsets=batch_dict["offset"],
                mappings=None,
            )
            print(loss)
            if i == 0:
                break

    def test_caption_alignment_loss(self):
        """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the
        necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch
        sizes correctly match.

        :param batch_size: Batch size of the data to be loaded by the dataloader.
        """
        with open("configs/data/regionplc_base15.yaml") as f:
            omega_config_dict = yaml.safe_load(f.read())
        cfg = OmegaConf.create(omega_config_dict)
        cfg.val_dataset = cfg.train_dataset
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        caption_head = CaptionAlignmentLoss()

        device = torch.device("cuda:0")
        text_encoder_cfg = OmegaConf.create(text_encoder_str)
        text_encoder = build_text_model(text_encoder_cfg).to(device)

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()

        for i, batch_dict in enumerate(loader):
            assert isinstance(batch_dict, dict)
            batch_dict = to_device(batch_dict, device)
            caption_embed = get_caption_batch(batch_dict["caption_data"]["caption"], text_encoder)

            rand_feats = torch.randn(batch_dict["coord"].shape[0], 512)
            pc = PointCollection(
                batch_dict["coord"].cpu(),
                rand_feats,
                offsets=batch_dict["offset"].cpu(),
            ).to(device)

            loss = caption_head.loss(
                pc.feature_tensor,
                caption_embed,
                batched_list_of_point_indices=batch_dict["caption_data"]["idx"],
                input_batch_offsets=batch_dict["offset"],
                mappings=None,
            )
            print(loss)
            if i == 0:
                break


if __name__ == "__main__":
    wp.init()
    unittest.main()
