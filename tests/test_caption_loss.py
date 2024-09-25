import os
import unittest

import hydra
import torch
import yaml
from lightning import LightningDataModule
from omegaconf import OmegaConf

import warp as wp
from src.models.losses.caption_loss import (
    CaptionAlignmentLoss,
    CaptionLoss,
    DenseCaptionAlignmentLoss,
)
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc.utils.caption_utils import (
    get_caption_batch,
    get_unique_caption_batch,
)
from warpconvnet.geometry.point_collection import PointCollection

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
    data_path = "/datasets"

    def setUp(self):
        with open("configs/data/regionplc_base15.yaml") as f:
            omega_config_dict = yaml.safe_load(f.read())
        cfg = OmegaConf.create(omega_config_dict)
        cfg.train_dataset.data_dir = os.path.join(self.data_path, "scannet_hf")
        cfg.train_dataset.caption_dir = os.path.join(self.data_path, "regionplc_converted")
        cfg.val_dataset = cfg.train_dataset

        # Set the data_path in the configuration

        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        self.device = torch.device("cuda:0")
        text_encoder_cfg = OmegaConf.create(text_encoder_str)
        text_encoder = build_text_model(text_encoder_cfg).to(self.device)

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()

        self.loader = loader
        self.text_encoder = text_encoder

    def test_caption_loss(self):
        """Tests the caption loss."""

        caption_loss = CaptionLoss(use_logit_scale=True)
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

            loss = caption_loss.loss(
                pc.feature_tensor,
                unique_caption_embeds=unique_caption_embed,
                caption_targets=caption_target,
                batched_list_of_point_indices=batch_dict["caption_data"]["idx"],
                input_batch_offsets=batch_dict["offset"],
                valid_mask=None,
            )
            print(loss)
            if i == 0:
                break

    def test_caption_alignment_loss(self):
        """Tests the caption alignment loss."""

        caption_head = CaptionAlignmentLoss()

        loader = self.loader
        device = self.device
        text_encoder = self.text_encoder

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
                valid_mask=None,
            )
            print(loss)
            if i == 0:
                break

    def test_dense_caption_alignment_loss(self):
        """Tests the dense caption alignment loss."""

        caption_head = DenseCaptionAlignmentLoss()

        loader = self.loader
        device = self.device
        text_encoder = self.text_encoder

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
                valid_mask=None,
            )
            print(loss)
            if i == 0:
                break


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run caption loss tests")
    parser.add_argument("--data_path", type=str, default="/datasets", help="Path to the data")
    args = parser.parse_args()

    # Set the data_path as a class attribute
    TestCaptionLoss.data_path = args.data_path

    # Print the data_path to verify it's set correctly
    print(f"Data path set to: {TestCaptionLoss.data_path}")

    # Create a test suite with our TestCaptionLoss class
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaptionLoss)

    # Run the tests
    wp.init()
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    main()
