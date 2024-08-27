import unittest
from pathlib import Path

import hydra
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from warp.convnet.geometry.point_collection import PointCollection

from src.data.regionplc.scannet_dataset import ScanNetDataset
from src.models.losses.clip_alignment_loss import CLIPAlignmentLoss

EMBED_PATH = "/datasets/regionplc/text_embed/scannet_clip-ViT-B16_id.pth"


class TestCLIPAlignmentLoss(unittest.TestCase):
    def test_clip_alignment(self, split: str = "train") -> None:
        with open("configs/data/regionplc_base15.yaml") as f:
            omega_config_dict = yaml.safe_load(f.read())
        cfg = OmegaConf.create(omega_config_dict)
        cfg.val_dataset = cfg.train_dataset
        datamodule = hydra.utils.instantiate(cfg)

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()

        alignment_head = CLIPAlignmentLoss(
            text_clip_path=EMBED_PATH,
            loss_type="cross_entropy",
            normalize_input=True,
        )

        for data_dict in loader:
            assert isinstance(data_dict, dict)
            print(data_dict.keys())

            labels = torch.LongTensor(data_dict["labels"])
            rand_feats = torch.randn(labels.shape[0], 512)
            loss = alignment_head.loss(rand_feats, labels)
            print(loss)

            # PointCollection test
            pc = PointCollection(
                data_dict["coord"],
                rand_feats,
                offsets=data_dict["offset"],
            )
            loss = alignment_head.loss(pc, labels)
            print(loss)
            break
