from pathlib import Path

import hydra
import pytest
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data.regionplc.scannet_dataset import ScanNetDataset
from src.models.heads.clip_alignment_head import FixedCLIPAlignmentHead


def yaml_to_dict(yaml_str: str) -> dict:
    """Converts a YAML string to a dictionary."""
    return yaml.safe_load(yaml_str)


@pytest.mark.parametrize("split", ["train"])
def test_clip_alignment(split: str) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = Path("~/datasets/regionplc").expanduser()
    project_root = Path(__file__).resolve().parents[1]
    config_path = "configs/tests/scannet.yaml"
    with open(project_root / config_path) as f:
        omega_config_dict = yaml_to_dict(f.read())
    cfg = OmegaConf.create(omega_config_dict)
    cfg.data_dir = str(data_dir)
    cfg.split = split

    dataset = ScanNetDataset(
        data_dir=cfg.data_dir,
        split=cfg.split,
        class_names=cfg.class_names,
        repeat=cfg.repeat,
        point_range=cfg.point_range,
        voxel_scale=cfg.voxel_scale,
        max_npoint=cfg.max_npoint,
        full_scale=cfg.full_scale,
        rgb_norm=cfg.rgb_norm,
        xyz_as_feat=cfg.xyz_as_feat,
        rgb_as_feat=cfg.rgb_as_feat,
        min_spatial_shape=cfg.min_spatial_shape,
        ignore_label=cfg.ignore_label,
        base_class_idx=cfg.base_class_idx,
        novel_class_idx=cfg.novel_class_idx,
        ignore_class_idx=cfg.ignore_class_idx,
        aug_cfg=cfg.aug_cfg,
        # caption_cfg=cfg.caption_cfg,
    )
    assert dataset is not None

    assert len(dataset) > 0

    alignment_head = FixedCLIPAlignmentHead(
        text_clip_path=Path(cfg.data_dir) / cfg.text_embed_path,
        loss_type="cross_entropy",
        normalize_input=True,
    )

    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, dict)
        print(sample.keys())

        labels = torch.LongTensor(sample["labels"])
        rand_feats = torch.randn(labels.shape[0], 512)
        loss = alignment_head.loss(rand_feats, labels)
        if i == 0:
            break


if __name__ == "__main__":
    test_clip_alignment("train")
