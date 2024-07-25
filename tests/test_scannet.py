from pathlib import Path

import hydra
import pytest
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data.regionplc.scannet_dataset import ScanNetDataset

omega_config_str = """
_target_: src.data.regionplc.scannet_dataset.ScanNetDataset
_partial_: True
data_dir: /datasets/regionplc
split: train
# class labels
base_class_idx: [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18]
ignore_class_idx: [19]
novel_class_idx: [5, 9, 12, 16]
ignore_label: -100
class_names:
  [
    wall,
    floor,
    cabinet,
    bed,
    chair,
    sofa,
    table,
    door,
    window,
    bookshelf,
    picture,
    counter,
    desk,
    curtain,
    refrigerator,
    showercurtain,
    toilet,
    sink,
    bathtub,
    otherfurniture,
  ]
# augmentations
aug_cfg:
  aug_list: [scene_aug, elastic, crop, shuffle]
  scene_aug:
    scaling_scene:
      enabled: False
      p: 1.0
      value: [0.9, 1.1]

    rotation:
      p: 1.0
      value: [0.0, 0.0, 1.0]

    jitter: True
    color_jitter: True

    flip:
      p: 0.5

    random_jitter:
      enabled: False
      value: 0.01
      accord_to_size: False
      p: 1.0

  elastic:
    enabled: True
    value: [[6, 40], [20, 160]]
    apply_to_feat: False
    p: 1.0

  crop:
    step: 32

  shuffle: True
# processor
repeat: 4
rgb_norm: True
point_range: 200000000
voxel_scale: 50 # voxel_size = 1 / scale, scale 25(0.02m)
max_npoint: 250000
full_scale: [128, 512]
xyz_as_feat: True
rgb_as_feat: True
min_spatial_shape: 128

# caption cfg
caption_cfg:
  KEY: [SCENE, VIEW, ENTITY]

  SCENE:
    ENABLED: False

  VIEW:
    ENABLED: True
    CAPTION_PATH: caption/caption_detic-template_and_kosmos_125k_iou0.2.json
    IMAGE_CORR_PATH: image_corr/scannet_caption_idx_detic-template_and_kosmos_125k_iou0.2.pkl
    SELECT: ratio
    NUM: 1
    RATIO: 0.2
    SAMPLE: 1
    GATHER_CAPTION: False

  ENTITY:
    ENABLED: False
"""


def yaml_to_dict(yaml_str: str) -> dict:
    """Converts a YAML string to a dictionary."""
    return yaml.safe_load(yaml_str)


@pytest.mark.parametrize("split", ["train"])
def test_scannet(split: str) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "~/datasets/regionplc".replace("~", str(Path.home()))
    omega_config_dict = yaml_to_dict(omega_config_str)
    cfg = OmegaConf.create(omega_config_dict)
    cfg.data_dir = data_dir
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
        caption_cfg=cfg.caption_cfg,
    )
    assert dataset is not None

    assert len(dataset) > 0

    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, dict)
        print(sample.keys())
        if i == 0:
            break
