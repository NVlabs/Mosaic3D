import pytest
import yaml
from omegaconf import OmegaConf

from src.data.scannet.dataset import ScanNetDataset


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
    with open("configs/tests/scannet.yaml") as f:
        omega_config_dict = yaml_to_dict(f.read())
    cfg = OmegaConf.create(omega_config_dict)
    cfg.split = split

    dataset = ScanNetDataset(
        data_dir=cfg.data_dir,
        split=cfg.split,
        transforms=cfg.transforms,
        caption_dir=cfg.caption_dir,
        caption_subset=cfg.caption_subset,
        object_sample_ratio=cfg.object_sample_ratio,
        base_class_idx=cfg.base_class_idx,
        novel_class_idx=cfg.novel_class_idx,
        ignore_class_idx=cfg.ignore_class_idx,
        ignore_label=cfg.ignore_label,
        repeat=cfg.repeat,
    )
    assert dataset is not None

    assert len(dataset) > 0

    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, dict)
        print(sample.keys())
        if i == 0:
            break


if __name__ == "__main__":
    test_scannet("train")
