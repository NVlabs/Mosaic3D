import hydra
import pytest
import torch
import yaml
from lightning import LightningDataModule
from omegaconf import OmegaConf

from warpconvnet.geometry.point_collection import PointCollection

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("fn_name", ["src.data.collate.point_collate_fn"])
def test_collate_fns(fn_name: str) -> None:
    with open("configs/data/regionplc_base15.yaml") as f:
        omega_config_dict = yaml.safe_load(f.read())
    cfg = OmegaConf.create(omega_config_dict)
    cfg.val_dataset = cfg.train_dataset
    cfg.collate_fn._target_ = fn_name
    cfg.num_workers = 0
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    for i, batch_dict in enumerate(loader):
        assert isinstance(batch_dict, dict)

        print(batch_dict.keys())
        pc = PointCollection(
            batch_dict["coord"], batch_dict["feats"], offsets=batch_dict["offsets"]
        ).to(DEVICE)
        if i == 0:
            break


if __name__ == "__main__":
    test_collate_fns("src.data.collate.point_collate_fn")
