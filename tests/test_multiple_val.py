import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def print_batch_info(batch, dataset_name):
    print(f"\n{dataset_name} batch keys:", batch.keys())


@hydra.main(config_path="../configs", config_name="data/ours_openvocab_scannet")
def test_data_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Instantiate the DataModule
    data_module = hydra.utils.instantiate(cfg)["data"]

    # Set up the DataModule
    data_module.setup("fit")

    # Test training dataloader
    train_loader = data_module.train_dataloader()
    train_batch = next(iter(train_loader))
    print_batch_info(train_batch, "Training")

    # Test all validation dataloaders
    val_loaders = data_module.val_dataloader()
    if isinstance(val_loaders, list):
        for i, val_loader in enumerate(val_loaders):
            val_batch = next(iter(val_loader))
            print_batch_info(val_batch, f"Validation dataset {i+1}")
    else:
        val_batch = next(iter(val_loaders))
        print_batch_info(val_batch, "Validation")


if __name__ == "__main__":
    import os
    import glob

    # Get all YAML files in the configs/data directory
    config_files = glob.glob(os.path.join("configs", "data", "*.yaml"))

    for config_file in config_files:
        config_name = os.path.basename(config_file).split(".")[0]
        print(f"\nTesting config: {config_name}")

        # Update the Hydra configuration
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="../configs")
        cfg = hydra.compose(config_name=f"data/{config_name}")

        # Run the test for each config
        test_data_config(cfg)
