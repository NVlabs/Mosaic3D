# pytest -v -s tests/data/scannet_caption3d.py

import hydra
import pytest
import rootutils
from omegaconf import DictConfig
import os

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datamodule import MultiDataModule

# Determine the absolute path to the configs directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(TEST_DIR, "../../configs"))


@pytest.mark.slow  # Mark as slow if instantiation takes time
def test_scannet_caption3d_config_instantiation():
    """Tests if the ScanNetCaption3D datamodule config can be loaded and instantiated."""
    # Initialize Hydra temporarily for this test using the absolute path
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        # Compose the configuration
        cfg_data = hydra.compose(
            config_name="data/mosaic3d++/sn-250k_3d_caption.yaml",
            # Overrides are applied relative to the composed config
            overrides=[
                # We might need to adjust paths depending on where pytest is run from
                # If datasets are expected at absolute paths defined in the yaml,
                # we might not need to override paths.project_root here.
                # For now, let's remove it to rely on the paths in the yaml.
                # "paths.project_root=.",
            ],
        )

    assert cfg_data is not None
    assert isinstance(cfg_data, DictConfig)

    # Instantiate the datamodule part of the config
    # The config loaded is nested under 'data', but the yaml itself is the datamodule config.
    datamodule = hydra.utils.instantiate(cfg_data)
    datamodule = datamodule["data"]

    assert datamodule is not None
    datamodule.setup("fit")

    # Get the dataset
    train_dataset = datamodule.data_train.datasets[0]
    for i in range(10):
        datum = train_dataset[i]
        print(datum.keys())
