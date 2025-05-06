# %%
import torch
from src.models.components.text_encoder import RecapCLIPTextEncoder, Siglip2TextEncoder
from src.utils.meter import GPUMemoryMeter

# %%
# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


siglip2_encoder = Siglip2TextEncoder()
recap_encoder = RecapCLIPTextEncoder()

meters = {
    "SigLIP": GPUMemoryMeter(name="SigLIP"),
    "Recap CLIP": GPUMemoryMeter(name="Recap CLIP"),
}

# %%
# Benchmark parameters
import hydra
import os
from src.data.datamodule import MultiDataModule
from src.data.scannet.scannet_caption3d_dataset import ScanNetCaption3DDataset

CONFIG_DIR = os.path.abspath("./configs")
CONFIG_NAME = "data/mosaic3d++/sn-250k_caption3d.yaml"
SCENE_INDEX = 0  # Index of the scene to load from the dataset

# --- Load Hydra Config and Instantiate DataModule ---
print("Initializing Hydra and loading configuration...")
cfg_data = None
with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
    cfg_data = hydra.compose(
        config_name=CONFIG_NAME,
        # Add any overrides if necessary, e.g., if running on a different machine
        # overrides=["data_dir=/new/path/to/scannet", "..."],
    )

assert cfg_data is not None
print(f"Loaded config: {CONFIG_NAME}")

print("Instantiating DataModule...")
# The loaded config (cfg_data) directly represents the datamodule configuration
# as defined in sn-250k_3d_caption.yaml
datamodule: MultiDataModule = hydra.utils.instantiate(cfg_data)["data"]
datamodule.setup("fit")  # Setup for train/val splits
print("DataModule instantiated successfully.")

# %%
# --- Get the Training Dataset ---
print("Accessing the training dataset...")
# The train_dataset is likely a ConcatDataset, access the underlying dataset
# Adjust index [0] if your ConcatDataset structure is different

train_dataset = datamodule.data_train.datasets[0]
data_dict = train_dataset[SCENE_INDEX]

assert "caption_data" in data_dict
print(f"Found {len(data_dict['caption_data']['idx'])} captioned objects.")
caption_data_available = True
texts = data_dict["caption_data"]["caption"]

# %%
# Benchmark loop
results = {}
BATCH_SIZE = 64
WARMUP_BATCHES = 5

# SigLIP
# warmup
for _ in range(WARMUP_BATCHES):
    text = texts[:BATCH_SIZE]
    _ = siglip2_encoder(text)
    _ = recap_encoder(text)

# measurement
for i in range(0, len(texts), BATCH_SIZE):
    text = texts[i : i + BATCH_SIZE]
    with meters["SigLIP"]:
        _ = siglip2_encoder(text)  # 1152

    with meters["Recap CLIP"]:
        _ = recap_encoder(text)  # 768

# %%
# Print results
for meter_name, meter in meters.items():
    print(
        f"{meter_name} - Average Time: {meter.average_time:.4f} seconds - Max GPU Memory Usage: {meter.max_memory_usage:.2f} MB"
    )

# %%
