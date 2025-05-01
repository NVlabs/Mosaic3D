# %%
# --- Imports and Setup ---
import os
import sys
import numpy as np
import plotly.graph_objects as go
import colorsys
import random
import hydra
import rootutils
from omegaconf import DictConfig

# Setup project root
rootutils.setup_root("./", indicator=".project-root", pythonpath=True)

# Determine absolute path for Hydra config
CONFIG_DIR = os.path.abspath("./configs")

from src.data.datamodule import MultiDataModule
from src.data.scannet.scannet_caption3d_dataset import ScanNetCaption3DDataset

# %%
# --- Configuration ---
CONFIG_NAME = "data/mosaic3d++/sn-250k_3d_caption.yaml"
SCENE_INDEX = 0  # Index of the scene to load from the dataset
NUM_OBJECTS_TO_VIS = 10  # Number of random objects to highlight
VIS_DIR = "visualizations/dataloader_checks"  # Directory to potentially save visualizations
POINT_SIZE_BG = 2  # Point size for background points
POINT_SIZE_OBJ = 4  # Point size for highlighted object points

os.makedirs(VIS_DIR, exist_ok=True)

# %%
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
if hasattr(datamodule.data_train, "datasets") and datamodule.data_train.datasets:
    train_dataset = datamodule.data_train.datasets[0]
    if not isinstance(train_dataset, ScanNetCaption3DDataset):
        print(
            f"Warning: Expected ScanNetCaption3DDataset, but found {type(train_dataset)}. Trying to proceed..."
        )
else:
    # Handle cases where data_train might be the dataset itself
    if isinstance(datamodule.data_train, ScanNetCaption3DDataset):
        train_dataset = datamodule.data_train
    else:
        raise TypeError(
            f"Could not extract ScanNetCaption3DDataset from datamodule. Found {type(datamodule.data_train)}"
        )

print(f"Using dataset: {type(train_dataset).__name__}")
print(f"Dataset size: {len(train_dataset)}")

# %%
# --- Load Data Sample ---
print(f"Loading data for scene index {SCENE_INDEX}...")
# Ensure index is valid
if SCENE_INDEX >= len(train_dataset):
    raise IndexError(
        f"SCENE_INDEX {SCENE_INDEX} is out of bounds for dataset size {len(train_dataset)}"
    )

# Access the sample using the dataset's __getitem__
# Note: This will apply the transforms defined in the config
data_dict = train_dataset[SCENE_INDEX]
print(
    f"Loaded scene: {data_dict.get('scene_name', 'N/A')} (Note: scene_name might not be collected by default transforms)"
)
print("Data keys:", data_dict.keys())

assert "caption_data" in data_dict
print(f"Found {len(data_dict['caption_data']['idx'])} captioned objects.")
caption_data_available = True

# %%
# --- Prepare Visualization Data ---
print("Preparing data for visualization...")
scene_name = data_dict.get("scene_name", f"SceneIndex_{SCENE_INDEX}")  # Fallback name

# Extract data (these might be tensors, convert to numpy)
coords = data_dict["coord"]
colors = data_dict["color"]

# Color is -1 to 1, convert to 0-1
colors = (colors + 1) / 2

object_indices_list = []
captions = []
# Indices might be tensors, convert to numpy list/arrays
object_indices_list = data_dict["caption_data"]["idx"]
captions = data_dict["caption_data"]["caption"]  # Should already be a list of strings
print(f"Number of objects to potentially visualize: {len(object_indices_list)}")

# %%
# --- Visualize using Plotly ---
print("Creating Plotly visualization...")
fig = go.Figure()

# 1. Add the full point cloud
fig.add_trace(
    go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(size=POINT_SIZE_BG, color=colors, opacity=0.8),
        name="Full Scene",
    )
)

# 2. Select random objects to highlight
num_objects_total = len(object_indices_list)
num_to_visualize = min(NUM_OBJECTS_TO_VIS, num_objects_total)
indices_to_visualize = []
if num_objects_total > 0:
    try:
        indices_to_visualize = random.sample(range(num_objects_total), num_to_visualize).sort()
        print(
            f"Randomly selected {num_to_visualize} objects (indices: {indices_to_visualize}) out of {num_objects_total} to visualize."
        )
    except ValueError as e:
        print(f"Could not sample objects: {e}")
else:
    print("No objects with captions to visualize.")

# 3. Add highlighted objects and their captions
for i in indices_to_visualize:
    if i >= len(object_indices_list) or i >= len(captions):
        print(f"Warning: Index {i} out of bounds for object_indices or captions. Skipping.")
        continue

    obj_indices = object_indices_list[i]
    caption = captions[i]

    # The indices from caption_data usually refer to the *original* point cloud indices,
    # before any subsampling/cropping transforms. If transforms like SphereCrop are used,
    # these indices might be invalid for the current `coords` array.
    # We need a way to map these original indices to the current `coords` array.
    # This often requires the dataset item to also return 'origin_idx' or similar mapping.
    # Check if 'origin_idx' exists in data_dict.

    if "origin_idx" not in data_dict:
        print(
            f"Warning: 'origin_idx' not found in data_dict. Cannot reliably map caption indices to transformed coords. Skipping object {i}."
        )
        continue

    origin_idx = (
        data_dict["origin_idx"].numpy()
        if hasattr(data_dict["origin_idx"], "numpy")
        else data_dict["origin_idx"]
    )

    # Find which of the *current* points correspond to the original object indices
    # This requires finding where origin_idx matches the values in obj_indices
    current_mask = np.isin(origin_idx, obj_indices)
    current_obj_indices = np.where(current_mask)[0]

    if current_obj_indices.size == 0:
        print(
            f"Warning: Skipping object index {i} ('{caption[:30]}...') as it has no points in the *current* point cloud (possibly due to transforms)."
        )
        continue

    obj_coords = coords[current_obj_indices]

    # Generate a distinct color for the object mask
    hue = i / num_objects_total if num_objects_total > 0 else 0
    saturation = 0.9
    value = 0.9
    rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
    obj_color_rgb = (np.array(rgb_float) * 255).astype(np.uint8)
    obj_color_str = f"rgb({obj_color_rgb[0]}, {obj_color_rgb[1]}, {obj_color_rgb[2]})"

    # Add object mask points
    fig.add_trace(
        go.Scatter3d(
            x=obj_coords[:, 0],
            y=obj_coords[:, 1],
            z=obj_coords[:, 2],
            mode="markers",
            marker=dict(
                size=POINT_SIZE_OBJ,
                color=obj_color_str,
                opacity=1.0,
            ),
            name=f"Obj {i}",
        )
    )

    # Add caption as text near the center
    if obj_coords.shape[0] > 0:
        center_point = obj_coords.mean(axis=0)
        display_caption = (
            "Object " + str(i) + ": " + caption[:20] + "..."
            if len(caption) > 20
            else "Object " + str(i) + ": " + caption
        )
        fig.add_trace(
            go.Scatter3d(
                x=[center_point[0]],
                y=[center_point[1]],
                z=[center_point[2]],
                mode="text",
                text=[display_caption],
                textposition="middle center",
                textfont=dict(size=10, color=obj_color_str),
                name=f"Caption {i}",
                showlegend=False,  # Hide caption trace from legend
            )
        )
    print(f"  Object {i}: {caption}")

# --- Finalize and Show Plot ---
print("Finalizing plot layout...")
fig.update_layout(
    title=f"Scene: {scene_name} (Index: {SCENE_INDEX}) - {num_to_visualize} Random Objects Highlighted",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data",  # Use 'data' for ScanNet aspect ratio
        # aspectratio=dict(x=1, y=1, z=1), # Usually not needed with aspectmode='data'
        camera_eye=dict(x=1.5, y=1.5, z=1.5),  # Adjust camera view
    ),
    margin=dict(l=10, r=10, b=10, t=50),  # Adjust margins
    legend=dict(itemsizing="constant"),  # Prevent legend items from resizing
)

print("Displaying plot.")
fig.show()

# Optional: Save to HTML
# html_path = os.path.join(VIS_DIR, f"scannet_caption3d_scene_{SCENE_INDEX}.html")
# fig.write_html(html_path)
# print(f"Plot saved to {html_path}")

# %%
