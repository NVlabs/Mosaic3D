# %%
# --- Imports and Setup ---
from typing import List
import os
import sys
import numpy as np
import plotly.graph_objects as go
import colorsys
import random
import hydra
import rootutils
from omegaconf import DictConfig
import torch  # Added torch

# Setup project root
rootutils.setup_root("./", indicator=".project-root", pythonpath=True)

# Determine absolute path for Hydra config
CONFIG_DIR = os.path.abspath("./configs")

from src.data.datamodule import MultiDataModule

# %%
# --- Configuration ---
CONFIG_NAME = "data/mosaic3d++/sn-250k_caption3d.yaml"
NUM_OBJECTS_TO_VIS = 10  # Number of random objects to highlight
POINT_SIZE_BG = 2  # Point size for background points
POINT_SIZE_OBJ = 4  # Point size for highlighted object points


# %%
# --- Load Hydra Config and Instantiate DataModule ---
print("Initializing Hydra and loading configuration...")
cfg_data = None

# Set the random seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

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

# Setup a train loader
train_loader = datamodule.train_dataloader()

# %%
# Extract a single batch from the train loader
data_dict = next(iter(train_loader))

# Convert results to numpy for easier use if needed
point_indices_np = data_dict["caption_data"]["point_indices"].numpy()
caption_offsets_np = data_dict["caption_data"]["caption_offsets"].numpy()
num_points_per_caption_np = data_dict["caption_data"]["num_points_per_caption"].numpy()
captions: List[List[str]] = data_dict["caption_data"]["caption"]
offsets = data_dict["offset"]

print(f"Generated point_indices shape: {point_indices_np.shape}")
print(f"Generated caption_offsets shape: {caption_offsets_np.shape}")
print(f"Generated num_points_per_caption shape: {num_points_per_caption_np.shape}")

# %%
# --- Prepare Visualization Data ---
print("Preparing data for visualization...")

coords_np = data_dict["coord"].numpy()
colors_np = data_dict["color"].numpy()
num_objs = [len(c) for c in captions]
cum_num_objs = [0] + list(np.cumsum(num_objs))

batch_idx = 1
# Extract first batch
curr_coords_np = coords_np[offsets[batch_idx] : offsets[batch_idx + 1]]
curr_colors_np = colors_np[offsets[batch_idx] : offsets[batch_idx + 1]]
curr_caption_offsets_np = caption_offsets_np[
    cum_num_objs[batch_idx] : cum_num_objs[batch_idx + 1] + 1
]
curr_captions = captions[batch_idx]
curr_num_obj = len(curr_captions)

assert curr_num_obj == len(curr_caption_offsets_np) - 1

# %%
# Color is -1 to 1, convert to 0-1
curr_colors_np = (curr_colors_np + 1) / 2

# %%
# --- Visualize using Plotly ---
print("Creating Plotly visualization (using point_indices and caption_offsets)...")
fig = go.Figure()

# 1. Add the full point cloud
fig.add_trace(
    go.Scatter3d(
        x=curr_coords_np[:, 0],
        y=curr_coords_np[:, 1],
        z=curr_coords_np[:, 2],
        mode="markers",
        marker=dict(
            size=POINT_SIZE_BG, color=curr_colors_np, opacity=0.8
        ),  # Use numpy colors here
        name="Full Scene",
    )
)

# 2. Select random objects to highlight
num_to_visualize = min(NUM_OBJECTS_TO_VIS, len(curr_captions))
indices_to_visualize = sorted(random.sample(range(curr_num_obj), num_to_visualize))
print(
    f"Randomly selected {num_to_visualize} objects (indices: {indices_to_visualize}) out of {curr_num_obj} to visualize."
)

# 3. Add highlighted objects and their captions using point_indices and caption_offsets
for i in indices_to_visualize:
    if i >= curr_num_obj:
        print(
            f"Warning: Sampled index {i} is out of bounds for captions ({curr_num_obj}). Skipping."
        )
        continue

    caption = curr_captions[i]

    # Get the range of indices in the flattened 'point_indices' array for this object
    point_indices_obj = point_indices_np[
        curr_caption_offsets_np[i] : curr_caption_offsets_np[i + 1]
    ]
    num_pts_obj = len(point_indices_obj)

    if num_pts_obj <= 0:
        print(
            f"Warning: Skipping object index {i} ('{caption[:30]}...') as it has no points according to caption_offsets."
        )
        continue

    obj_coords = coords_np[point_indices_obj]

    # Generate a distinct color for the object mask
    hue = i / curr_num_obj
    saturation = 0.2
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
    title=f"Scene {batch_idx} - {num_to_visualize} Random Objects Highlighted (using simulated batched indices)",
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

# %%
# Optional: Save to HTML
VIS_DIR = "visualizations/dataloader_checks"  # Directory to potentially save visualizations
os.makedirs(VIS_DIR, exist_ok=True)
html_path = os.path.join(VIS_DIR, "mosaic_batched_caption3d_scene.html")
fig.write_html(html_path)
print(f"Plot saved to {html_path}")
