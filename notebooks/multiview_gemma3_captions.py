# %%
import os
import sys
import numpy as np
import plotly.graph_objects as go
import colorsys
import random

# Adjust path to import from src
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data.scannet.scannet_caption3d_dataset import ScanNetCaption3DDataset

# %%
# --- Configuration ---
DATA_DIR = "/datasets/scannet_hf"  # Replace with your ScanNet data path if different
CAPTION_DIR = "/datasets/mosaic3d++"  # Replace with your caption data path if different
CAPTION_SUBSET = "caption-mc-mv.gemma3.scannet-250k"
SEGMENT_DIR = "/datasets/mosaic3d++"
SEGMENT_SUBSET = "mask_clustering.cropformer.scannet-250k+combined"
SPLIT = "train"
SCENE_INDEX = 0  # Index of the scene to visualize (e.g., 0 for scene0000_00)
OBJECT_NUM_MAX = 100  # Max number of objects to load per scene (matches dataset example)
VIS_DIR = "visualizations/caption_checks"  # Directory to save visualizations
POINT_SIZE = 15  # Point size for visualization

os.makedirs(VIS_DIR, exist_ok=True)

# %%
# --- Instantiate Dataset ---
print("Initializing dataset...")
dataset = ScanNetCaption3DDataset(
    data_dir=DATA_DIR,
    split=SPLIT,
    transforms=None,  # No transforms for basic visualization
    object_num_max=OBJECT_NUM_MAX,
    caption_dir=CAPTION_DIR,
    caption_subset=CAPTION_SUBSET,
    segment_dir=SEGMENT_DIR,
    segment_subset=SEGMENT_SUBSET,
)
print(f"Dataset size: {len(dataset)}")

# %%
# --- Load Data Sample ---
print(f"Loading data for scene index {SCENE_INDEX}...")
# Ensure index is valid
if SCENE_INDEX >= len(dataset):
    raise IndexError(f"SCENE_INDEX {SCENE_INDEX} is out of bounds for dataset size {len(dataset)}")

data_dict = dataset[SCENE_INDEX]
print(f"Loaded scene: {data_dict['scene_name']}")
print("Data keys:", data_dict.keys())

# Check if caption data exists
if "caption_data" not in data_dict or not data_dict["caption_data"]["idx"]:
    print(f"No caption data found for scene {data_dict['scene_name']}. Exiting.")
    # Or handle appropriately, e.g., load next scene
else:
    print(f"Found {len(data_dict['caption_data']['idx'])} captioned objects.")

# %%
# --- Prepare Visualization Data ---
scene_name = data_dict["scene_name"]
coords = data_dict["coord"]  # (N, 3)
colors = data_dict["color"]  # (N, 3) - Assuming colors are 0-1 floats

# Convert colors to 0-255 uint8
if colors.max() <= 1.0:
    colors = (colors * 255).astype(np.uint8)

# Check if caption data exists and is not empty before accessing
if "caption_data" in data_dict and data_dict["caption_data"]["idx"]:
    object_indices_list = data_dict["caption_data"]["idx"]  # List of tensors
    captions = data_dict["caption_data"]["caption"]  # List of strings
    print(f"Number of objects to visualize: {len(object_indices_list)}")
else:
    print(f"No caption data found for scene {scene_name}. Cannot create object visualization.")
    object_indices_list = []
    captions = []

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
        marker=dict(size=2, color=colors, opacity=0.8),  # Use the processed colors array
        name="Full Scene",
    )
)

# 2. Select up to 5 random objects to highlight
num_objects_total = len(object_indices_list)
num_to_visualize = min(5, num_objects_total)
if num_objects_total > 0:
    indices_to_visualize = random.sample(range(num_objects_total), num_to_visualize)
    # Append 13 to the end of the list
    indices_to_visualize.append(13)
    print(f"Visualizing {num_to_visualize} random objects out of {num_objects_total}.")
else:
    indices_to_visualize = []
    print("No objects with captions to visualize.")


# 3. Add highlighted objects and their captions
for i in indices_to_visualize:
    obj_indices_tensor = object_indices_list[i]
    caption = captions[i]
    obj_indices = obj_indices_tensor.numpy().astype(int)

    if obj_indices.size == 0:
        print(f"Warning: Skipping object index {i} ('{caption}') as it has no points.")
        continue

    obj_coords = coords[obj_indices]

    # Generate a distinct color for the object mask
    hue = i / num_objects_total  # Use total number for consistent coloring if run multiple times
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
                size=4,  # Slightly larger markers for highlighted objects
                color=obj_color_str,
                opacity=1.0,
            ),
            name=f"Obj {i}",
        )
    )

    # Add caption as text near the center
    if obj_coords.size > 0:
        center_point = obj_coords.mean(axis=0)
        display_caption = caption[:80] + "..." if len(caption) > 80 else caption
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
            )
        )
    print(f"Caption {i}: {display_caption}")

# Update layout for better viewing
fig.update_layout(
    title=f"Scene: {scene_name} - Point Cloud with {num_to_visualize} Random Captioned Objects",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data",
        aspectratio=dict(x=1, y=1, z=1),
        camera_eye=dict(x=1.25, y=1.25, z=1.25),
    ),
    margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins
)

fig.show()

# %%
