import sys

sys.path.append("/workspace")
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from omegaconf import OmegaConf

from src.data.arkitscenes.dataset import ARKitScenesDataset
from src.data.matterport3d.dataset import Matterport3DDataset
from src.data.scannet.dataset import ScanNetDataset
from src.data.structured3d.dataset import Structured3DDataset

cfg_str = """
val_transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
  - type: CenterShift
    apply_z: true
  - type: NormalizeColor
  - type: ToTensor
  - type: Collect
    keys:
      [
        "origin_coord",
        "coord",
        "color",
        "segment",
        "instance",
        "binary",
        "caption_data",
        "origin_idx",
      ]
    offset_keys_dict:
      offset: coord
      pc_count: origin_coord
    feat_keys: ["color"]
"""
# Initialize dataset
cfg = OmegaConf.create(cfg_str)
kwargs = {
    "split": "train",
    "caption_dir": "/datasets/openvocab-3d-captions",
    "segment_dir": "/datasets/openvocab-3d-captions",
    "object_sample_ratio": 1.0,
    "transforms": cfg.val_transforms,
}
scannet_dataset = ScanNetDataset(
    **kwargs,
    data_dir="/datasets/scannet_hf",
    caption_subset="caption.osprey.scannet-125k",
    segment_subset="segment.sam2.scannet-125k",
)
matterport3_dataset = Matterport3DDataset(
    **kwargs,
    data_dir="/datasets/matterport3d/matterport_3d",
    caption_subset="caption.osprey.matterport3d",
    segment_subset="segment.sam2.matterport3d",
)
arkitscenes_dataset = ARKitScenesDataset(
    **kwargs,
    data_dir="/datasets/arkitscenes/3dod",
    caption_subset="caption.osprey.arkitscenes",
    segment_subset="segment.sam2.arkitscenes",
)
structured3d_dataset = Structured3DDataset(
    **kwargs,
    data_dir="/datasets/structured3d",
    caption_subset=["caption.osprey.structured3d", "caption.osprey.structured3d-pano"],
    segment_subset=["segment.sam2.structured3d", "segment.sam2.structured3d-pano"],
)

# Create dataset mapping
datasets = {
    "ScanNet": scannet_dataset,
    "Matterport3D": matterport3_dataset,
    "ARKitScenes": arkitscenes_dataset,
    "Structured3D": structured3d_dataset,
}

# default dataset
dataset = scannet_dataset


def create_point_cloud_plot(xyz, color):
    rgb_colors = [f"rgb({r*255},{g*255},{b*255})" for r, g, b in color]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=2, color=rgb_colors, line_width=0),
            )
        ]
    )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig


def visualize_scene(scene_index, object_index):
    # Get sample
    sample = dataset[scene_index]

    # Process data
    origin_coord = sample["origin_coord"]
    color = sample["feat"]
    color = color.numpy()
    color = color * 0.5 + 0.5
    caption_data = sample["caption_data"]
    indices = caption_data["idx"]
    captions = caption_data["caption"]

    # Get specific object data
    idx = indices[object_index]
    caption = captions[object_index]

    # Process coordinates
    xyz = origin_coord.numpy()
    xyz = xyz - xyz.mean(axis=0)
    xyz[:, [1, 0, 2]] = xyz[:, [0, 1, 2]]  # Fixed coordinate swapping

    # Highlight selected object
    color_vis = color.copy()
    color_vis[idx] = [1, 0, 0]

    # Create plot
    fig = create_point_cloud_plot(xyz, color_vis)

    # Format all captions with indices
    all_captions = "\n".join([f"{i}: {cap}" for i, cap in enumerate(captions)])

    return fig, caption, f"Total objects in scene: {len(captions)}", all_captions


def update_scene_info(scene_index):
    """Update all info when scene changes"""
    sample = dataset[scene_index]
    captions = sample["caption_data"]["caption"]

    # Get visualization for first object (index 0)
    origin_coord = sample["origin_coord"]
    color = sample["feat"].numpy() * 0.5 + 0.5
    idx = sample["caption_data"]["idx"][0]

    xyz = origin_coord.numpy()
    xyz = xyz - xyz.mean(axis=0)
    xyz[:, [1, 0, 2]] = xyz[:, [0, 1, 2]]

    color_vis = color.copy()
    color_vis[idx] = [1, 0, 0]

    fig = create_point_cloud_plot(xyz, color_vis)

    return [
        gr.Slider(minimum=0, maximum=len(captions) - 1, value=0, step=1),  # object slider
        captions[0],  # selected caption
        f"Total objects in scene: {len(captions)}",  # scene info
        fig,  # point cloud
        captions,  # list of captions for buttons
    ]


def update_visualization_only(scene_index, object_index):
    """Update only the visualization"""
    sample = dataset[scene_index]
    origin_coord = sample["origin_coord"]
    color = sample["feat"].numpy() * 0.5 + 0.5
    caption_data = sample["caption_data"]
    indices = caption_data["idx"]

    idx = indices[object_index]

    xyz = origin_coord.numpy()
    xyz = xyz - xyz.mean(axis=0)
    xyz[:, [1, 0, 2]] = xyz[:, [0, 1, 2]]

    color_vis = color.copy()
    color_vis[idx] = [1, 0, 0]

    fig = create_point_cloud_plot(xyz, color_vis)

    return fig  # Return just the figure, not a list


def get_caption_length(scene_index):
    sample = dataset[scene_index]
    caption_data = sample["caption_data"]
    return len(caption_data["caption"])


with gr.Blocks(
    title="3D Scene Visualizer",
) as iface:
    gr.Markdown("# 3D Scene Visualizer")
    gr.Markdown("Visualize 3D scenes from multiple datasets with object highlighting")

    # Get initial visualization data
    initial_fig, _, _, _ = visualize_scene(0, 0)

    # Initialize dropdown choices with first scene's captions
    sample = dataset[0]
    initial_choices = [f"{i}: {cap}" for i, cap in enumerate(sample["caption_data"]["caption"])]

    with gr.Row():
        dataset_dropdown = gr.Dropdown(
            label="Select Dataset",
            choices=list(datasets.keys()),
            value="ScanNet",
            interactive=True,
        )
        scene_slider = gr.Slider(
            minimum=0, maximum=len(dataset) - 1, step=1, label="Scene Index", value=0
        )
        caption_dropdown = gr.Dropdown(
            label="Select Object",
            choices=initial_choices,  # Set initial choices here
            value=initial_choices[0],  # Set initial value here
            interactive=True,
        )

    with gr.Row():
        # Left column: Point cloud
        with gr.Column(scale=2):
            point_cloud = gr.Plot(label="Point Cloud", value=initial_fig)

        with gr.Column(scale=1):
            caption_text = gr.Textbox(
                label="Caption",
                value="\n".join(initial_choices[0].split(": ", 1)[1].split("@")),
            )

    # Event handlers
    def dataset_change(dataset_name):
        """Update when dataset changes"""
        global dataset
        dataset = datasets[dataset_name]
        sample = dataset[0]
        captions = sample["caption_data"]["caption"]
        choices = [f"{i}: {cap}" for i, cap in enumerate(captions)]

        return [
            gr.Slider(minimum=0, maximum=len(dataset) - 1, value=0),
            gr.Dropdown(choices=choices, value=choices[0]),
            update_visualization_only(0, 0),
            choices[0].split(": ", 1)[1],
        ]

    def scene_change(scene_idx):
        """Update dropdown and visualization when scene changes"""
        sample = dataset[scene_idx]
        captions = sample["caption_data"]["caption"]
        # Create dropdown choices with index and caption
        choices = [f"{i}: {cap}" for i, cap in enumerate(captions)]

        fig = update_visualization_only(scene_idx, 0)
        return [gr.Dropdown(choices=choices, value=choices[0]), fig, choices[0].split(": ", 1)[1]]

    def caption_change(scene_idx, caption_choice):
        """Update visualization when caption is selected"""
        if caption_choice is None:
            return None
        # Extract index from the caption choice string
        obj_idx = int(caption_choice.split(":")[0])
        caption = caption_choice.split(": ", 1)[1]
        return [update_visualization_only(scene_idx, obj_idx), caption]

    dataset_dropdown.change(
        fn=dataset_change,
        inputs=[dataset_dropdown],
        outputs=[scene_slider, caption_dropdown, point_cloud, caption_text],
    )

    scene_slider.change(
        fn=scene_change,
        inputs=[scene_slider],
        outputs=[caption_dropdown, point_cloud, caption_text],
    )

    caption_dropdown.change(
        fn=caption_change,
        inputs=[scene_slider, caption_dropdown],
        outputs=[point_cloud, caption_text],
    )


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
