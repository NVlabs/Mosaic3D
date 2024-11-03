import sys

sys.path.append("/workspace")
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from omegaconf import OmegaConf

from src.data.scannet.dataset import ScanNetDataset

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


# Initialize dataset
cfg = OmegaConf.create(cfg_str)
dataset = ScanNetDataset(
    data_dir="/datasets/scannet_hf",
    split="train",
    transforms=cfg.val_transforms,
    caption_dir="/datasets/openvocab-3d-captions",
    caption_subset="caption.osprey.scannet-125k",
    segment_dir="/datasets/openvocab-3d-captions",
    segment_subset="segment.sam2.scannet-125k",
    object_sample_ratio=1.0,
    binary_mask_dir="/datasets/scannet_masks/oracle_scannet200_full",
    num_captions_per_mask=-1,
)

with gr.Blocks(
    title="ScanNet Scene Visualizer",
) as iface:
    gr.Markdown("# ScanNet Scene Visualizer")
    gr.Markdown("Visualize 3D scenes from ScanNet dataset with object highlighting")

    # Get initial visualization data
    initial_fig, _, _, _ = visualize_scene(0, 0)

    # Initialize dropdown choices with first scene's captions
    sample = dataset[0]
    initial_choices = [f"{i}: {cap}" for i, cap in enumerate(sample["caption_data"]["caption"])]
    initial_mask_dir = dataset.binary_mask_dir

    with gr.Row():
        scene_slider = gr.Slider(
            minimum=0, maximum=len(dataset) - 1, step=1, label="Scene Index", value=0
        )
        caption_dropdown = gr.Dropdown(
            label="Select Object",
            choices=initial_choices,  # Set initial choices here
            value=initial_choices[0],  # Set initial value here
            interactive=True,
        )
        mask_dir = gr.Dropdown(
            label="Mask Directory",
            choices=[
                "/datasets/scannet_masks/oracle_scannet200_full",
                "/datasets/scannet_masks/segment3d",
            ],
            value=initial_mask_dir,
            interactive=True,
        )
        load_all_captions = gr.Checkbox(
            label="Load All Captions per Mask",
            value=dataset.num_captions_per_mask < 0,
            interactive=True,
        )

    with gr.Row():
        # Left column: Point cloud
        with gr.Column(scale=2):
            point_cloud = gr.Plot(label="Point Cloud", value=initial_fig)

        with gr.Column(scale=-1):
            caption_text = gr.Textbox(
                label="Caption",
                value="\n".join(initial_choices[0].split(": ", 1)[1].split("@")),
            )

    # Event handlers
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

    def mask_dir_change(scene_idx, mask_dir):
        """Update visualization when mask directory is changed"""
        dataset.binary_mask_dir = Path(mask_dir)
        sample = dataset[scene_idx]
        captions = sample["caption_data"]["caption"]
        choices = [f"{i}: {cap}" for i, cap in enumerate(captions)]
        return [
            gr.Dropdown(choices=choices, value=choices[0]),
            update_visualization_only(scene_idx, 0),
            choices[0].split(": ", 1)[1],
        ]

    def load_all_captions_change(scene_idx, load_all):
        """Update visualization when load all captions checkbox changes"""
        dataset.num_captions_per_mask = -1 if load_all else 1
        sample = dataset[scene_idx]
        captions = sample["caption_data"]["caption"]
        choices = [f"{i}: {cap}" for i, cap in enumerate(captions)]
        return [
            gr.Dropdown(choices=choices, value=choices[0]),
            update_visualization_only(scene_idx, 0),
            choices[0].split(": ", 1)[1],
        ]

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

    load_all_captions.change(
        fn=load_all_captions_change,
        inputs=[scene_slider, load_all_captions],
        outputs=[caption_dropdown, point_cloud, caption_text],
    )

    mask_dir.change(
        fn=mask_dir_change,
        inputs=[scene_slider, mask_dir],
        outputs=[caption_dropdown, point_cloud, caption_text],
    )

    # Initialize dropdown with first scene's captions
    # sample = dataset[0]
    # initial_choices = [f"{i}: {cap}" for i, cap in enumerate(sample["caption_data"]["caption"])]
    # caption_dropdown.update(choices=initial_choices, value=initial_choices[0])

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
