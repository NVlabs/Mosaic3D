import torch

import numpy as np
import glob

import hydra
from omegaconf import DictConfig
import os
from cuml.cluster import DBSCAN
import json
from src.models.networks.segment3d.demo_utils import get_model, load_mesh, prepare_data
from torch_scatter import scatter_mean
import pyviz3d.visualizer as viz

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CONFIG_DIR = os.path.join(BASE_DIR, "configs", "experiment")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="segment3d_scannetpp.yaml")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    model = get_model(cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load input data
    scene_name = os.path.basename(cfg.general.test_scene)
    ply_files = glob.glob(f"{cfg.general.test_scene}/*.ply")
    assert len(ply_files) == 1, "Expected one ply file in the test scene directory"
    mesh = load_mesh(ply_files[0])

    point2segment = None
    if cfg.general.train_on_segments:
        json_files = glob.glob(f"{cfg.general.test_scene}/*.segs.json")
        assert len(json_files) == 1, "Expected one json file in the test scene directory"
        segment_filepath = json_files[0]
        with open(segment_filepath) as f:
            segments = json.load(f)
            point2segment = np.array(segments["segIndices"])

    # prepare data
    data, point2segment, point2segment_full, raw_coordinates, inverse_map = prepare_data(
        cfg, mesh, point2segment, device
    )

    # run model
    with torch.no_grad():
        outputs = model(data, point2segment=point2segment, raw_coordinates=raw_coordinates)

    # parse predictions
    scores, masks_binary = parse_predictions(
        cfg, outputs, point2segment, point2segment_full, raw_coordinates, inverse_map
    )

    if cfg.general.save_visualizations:
        # filter masks with a confidence threshold
        save_visualization(cfg, mesh, scores, masks_binary, scene_name, confidence_threshold=-1)


def get_mask_and_scores(cfg, mask_cls, mask_pred):
    result_pred_mask = (mask_pred > 0).float()

    mask_pred = mask_pred[:, result_pred_mask.sum(0) > 0]
    mask_cls = mask_cls[result_pred_mask.sum(0) > 0]
    result_pred_mask = result_pred_mask[:, result_pred_mask.sum(0) > 0]
    heatmap = mask_pred.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    score = mask_cls * mask_scores_per_image

    topk_count = (
        min(cfg.general.topk_per_image, len(score))
        if cfg.general.topk_per_image != -1
        else len(score)
    )
    score, topk_indices = score.topk(topk_count, sorted=True)

    result_pred_mask = result_pred_mask[:, topk_indices]
    return score, result_pred_mask


def get_full_res_mask(mask, inverse_map, point2segment_full):
    mask = mask[inverse_map]  # full res
    if point2segment_full is not None:
        mask = scatter_mean(mask, point2segment_full.squeeze(0), dim=0)  # full res segments
        mask = (mask > 0.5).float()
        mask = mask[point2segment_full.squeeze(0)]  # full res points
    return mask


def parse_predictions(
    cfg, outputs, point2segment, point2segment_full, raw_coordinates, inverse_map
):
    assert len(outputs["pred_logits"]) == 1, "Batch size must be 1"
    assert len(outputs["pred_masks"]) == 1, "Batch size must be 1"

    logits = torch.functional.F.softmax(outputs["pred_logits"][0], dim=-1)[:, 0].detach().cpu()
    masks = outputs["pred_masks"][0].detach().cpu()

    if cfg.model.train_on_segments:
        masks = outputs["pred_masks"][0].detach().cpu()[point2segment.cpu()].squeeze(0)
    else:
        masks = outputs["pred_masks"][0].detach().cpu()

    if cfg.general.use_dbscan:
        new_logits = []
        new_masks = []
        for curr_query in range(masks.shape[1]):
            curr_masks = masks[:, curr_query] > 0
            if raw_coordinates[curr_masks].shape[0] > 0:
                clusters = (
                    DBSCAN(
                        eps=cfg.general.dbscan_eps,
                        min_samples=cfg.general.dbscan_min_points,
                        verbose=2,
                    )
                    .fit(raw_coordinates[curr_masks].cuda())
                    .labels_
                )
                clusters = clusters.get()
                new_mask = np.zeros(curr_masks.shape, dtype=int)
                new_mask[curr_masks] = clusters + 1

                for cluster_id in np.unique(clusters):
                    original_pred_masks = masks[:, curr_query].numpy()
                    if cluster_id != -1:
                        if (new_mask == cluster_id + 1).sum() > cfg.data.remove_small_group:
                            new_logits.append(logits[curr_query])
                            new_masks.append(
                                torch.from_numpy(
                                    original_pred_masks * (new_mask == cluster_id + 1)
                                )
                            )
        logits = torch.stack(new_logits).cpu()
        masks = torch.stack(new_masks).T

    scores, masks = get_mask_and_scores(
        cfg,
        logits,
        masks,
    )

    masks_binary = get_full_res_mask(masks, inverse_map, point2segment_full)
    masks_binary = masks_binary.permute(1, 0).bool()
    return scores, masks_binary


def save_visualization(
    cfg, mesh, scores, masks_binary, scene_name, confidence_threshold, point_size=20
):
    v = viz.Visualizer()
    point_positions = np.asarray(mesh.vertices)
    point_colors = np.asarray(mesh.vertex_colors)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    point_normals = np.asarray(mesh.vertex_normals)

    v.add_points("rgb", point_positions, point_colors * 255, point_normals, point_size=point_size)
    pred_coords = []
    pred_inst_color = []
    pred_normals = []
    for i in reversed(range(len(masks_binary))):
        mask_i = masks_binary[i]
        score_i = scores[i]
        if score_i > confidence_threshold:
            num_i = point_positions[mask_i].shape[0]
            color_i = np.tile(np.random.rand(3) * 255, [num_i, 1])
            v.add_points(
                f"{i}_{score_i:.2f}",
                point_positions[mask_i],
                color_i,
                point_normals[mask_i],
                point_size=point_size,
                visible=False,
            )
            pred_coords.append(point_positions[mask_i])
            pred_inst_color.append(color_i)
            pred_normals.append(point_normals[mask_i])
    pred_coords = np.concatenate(pred_coords)
    pred_inst_color = np.concatenate(pred_inst_color)
    pred_normals = np.concatenate(pred_normals)
    v.add_points(
        "Instances",
        pred_coords,
        pred_inst_color,
        pred_normals,
        point_size=point_size,
        alpha=0.8,
        visible=False,
    )
    output_path = f"{cfg.general.visualization_dir}/{scene_name}"
    v.save(output_path)


if __name__ == "__main__":
    # example command:
    #     python -m src.models.networks.segment3d.demo general.test_scene=/path/to/test/scene general.save_visualizations=true

    main()
