import torch

import numpy as np
import glob
import copy

import hydra
from omegaconf import DictConfig
import os
from cuml.cluster import DBSCAN
import json
from src.models.networks.segment3d.demo_utils import get_model, load_mesh, prepare_data
from torch_scatter import scatter_mean
from src.models.networks.segment3d.demo import save_visualization

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CONFIG_DIR = os.path.join(BASE_DIR, "configs", "experiment")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="mask3d_scannet200.yaml")
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
    scores, _, masks_binary, _ = parse_predictions(
        cfg, outputs, point2segment, point2segment_full, raw_coordinates, inverse_map
    )

    if cfg.general.save_visualizations:
        # filter masks with a confidence threshold
        save_visualization(cfg, mesh, scores, masks_binary, scene_name, confidence_threshold=-1)


def get_mask_and_scores(cfg, mask_cls, mask_pred):
    num_queries = len(mask_cls)
    num_classes = cfg.model.num_classes - 1

    labels = (
        torch.arange(num_classes, device=mask_cls.device)
        .unsqueeze(0)
        .repeat(num_queries, 1)
        .flatten(0, 1)
    )

    scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(num_queries, sorted=True)

    labels_per_query = labels[topk_indices]
    topk_indices = torch.div(topk_indices, num_classes, rounding_mode="trunc")
    mask_pred = mask_pred[:, topk_indices]
    result_pred_mask = (mask_pred > 0).float()
    heatmap = mask_pred.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    score = scores_per_query * mask_scores_per_image
    classes = labels_per_query

    # sort by score
    sorted_score, sorted_indices = score.sort(dim=-1, descending=True)
    score = sorted_score
    result_pred_mask = result_pred_mask[:, sorted_indices]
    classes = classes[sorted_indices]
    heatmap = heatmap[:, sorted_indices]

    return score, result_pred_mask, classes, heatmap


def get_full_res_mask(mask, inverse_map, point2segment_full, is_heatmap: bool = False):
    mask = mask[inverse_map]  # full res
    if point2segment_full is not None and not is_heatmap:
        mask = scatter_mean(mask, point2segment_full.squeeze(0), dim=0)  # full res segments
        mask = (mask > 0.5).float()
        mask = mask[point2segment_full.squeeze(0)]  # full res points
    return mask


def parse_predictions(
    cfg, outputs, point2segment, point2segment_full, raw_coordinates, inverse_map
):
    assert len(outputs["pred_logits"]) == 1, "Batch size must be 1"
    assert len(outputs["pred_masks"]) == 1, "Batch size must be 1"

    logits = torch.functional.F.softmax(outputs["pred_logits"][0], dim=-1)[..., :-1].detach().cpu()
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

    scores, masks, classes, heatmap = get_mask_and_scores(
        cfg,
        logits,
        masks,
    )
    classes[classes == 0] = -1

    # remap labels
    label_offset = 2
    remapped_classes = classes + label_offset

    masks_binary = get_full_res_mask(masks, inverse_map, point2segment_full)
    masks_binary = masks_binary.permute(1, 0).bool()

    heatmap = get_full_res_mask(heatmap, inverse_map, point2segment_full, is_heatmap=True)
    heatmap = heatmap.permute(1, 0)

    return scores, remapped_classes, masks_binary, heatmap


if __name__ == "__main__":
    # example command:
    #     python -m src.models.networks.mask3d.demo general.test_scene=/path/to/test/scene

    main()
