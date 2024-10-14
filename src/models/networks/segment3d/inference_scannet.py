import os
import glob
import json

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm

from src.models.networks.segment3d.demo_utils import get_model, load_mesh, prepare_data
from src.models.networks.segment3d.demo import CONFIG_DIR, parse_predictions, save_visualization


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="segment3d_scannet.yaml")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = get_model(cfg)
    model.eval()
    model.to(device)

    # inference on all scannet scenes (trainval)
    scene_ids = sorted(os.listdir(cfg.general.scannet_dir))
    for scene_id in (pbar := tqdm(scene_ids)):
        pbar.set_description(f"Processing {scene_id}")

        scene_dir = os.path.join(cfg.general.scannet_dir, scene_id)
        ply_files = glob.glob(f"{scene_dir}/*_vh_clean_2.ply")
        assert len(ply_files) == 1, "Expected one ply file in the scene directory"
        mesh = load_mesh(ply_files[0])

        point2segment = None
        if cfg.general.train_on_segments:
            json_files = glob.glob(f"{scene_dir}/*_vh_clean_2.0.010000.segs.json")
            assert len(json_files) == 1, "Expected one json file in the scene directory"
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

        # save masks with npz
        np.savez(
            os.path.join(cfg.general.output_dir, f"{scene_id}.npz"),
            scores=scores.numpy(),
            masks_binary=masks_binary.numpy(),
        )

        # save visualization
        if cfg.general.save_visualizations:
            save_visualization(cfg, mesh, scores, masks_binary, scene_id, confidence_threshold=-1)


if __name__ == "__main__":
    # example command:
    #     python -m src.models.networks.segment3d.inference_scannet

    main()
