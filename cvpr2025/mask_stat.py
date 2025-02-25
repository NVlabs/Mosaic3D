import os
from copy import deepcopy
from typing import List, Tuple

import hydra
import numpy as np
from lexical_diversity import lex_div as ld
from omegaconf import OmegaConf
from rich.progress import track

from src.data.metadata.scannet import CLASS_LABELS_200

scannet_dataset_cfg = """
_target_: src.data.scannet.dataset.ScanNetDataset
data_dir: /datasets/scannet_hf
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
split: train
caption_dir: /datasets/openvocab-3d-captions
caption_subset:
  - caption.osprey.scannet-125k
  - caption.seem-panoptic-osprey.scannet-125k
segment_dir: /datasets/openvocab-3d-captions
segment_subset:
  - segment.sam2.scannet-125k
  - segment.seem-panoptic.scannet-125k
"""

arkitscenes_dataset_cfg = """
_target_: src.data.arkitscenes.dataset.ARKitScenesDataset
data_dir: /datasets/arkitscenes/3dod
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
split: train
caption_dir: /datasets/openvocab-3d-captions
caption_subset:
- caption.osprey.arkitscenes-rectified
- caption.seem-panoptic-osprey.arkitscenes-rectified
segment_dir: /datasets/openvocab-3d-captions
segment_subset:
- segment.sam2.arkitscenes-rectified
- segment.seem-panoptic.arkitscenes-rectified
"""

matterport3d_dataset_cfg = """
_target_: src.data.matterport3d.dataset.Matterport3DDataset
data_dir: /datasets/matterport3d/matterport_3d
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
split: train
caption_dir: /datasets/openvocab-3d-captions
caption_subset:
- caption.osprey.matterport3d
- caption.seem-panoptic-osprey.matterport3d
segment_dir: /datasets/openvocab-3d-captions
segment_subset:
- segment.sam2.matterport3d
- segment.seem-panoptic.matterport3d
"""

structured3d_dataset_cfg = """
_target_: src.data.structured3d.dataset.Structured3DDataset
data_dir: /datasets/structured3d
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
split: train
caption_dir: /datasets/openvocab-3d-captions
caption_subset:
- caption.osprey.structured3d
- caption.osprey.structured3d-pano
- caption.seem-panoptic-osprey.structured3d
- caption.seem-panoptic-osprey.structured3d-pano
segment_dir: /datasets/openvocab-3d-captions
segment_subset:
- segment.sam2.structured3d
- segment.sam2.structured3d-pano
- segment.seem-panoptic.structured3d
- segment.seem-panoptic.structured3d-pano
"""

scannetpp_dataset_cfg = """
_target_: src.data.scannetpp.dataset.ScanNetPPDataset
data_dir: /datasets/scannetpp/preprocessed_pcd
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
split: train
caption_dir: /datasets/openvocab-3d-captions
caption_subset:
- caption.osprey.scannetpp
- caption.seem-panoptic-osprey.scannetpp
segment_dir: /datasets/openvocab-3d-captions
segment_subset:
- segment.sam2.scannetpp
- segment.seem-panoptic.scannetpp
"""

sceneverse_dataset_cfg = """
_target_: src.data.sceneverse.dataset.SceneverseDataset
data_dir: /datasets/SceneVerse
split: train
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
"""

regionplc_dataset_cfg = """
_target_: src.data.scannet.dataset.ScanNetDataset
data_dir: /datasets/scannet_hf
split: train
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
caption_dir: /datasets/regionplc_converted
caption_subset:
- detic-template_and_kosmos_125k_iou0.2
"""

leo_dataset_cfg = """
_target_: src.data.leo.dataset.LeoDataset
data_dir: /datasets/leo
split: train
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
"""

embodiedscan_dataset_cfg = """
_target_: src.data.embodiedscan.dataset.EmbodiedScanDataset
data_dir: /datasets/mmscan_data/embodiedscan_split/embodiedscan-v1
split: train
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
"""

mmscan_dataset_cfg = """
_target_: src.data.mmscan.dataset.MMScanDataset
data_dir: /datasets/mmscan_data/MMScan-beta-release
split: train
transforms:
  - type: Copy
    keys_dict:
      coord: origin_coord
"""

CLASS_LABELS_DICT = {label: i for i, label in enumerate(CLASS_LABELS_200)}


def measure_coverage(dataset, dataset_name: str):
    coverage = []
    for i in track(range(len(dataset)), description=f"Processing {dataset_name} dataset"):
        sample = dataset[i]
        covered = np.zeros(sample["coord"].shape[0])
        for idx in sample["caption_data"]["idx"]:
            covered[idx] = 1
        coverage.append(covered.mean())
    return np.mean(coverage)


def measure_category_frequency(dataset, dataset_name: str):
    occurance = []
    for i in track(range(len(dataset)), description=f"Processing {dataset_name} dataset"):
        sample = dataset[i]
        # Process all captions in parallel using any()
        captions = sample["caption_data"]["caption"]
        has_class = [
            any(word in CLASS_LABELS_DICT for word in caption.strip().split())
            for caption in captions
        ]
        occurance.extend(has_class)
    return np.mean(occurance)


def measure_lexical_diversity(datset, dataset_name: str):
    lexical_diversity = []
    for i in track(range(len(datset)), description=f"Processing {dataset_name} dataset"):
        sample = datset[i]
        captions = sample["caption_data"]["caption"]
        for caption in captions:
            flt = ld.flemmatize(caption.strip())
            ttr = ld.ttr(flt)
            lexical_diversity.append(ttr)
    return np.mean(lexical_diversity)


def main(dataset: str, mode: str):
    if mode == "coverage":
        measure_fn = measure_coverage
    elif mode == "category":
        measure_fn = measure_category_frequency
    elif mode == "lexical":
        measure_fn = measure_lexical_diversity
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if dataset == "ours":
        scannet_train_cfg = OmegaConf.create(scannet_dataset_cfg)
        arkitscenes_train_cfg = OmegaConf.create(arkitscenes_dataset_cfg)
        matterport3d_train_cfg = OmegaConf.create(matterport3d_dataset_cfg)
        structured3d_train_cfg = OmegaConf.create(structured3d_dataset_cfg)
        scannetpp_train_cfg = OmegaConf.create(scannetpp_dataset_cfg)

        scannet_train = hydra.utils.instantiate(scannet_train_cfg)
        arkitscenes_train = hydra.utils.instantiate(arkitscenes_train_cfg)
        matterport3d_train = hydra.utils.instantiate(matterport3d_train_cfg)
        structured3d_train = hydra.utils.instantiate(structured3d_train_cfg)
        scannetpp_train = hydra.utils.instantiate(scannetpp_train_cfg)

        datasets = [
            (scannet_train, "scannet"),
            (arkitscenes_train, "arkitscenes"),
            (matterport3d_train, "matterport3d"),
            (structured3d_train, "structured3d"),
            (scannetpp_train, "scannetpp"),
        ]

        results = {}
        for dataset, name in datasets:
            results[name] = measure_fn(dataset, name)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
    elif dataset == "sceneverse":
        sceneverse_train_cfg = OmegaConf.create(sceneverse_dataset_cfg)
        sceneverse_train = hydra.utils.instantiate(sceneverse_train_cfg)
        result = measure_fn(sceneverse_train, "sceneverse")
    elif dataset == "regionplc":
        regionplc_train_cfg = OmegaConf.create(regionplc_dataset_cfg)
        regionplc_train = hydra.utils.instantiate(regionplc_train_cfg)
        result = measure_fn(regionplc_train, "regionplc")
    elif dataset == "leo":
        leo_train_cfg = OmegaConf.create(leo_dataset_cfg)
        leo_train = hydra.utils.instantiate(leo_train_cfg)
        result = measure_fn(leo_train, "leo")
    elif dataset == "embodiedscan":
        embodiedscan_train_cfg = OmegaConf.create(embodiedscan_dataset_cfg)
        embodiedscan_train = hydra.utils.instantiate(embodiedscan_train_cfg)
        result = measure_fn(embodiedscan_train, "embodiedscan")
    elif dataset == "mmscan":
        mmscan_train_cfg = OmegaConf.create(mmscan_dataset_cfg)
        mmscan_train = hydra.utils.instantiate(mmscan_train_cfg)
        result = measure_fn(mmscan_train, "mmscan")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    print(f"{dataset}: {result:.4f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
