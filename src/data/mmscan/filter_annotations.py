import json
from pathlib import Path

import torch
from tqdm import tqdm


def filter_erroneous_annotations(anno_type: str):
    data_dir = Path(
        "/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/mmscan_data/MMScan-beta-release"
    )

    # mp3d mapping
    with open("src/data/embodiedscan/meta_data/mp3d_mapping.json") as f:
        mp3d_mapping = json.load(f)

    def convert_scan_id(scan_id: str):
        if scan_id.startswith("matterport3d"):
            scene, room = scan_id.split("/")[1:]
            scene_mapped = mp3d_mapping[scene]
            scan_id = f"{scene_mapped}_{room}"
        else:
            scan_id = scan_id.split("/")[-1]
        return scan_id

    if anno_type == "vg":
        input_file = "MMScan_VG.json"
        output_file = "MMScan_VG_filtered.json"
    elif anno_type == "oc":
        input_file = "MMScan_Caption_object.json"
        output_file = "MMScan_Caption_object_filtered.json"
    else:
        raise ValueError(f"Unknown annotation type: {anno_type}")

    # Load raw annotations
    with open(data_dir / "MMScan_samples" / input_file) as f:
        anno_raw = json.load(f)

    anno = {
        "train": [],
        "val": [],
    }

    # Filter annotations
    for split in ["train", "val"]:
        for data in tqdm(anno_raw[split]):
            scan_id = convert_scan_id(data["scan_id"])
            target_ids = data["target_id"] if anno_type == "vg" else data["object_id"]
            instance = torch.load(
                data_dir.parent
                / "embodiedscan_split"
                / "embodiedscan-v1"
                / "process_pcd"
                / f"{scan_id}.pth"
            )[-1]

            # Skip if any target_id is not in instance
            if anno_type == "vg":
                valid = all(target_id in instance for target_id in target_ids)
            else:
                valid = target_ids in instance

            if valid:
                anno[split].append(data)

    # Save filtered annotations
    with open(data_dir / "MMScan_samples" / output_file, "w") as f:
        json.dump(anno, f)

    # Print statistics
    type_str = "VG" if anno_type == "vg" else "OC"
    print(f"[{type_str} - train] {len(anno_raw['train'])} -> {len(anno['train'])}")
    print(f"[{type_str} - val] {len(anno_raw['val'])} -> {len(anno['val'])}")


if __name__ == "__main__":
    filter_erroneous_annotations("vg")
    filter_erroneous_annotations("oc")
