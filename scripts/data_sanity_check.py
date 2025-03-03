import os
from pathlib import Path

from rich.progress import Progress

from src.utils.io import unpack_list_of_np_arrays

DATA_DIRS = {
    "scannetv2": {
        "segment_dir": "mask_clustering.cropformer.scannet-125k",
        "caption_dir": "caption-mc.osprey.scannet-125k",
    },
    "arkitscenes": {
        "segment_dir": "mask_clustering.cropformer.arkitscenes+combined",
        "caption_dir": "caption-mc.osprey.arkitscenes",
    },
    "scannet++": {
        "segment_dir": "mask_clustering.cropformer.scannetpp+combined",
        "caption_dir": "caption-mc.osprey.scannetpp",
    },
    "matterport3d": {
        "segment_dir": "mask_clustering.cropformer.matterport3d+combined",
        "caption_dir": "caption-mc.osprey.matterport3d",
    },
}


def main(dataset: str, datadir: str = "/datasets/mosaic3d++"):
    segment_dir = os.path.join(datadir, DATA_DIRS[dataset]["segment_dir"])
    caption_dir = os.path.join(datadir, DATA_DIRS[dataset]["caption_dir"])

    split_files = [
        Path(__file__).parent.parent
        / "src"
        / "data"
        / "metadata"
        / "split_files"
        / f"{dataset}_{split}.txt"
        for split in ["train"]
    ]
    scenes = []
    for split_file in split_files:
        with open(split_file) as f:
            scenes.extend([line.strip() for line in f.readlines() if not line.startswith("#")])

    bad_scenes = []
    with Progress() as progress:
        task = progress.add_task(f"Sanity checking {dataset}", total=len(scenes))
        for scene in scenes:
            try:
                point_indices = unpack_list_of_np_arrays(
                    os.path.join(segment_dir, scene, "point_indices.npz")
                )
                num_point_indices = len(point_indices)
            except Exception as e:
                num_point_indices = 0
            try:
                captions = unpack_list_of_np_arrays(
                    os.path.join(caption_dir, scene, "captions-gathered.npz")
                )
                num_captions = len(captions)
            except Exception as e:
                num_captions = 0

            if num_point_indices == 0:
                progress.console.print(
                    f"[red]scene: {scene}, len(point_indices): {num_point_indices:4d}, len(captions): {num_captions:4d}, match? {num_point_indices == num_captions}"
                )
                bad_scenes.append((scene, "no point indices"))
            if num_captions == 0:
                progress.console.print(
                    f"[red]scene: {scene}, len(point_indices): {num_point_indices:4d}, len(captions): {num_captions:4d}, match? {num_point_indices == num_captions}"
                )
                bad_scenes.append((scene, "no captions"))

            progress.update(task, advance=1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
