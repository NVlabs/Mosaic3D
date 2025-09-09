import time
from pathlib import Path

import numpy as np
import viser


def split_list_into_chunks(lst, chunk_sizes):
    """
    Split a list into variable-size chunks.

    Args:
        lst (list): The input list to be split.
        chunk_sizes (list): A list of integers representing the sizes of each chunk.

    Returns:
        list: A list of sublists, where each sublist is a chunk of the input list.
    """
    if sum(chunk_sizes) != len(lst):
        raise ValueError("Sum of chunk sizes must equal the length of the input list")

    return [lst[sum(chunk_sizes[:i]) : sum(chunk_sizes[: i + 1])] for i in range(len(chunk_sizes))]


def unpack_list_of_np_arrays(filename):
    with np.load(filename) as data:
        packed = data["packed"]
        if "outer_lengths" in data:
            # Unpack list of list of 1D numpy arrays
            outer_lengths = data["outer_lengths"]
            inner_lengths = data["inner_lengths"]
            inner_splits = np.split(packed, np.cumsum(inner_lengths)[:-1])
            outer_splits = split_list_into_chunks(inner_splits, outer_lengths)
            return outer_splits
        else:
            # Unpack list of 1D numpy arrays
            lengths = data["lengths"]
            return [np.array(arr) for arr in np.split(packed, np.cumsum(lengths)[:-1])]


def visualize_scene(scene_dir: str):
    scene_dir = Path(scene_dir)
    assert scene_dir.exists(), f"Scene directory {scene_dir} does not exist"

    coords, colors = np.load(scene_dir / "coord.npy"), np.load(scene_dir / "color.npy")

    captions = unpack_list_of_np_arrays(scene_dir / "captions.npz")
    captions = [item for sublist in captions for item in sublist]

    point_indices = unpack_list_of_np_arrays(scene_dir / "point_indices.npz")
    point_indices = [item for sublist in point_indices for item in sublist]

    num_objects = len(captions)

    rand_idx = np.random.randint(0, num_objects)
    colors_for_vis = colors.copy()
    # visualize the random object in red
    colors_for_vis[point_indices[rand_idx]] = [255, 0, 0]
    colors_for_vis /= 255.0

    server = viser.Server()

    server.scene.add_point_cloud(
        name="point cloud",
        points=coords,
        colors=colors_for_vis,
        point_size=0.02,
    )
    print(f"caption: {captions[rand_idx]}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    args = parser.parse_args()

    visualize_scene(args.scene_dir)
