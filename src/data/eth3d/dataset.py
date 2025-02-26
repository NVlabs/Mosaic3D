from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import OmegaConf
from plyfile import PlyData
from torch.utils.data import Dataset

from src.data.transform import Compose
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ETH3DDataset(Dataset):
    CLASS_LABELS = ["background", "foreground"]

    def __init__(self, data_dir: str, transforms: Optional[OmegaConf] = None):
        super().__init__()

        self.dataset_name = "eth3d"
        self.log_postfix = "eth3d"

        self.base_class_idx = None
        self.novel_class_idx = None
        self.ignore_class_idx = None
        self.instance_ignore_class_idx = None
        self.ignore_label = -100

        self.fg_class_idx = [0]
        self.bg_class_idx = [1]
        self.mask_dir = None
        self.repeat = 1
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"{self.data_dir} not exist."

        self.scenes = [
            "courtyard",
            "delivery_area",
            "electro",
            "facade",
            "kicker",
            "meadow",
            "office",
            "pipes",
            "playground",
            "relief",
            "relief_2",
            "terrace",
            "terrains",
        ]

        if transforms is not None:
            transforms_cfg = OmegaConf.to_container(transforms)
            self.transforms = Compose(transforms_cfg)
        else:
            self.transforms = lambda x: x

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene_name = self.scenes[idx]

        ply_file = self.data_dir / scene_name / "scan_clean" / "scan_aggr.ply"
        # read ply file
        with open(ply_file, "rb") as f:
            plydata = PlyData.read(f)

        vertices = plydata["vertex"]
        coord = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
        colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
        colors = (colors * 255.0).astype(np.uint8)

        segment = np.ones_like(coord[:, 0]).astype(np.int64) * self.ignore_label  # all are ignored
        binary_label = np.ones_like(segment)
        origin_idx = np.arange(coord.shape[0]).astype(np.int64)

        data_dict = dict(
            coord=coord,
            color=colors,
            segment=segment,
            binary=binary_label,
            origin_idx=origin_idx,
        )
        data_dict = self.transforms(data_dict)
        return data_dict


def parse_matrix(matrix_text):
    """Parse the matrix text into a numpy array"""
    # Split the text into rows and convert to floats
    rows = []
    for line in matrix_text.strip().split("\n"):
        # Convert each row into float values
        row = [float(x) for x in line.strip().split()]
        rows.append(row)
    return np.array(rows)


def parse_meshlab_file(filename):
    import xml.etree.ElementTree as ET

    """Parse MeshLab file and return a dictionary of matrices"""
    # Parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Dictionary to store matrices
    matrices = {}

    # Find all MLMesh elements
    for mesh in root.findall(".//MLMesh"):
        # Get the label/filename
        label = mesh.get("label")

        # Find the matrix element
        matrix_elem = mesh.find("MLMatrix44")
        if matrix_elem is not None:
            # Parse the matrix text
            matrix = parse_matrix(matrix_elem.text)
            matrices[label] = matrix

    return matrices


def aggregate(scene_name: str, voxel_size: float = 0.05):
    import open3d as o3d

    data_dir = Path("/datasets/eth3d")
    scene_dir = data_dir / scene_name / "scan_clean"

    # read alignment file
    alignment_file = scene_dir / "scan_alignment.mlp"
    print(f"Reading alignment file: {alignment_file}")
    matrices = parse_meshlab_file(alignment_file)

    # read pointclouds
    print(f"Reading pointclouds: {matrices.keys()}")
    pcds = {}
    for k in matrices.keys():
        print(f"Reading pointcloud: {k}")
        pcd_file = str(scene_dir / k)
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcds[k] = pcd

    # aggregate pointclouds
    print("Aggregating pointclouds")
    pcd_agg = o3d.geometry.PointCloud()
    for k in pcds.keys():
        pcd_transformed = pcds[k].transform(matrices[k])
        pcd_agg += pcd_transformed
    pcd_agg = pcd_agg.voxel_down_sample(voxel_size=voxel_size)

    print(f"Writing aggregated pointcloud to: {scene_dir / 'scan_aggr.ply'}")
    o3d.io.write_point_cloud(str(scene_dir / "scan_aggr.ply"), pcd_agg)


if __name__ == "__main__":
    dataset = ETH3DDataset(data_dir="/datasets/eth3d")
    for scene_name in dataset.scenes:
        aggregate(scene_name)
