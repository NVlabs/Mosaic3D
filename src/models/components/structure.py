import numpy as np
import spconv.pytorch as spconv
import torch
from addict import Dict
from torch_scatter import scatter

from src.models.components.misc import batch2offset, offset2batch

# from pointcept.models.utils import batch2offset, offset2batch
# from pointcept.models.utils.serialization import decode, encode


def encode():
    pass


def fnv_hash_vec(arr):
    """FNV64-1A."""
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def mean_pooling(coordinates, features, return_inverse: bool = True):
    device = features.device

    key = fnv_hash_vec(coordinates.cpu().numpy())
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, index, inverse, count = np.unique(
        key_sort, return_index=True, return_inverse=True, return_counts=True
    )

    voxel_ids = offset2batch(torch.from_numpy(np.cumsum(count)).to(device))
    voxel_coords = coordinates[idx_sort[index]]
    voxel_feats = scatter(features[idx_sort], voxel_ids, dim=0, reduce="mean")
    v2p_map = np.zeros_like(inverse)
    v2p_map[idx_sort] = inverse
    v2p_map = torch.from_numpy(v2p_map).to(device).long()

    if not return_inverse:
        return voxel_coords, voxel_feats
    else:
        return voxel_coords, voxel_feats, v2p_map


class Point(Dict):
    """Point Structure of Pointcept.

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """Point Cloud Serialization.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipeline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """Point Cloud Serialization.

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipeline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            grid_coord = (self.coord / self.grid_size).int()
            self["grid_coord"] = grid_coord - grid_coord.min(0)[0]
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()

        batched_coords = torch.cat(
            [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
        ).contiguous()

        voxel_coords, voxel_feats, v2p_map = mean_pooling(
            batched_coords, self.feat, return_inverse=True
        )
        self.v2p_map = v2p_map

        sparse_conv_feat = spconv.SparseConvTensor(
            features=voxel_feats,
            indices=voxel_coords,
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )

        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat
