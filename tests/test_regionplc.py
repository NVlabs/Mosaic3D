# Test the regionplc SparseConvUNetTextSeg vs. warpconvnet RegionPLCToCLIP
from typing import Sequence, Dict
import unittest
import hydra
from omegaconf import DictConfig
from copy import deepcopy

import torch
import torch.nn as nn
import warp as wp
from warpconvnet.utils.batch_index import batch_index_from_offset
from warpconvnet.utils.unique import unique_inverse

from src.models.networks.regionplc.sparse_convnet import SparseConvUNetTextSeg
from src.models.networks.warp.regionplc import RegionPLCToCLIP, to_warp_sparse_tensor
from src.models.components.structure import mean_pooling


def sp2wp_weight(weight_sp):
    return weight_sp.permute(1, 2, 3, 4, 0).flatten(0, 2)


def match_conv_weights(conv_sp, conv_wp):
    # assert the shape of conv_wp.weight is (3^3, C_in, C_out) matches the shape of conv_sp.weight
    conv_weight_wp = sp2wp_weight(conv_sp.weight.data)
    assert (
        conv_weight_wp.shape == conv_wp.weight.shape
    ), f"conv_wp.weight.shape: {conv_wp.weight.shape}, conv_weight_wp.shape: {conv_weight_wp.shape}"
    conv_wp.weight.data = conv_weight_wp


def match_bn_weights(bn_sp, bn_wp):
    bn_wp.norm.weight.data = bn_sp.weight.data
    bn_wp.norm.bias.data = bn_sp.bias.data
    bn_wp.norm.running_mean.data = bn_sp.running_mean.data
    bn_wp.norm.running_var.data = bn_sp.running_var.data


def match_weights(conv_sp, conv_wp):
    # 1. Input convolution
    match_conv_weights(conv_sp.input_conv[0], conv_wp.input_conv)

    # 2. UBlock layers
    match_ublock_weights(conv_sp.unet, conv_wp.unet)

    # 3. Output layer
    match_bn_weights(conv_sp.output_layer[0], conv_wp.output_layer[0])


def match_ublock_weights(ublock_sp, ublock_wp):
    # Match weights for the main blocks
    for i, (block_sp, block_wp) in enumerate(zip(ublock_sp.blocks, ublock_wp.blocks)):
        match_residual_block_weights(block_sp, block_wp)

    # Match weights for the conv layer
    if hasattr(ublock_sp, "conv") and hasattr(ublock_wp, "conv"):
        match_bn_weights(ublock_sp.conv[0], ublock_wp.conv[0])
        match_conv_weights(ublock_sp.conv[2], ublock_wp.conv[2])

    # Recursively match weights for nested UBlocks
    if hasattr(ublock_sp, "u") and hasattr(ublock_wp, "u"):
        match_ublock_weights(ublock_sp.u, ublock_wp.u)

    # Match weights for the deconv layer
    if hasattr(ublock_sp, "deconv") and hasattr(ublock_wp, "deconv"):
        match_bn_weights(ublock_sp.deconv[0], ublock_wp.deconv_pre[0])
        match_conv_weights(ublock_sp.deconv[2], ublock_wp.deconv)

    # Match weights for the blocks_tail
    if hasattr(ublock_sp, "blocks_tail") and hasattr(ublock_wp, "blocks_tail"):
        for block_sp, block_wp in zip(ublock_sp.blocks_tail, ublock_wp.blocks_tail):
            match_residual_block_weights(block_sp, block_wp)


def match_residual_block_weights(block_sp, block_wp):
    # Match i_branch weights if it's not an Identity
    if not isinstance(block_sp.i_branch[0], nn.Identity):
        match_conv_weights(block_sp.i_branch[0], block_wp.i_branch)

    # Match conv_branch weights
    # 1 and 4 are ReLU
    match_bn_weights(block_sp.conv_branch[0], block_wp.conv_branch[0])
    match_conv_weights(block_sp.conv_branch[2], block_wp.conv_branch[2])
    match_bn_weights(block_sp.conv_branch[3], block_wp.conv_branch[3])
    match_conv_weights(block_sp.conv_branch[5], block_wp.conv_branch[5])


def match_linear_weights(linear_sp, linear_wp):
    linear_wp.weight.data = linear_sp.weight.data
    if linear_wp.bias is not None:
        linear_wp.bias.data = linear_sp.bias.data


def match_adapter_weights(adapter_sp, adapter_wp):
    match_linear_weights(adapter_sp.adapter[0], adapter_wp.adapter[0])
    match_linear_weights(adapter_sp.adapter[3], adapter_wp.adapter[3])


def get_dataloader():
    config_path = "../configs/"
    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name="train.yaml", overrides=["experiment=regionplc"])
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    return datamodule.train_dataloader()


class TestRegionPLC(unittest.TestCase):
    def setUp(self):
        wp.init()

    def test_load_model(self):
        spconv_model = SparseConvUNetTextSeg(
            backbone_cfg=DictConfig(
                {
                    "in_channel": 3,
                    "mid_channel": 16,
                    "block_reps": 2,
                    "block_residual": True,
                    "custom_sp1x1": True,
                    "num_blocks": 7,
                }
            ),
            adapter_cfg=DictConfig(
                {
                    "in_channel": 16,
                    "text_channel": 512,
                    "num_layers": 2,
                    "last_norm": True,
                }
            ),
        )

        wp_model = RegionPLCToCLIP(
            backbone_cfg=DictConfig(
                {
                    "in_channel": 3,
                    "in_channels": [16, 32, 48, 64, 80, 96, 112, 128],
                    "out_channels": [16, 32, 48, 64, 80, 96, 112, 128],
                    "num_blocks": 2,
                }
            ),
            adapter_cfg=DictConfig(
                {
                    "in_channel": 192,
                    "text_channel": 512,
                    "last_norm": True,
                    "normalize_output": True,
                }
            ),
            voxel_size=0.02,
        )

    def test_data_collate(self):
        batch = next(iter(get_dataloader()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            elif isinstance(v, Sequence) and not isinstance(v, torch.Tensor):
                print(k, [x.shape if isinstance(x, torch.Tensor) else x for x in v])
            elif isinstance(v, Dict):
                print(k, v.keys())
            else:
                print(k, v)

        # Create a warp sparse tensor
        sparse_conv_feat, data_dict = to_warp_sparse_tensor(batch)
        # Check the v2p map
        v2p_map = data_dict["v2p_map"]
        assert v2p_map.shape == (batch["coord"].shape[0],)

        batch_indices = batch_index_from_offset(data_dict["offset"])
        batched_coords = torch.cat(
            [batch_indices.view(-1, 1).int(), data_dict["grid_coord"].int()], dim=1
        ).contiguous()

        outs = []
        for hash_method in ["fnv", "ravel"]:
            voxel_coords, voxel_feats, v2p_map = mean_pooling(
                batched_coords,
                batch["feat"],
                return_inverse=True,
                hash_method=hash_method,
            )
            outs.append([voxel_coords, voxel_feats, v2p_map])

        fnv_indices, _ = unique_inverse(outs[0][0])
        fnv_voxel_feats = outs[0][1][fnv_indices]
        fnv_voxel_coords = outs[0][0][fnv_indices]
        fnv_voxel_coords_v2p = outs[0][0][outs[0][2]]

        ravel_indices, _ = unique_inverse(outs[1][0])
        ravel_voxel_feats = outs[1][1][ravel_indices]
        ravel_voxel_coords = outs[1][0][ravel_indices]
        ravel_voxel_coords_v2p = outs[1][0][outs[1][2]]

        assert torch.allclose(fnv_voxel_coords, ravel_voxel_coords)
        assert torch.allclose(fnv_voxel_feats, ravel_voxel_feats)
        assert torch.allclose(fnv_voxel_coords_v2p, ravel_voxel_coords_v2p)

    def test_regionplc_forward(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        channels = [16 * (i + 1) for i in range(7)]
        warp_model = RegionPLCToCLIP(
            backbone_cfg=DictConfig(
                {
                    "in_channel": 6,
                    "in_channels": channels,
                    "out_channels": channels,
                    "num_blocks": 2,
                }
            ),
            adapter_cfg=DictConfig(
                {
                    "in_channel": 16,
                    "text_channel": 512,
                    "num_layers": 2,
                    "last_norm": True,
                }
            ),
            point_to_voxel_method="mean",
        ).to(device)
        spconv_model = SparseConvUNetTextSeg(
            backbone_cfg=DictConfig(
                {
                    "in_channel": 6,
                    "mid_channel": 16,
                    "block_reps": 2,
                    "block_residual": True,
                    "custom_sp1x1": True,
                    "num_blocks": 7,
                }
            ),
            adapter_cfg=DictConfig(
                {
                    "in_channel": 16,
                    "text_channel": 512,
                    "num_layers": 2,
                    "last_norm": True,
                }
            ),
        ).to(device)
        # Match the weights
        match_weights(spconv_model.backbone, warp_model.backbone)
        match_adapter_weights(spconv_model.adapter, warp_model.adapter)

        batch = next(iter(get_dataloader()))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Deepcopy
        out_warp = warp_model(deepcopy(batch))
        out_spconv = spconv_model(deepcopy(batch))

        # Check the clip feats are the same
        diff = (
            out_warp["clip_feat"] - out_spconv.sparse_conv_feat.features[out_spconv.v2p_map]
        ).abs()
        print(diff.max())
        print(diff.mean())


if __name__ == "__main__":
    unittest.main()
