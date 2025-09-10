from contextlib import nullcontext
from typing import Dict, List, Optional

import pointops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from spconv.pytorch.core import ImplicitGemmIndiceData
from torch_scatter import scatter

from src.models.networks.opensegment3d.modules import (
    CrossAttentionLayer,
    FFNLayer,
    GenericMLP,
    PositionEmbeddingCoordsSine,
    SelfAttentionLayer,
)
from src.models.utils.misc import batch2offset
from src.models.utils.structure import Point
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class OpenSegment3D(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        feedforward_dim: int,
        clip_dim: int,
        num_queries: int,
        num_heads: int,
        decoder_iterations: int,
        max_sample_sizes: List[int],
        hlevels: List[int],
        backbone_ckpt: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        assert self.backbone.out_fpn, "Backbone must output FPN features"

        self.hidden_dim = hidden_dim
        self.feedforward_dim = feedforward_dim
        self.clip_dim = clip_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.decoder_iterations = decoder_iterations
        self.max_sample_sizes = max_sample_sizes
        self.hlevels = hlevels
        self.num_hlevels = len(hlevels)
        self.num_backbone_levels = self.backbone.num_stages + 1

        backbone_decoder_channels = self.backbone.channels[-self.num_backbone_levels :]
        backbone_decoder_channels[-1] = self.backbone.out_channels
        self.decoder_proj = nn.Linear(self.backbone.out_channels, hidden_dim)
        self.query_proj = GenericMLP(
            input_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim],
            output_dim=self.hidden_dim,
            use_conv=False,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        # Transformer decoder
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=hidden_dim,
            gauss_scale=1.0,
            normalize=True,
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.linear = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn = nn.ModuleList()
        for hlevel in self.hlevels:  # coarse to fine
            self.linear.append(nn.Linear(backbone_decoder_channels[hlevel], hidden_dim))
            self.cross_attention.append(CrossAttentionLayer(d_model=hidden_dim, nhead=num_heads))
            self.self_attention.append(SelfAttentionLayer(d_model=hidden_dim, nhead=num_heads))
            self.ffn.append(FFNLayer(d_model=hidden_dim, dim_feedforward=feedforward_dim))

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, 2)  # for Segment3D ckpt compatibility
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.clip_head = nn.Linear(hidden_dim, clip_dim)

        # rearrange layers
        self.b2q = Rearrange("b q d -> q b d")
        self.q2b = Rearrange("q b d -> b q d")

        if backbone_ckpt is not None:
            self.load_pretrained_backbone(backbone_ckpt)

        self.backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()
            self.backbone_frozen = True

        if decoder_ckpt is not None:
            self.load_pretrained_decoder(decoder_ckpt)

    def forward(self, input_dict: dict, num_queries: Optional[int] = None):
        num_queries = num_queries or self.num_queries

        # backbone
        with torch.no_grad() if self.backbone_frozen else nullcontext():
            if self.backbone_frozen and self.backbone.training:
                self.backbone.eval()
            point, fpn_stensors = self.backbone(input_dict)

        voxel_indices = point.sparse_conv_feat.indices
        batch_size = point.sparse_conv_feat.batch_size
        dtype = point.sparse_conv_feat.features.dtype
        device = point.sparse_conv_feat.features.device
        batch_splits_all = self.get_batch_splits(fpn_stensors, batch_size)

        # positional encodings
        centroids_all = self.get_centroids(point, self.num_backbone_levels)
        pos_encs_all = self.get_positional_encodings(
            batch_splits_all, centroids_all, dtype, device
        )

        # point features
        point_features = self.decoder_proj(fpn_stensors[-1].features)
        decomposed_pfeats = [point_features[batch_split] for batch_split in batch_splits_all[-1]]

        # query sampling
        voxel_coords = voxel_indices[:, 1:]
        batch_size = point.sparse_conv_feat.batch_size
        offset = batch2offset(voxel_indices[:, 0])
        new_offset = torch.tensor(
            [num_queries * (i + 1) for i in range(batch_size)],
            dtype=torch.long,
            device=offset.device,
        )
        query_indices = pointops.farthest_point_sampling(voxel_coords.float(), offset, new_offset)
        query_indices = query_indices.long()

        # queries
        query_pos = pos_encs_all[-1][query_indices]
        query_pos = self.query_proj(query_pos)
        query_pos = rearrange(query_pos, "(b q) d -> b q d", q=num_queries)
        queries = torch.zeros_like(query_pos)

        # decoding
        for _ in range(self.decoder_iterations):  # shared decoder weights
            for i, hlevel in enumerate(self.hlevels):  # coarse to fine
                # (1) mask module
                attn_mask = self.attention_mask_module(
                    queries,
                    decomposed_pfeats,
                    point.sparse_conv_feat.indice_dict,
                    self.num_backbone_levels - hlevel - 1,
                )

                # (2) query refinement
                batch_splits = batch_splits_all[hlevel]
                fpn_stensor = fpn_stensors[hlevel]
                pos_encs = pos_encs_all[hlevel]

                decomposed_fpn = [
                    fpn_stensor.features[batch_split] for batch_split in batch_splits
                ]
                decomposed_attn = [attn_mask[batch_split] for batch_split in batch_splits]
                decomposed_pos_enc = [pos_encs[batch_split] for batch_split in batch_splits]

                # key-value sampling
                sample_size = max([len(x) for x in decomposed_fpn])
                if min([len(x) for x in decomposed_fpn]) == 1:
                    raise RuntimeError("only a single point gives nans in cross-attention")

                if self.training:
                    sample_size = min(sample_size, self.max_sample_sizes[hlevel])

                indices_all = []
                masks_all = []
                for features in decomposed_fpn:
                    num_voxels = len(features)
                    if num_voxels <= sample_size:
                        indices = torch.zeros(sample_size, dtype=torch.long, device=device)
                        indices[:num_voxels] = torch.arange(num_voxels, device=device)
                        masks = torch.ones(sample_size, dtype=torch.bool, device=device)
                        masks[:num_voxels] = False
                    else:
                        indices = torch.randperm(num_voxels, device=device)[:sample_size]
                        masks = torch.zeros(sample_size, dtype=torch.bool, device=device)

                    indices_all.append(indices)
                    masks_all.append(masks)

                # batchify
                batched_fpn = torch.stack(
                    [features[indices] for features, indices in zip(decomposed_fpn, indices_all)]
                )
                batched_attn = torch.stack(
                    [attn[indices] for attn, indices in zip(decomposed_attn, indices_all)]
                )
                batched_pos_enc = torch.stack(
                    [
                        pos_encs[indices]
                        for pos_encs, indices in zip(decomposed_pos_enc, indices_all)
                    ]
                )

                batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == sample_size] = False
                batched_masks = torch.vstack(masks_all)
                batched_attn = torch.logical_or(batched_attn, batched_masks[..., None])

                # transformer decoder
                batched_attn = repeat(batched_attn, "b q n -> (b h) n q", h=self.num_heads)
                batched_fpn = self.linear[i](batched_fpn)
                queries = self.cross_attention[i](
                    self.b2q(queries),
                    self.b2q(batched_fpn),
                    memory_mask=batched_attn,
                    memory_key_padding_mask=None,
                    pos=self.b2q(batched_pos_enc),
                    query_pos=self.b2q(query_pos),
                )
                queries = self.self_attention[i](queries, query_pos=self.b2q(query_pos))
                queries = self.ffn[i](queries)
                queries = self.q2b(queries)

        pred_classes, pred_masks, pred_clip_feats = self.mask_module(
            queries,
            decomposed_pfeats,
        )

        return dict(
            backbone_point=point,
            logit=pred_classes,  # [B, Q, 1]
            mask=pred_masks,  # List[[N, Q]]
            clip_feat=pred_clip_feats,  # [B, Q, D]
        )

    def mask_module(self, queries: torch.Tensor, decomposed_point_feats: List[torch.Tensor]):
        queries = self.decoder_norm(queries)
        pred_classes = self.class_head(queries)
        mask_embeds = self.mask_head(queries)
        pred_masks = [
            point_feats @ mask_embed.T
            for point_feats, mask_embed in zip(decomposed_point_feats, mask_embeds)
        ]
        pred_clip_feats = self.clip_head(queries)
        return pred_classes, pred_masks, pred_clip_feats

    def attention_mask_module(
        self,
        queries: torch.Tensor,
        decomposed_point_feats: List[torch.Tensor],
        indice_dict: Dict[str, ImplicitGemmIndiceData],
        num_pooling_steps: int,
    ):
        queries = self.decoder_norm(queries)
        mask_embeds = self.mask_head(queries)
        pred_masks = [
            point_feats @ mask_embed.T
            for point_feats, mask_embed in zip(decomposed_point_feats, mask_embeds)
        ]
        attn_masks = torch.vstack(pred_masks)
        for i in range(num_pooling_steps):
            indice_data = indice_dict[f"spconv{i + 1}"]
            attn_masks, _ = spconv.ops.indice_avgpool_implicit_gemm(
                attn_masks, indice_data.pair_fwd, indice_data.pair_fwd.shape[1], False
            )
        attn_masks = attn_masks < 0

        return attn_masks.detach()

    def get_batch_splits(self, stensors: List[spconv.SparseConvTensor], batch_size: int):
        batch_splits_all = []
        for stensor in stensors:
            indices = stensor.indices
            batch_splits = []
            for i in range(batch_size):
                indices_i = torch.where(indices[:, 0] == i)[0]
                batch_splits.append(indices_i)
            batch_splits_all.append(batch_splits)

        return batch_splits_all

    def get_centroids(self, point: Point, max_hlevel: int):
        centroids = scatter(point.coord, point.v2p_map, dim=0, reduce="mean")
        indice_dict = point.sparse_conv_feat.indice_dict

        with torch.no_grad():
            centroids_all = [centroids]
            for i in range(max_hlevel - 1):
                indice_data = indice_dict[f"spconv{i + 1}"]
                centroids_down, _ = spconv.ops.indice_avgpool_implicit_gemm(
                    centroids_all[i],
                    indice_data.pair_fwd,
                    indice_data.pair_fwd.shape[1],
                    False,
                )
                centroids_all.append(centroids_down)
            centroids_all.reverse()  # coarse to fine

        return centroids_all

    def get_positional_encodings(
        self,
        batch_splits_all: List[List[torch.Tensor]],
        centroids_all: List[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        pos_encs_all = []
        for batch_splits, centroids in zip(batch_splits_all, centroids_all):
            num_voxels = len(centroids)

            pos_encs = torch.zeros(num_voxels, self.hidden_dim, dtype=dtype, device=device)
            for batch_split in batch_splits:
                masked_centroids = centroids[batch_split]
                scene_bounds = [
                    masked_centroids.min(dim=0, keepdim=True).values,
                    masked_centroids.max(dim=0, keepdim=True).values,
                ]

                pos_enc = self.pos_enc(masked_centroids[None, ...], input_range=scene_bounds)
                pos_enc = pos_enc.to(dtype)
                pos_enc = rearrange(pos_enc, "1 d n -> n d")
                pos_encs[batch_split] = pos_enc

            pos_encs_all.append(pos_encs)

        return pos_encs_all

    def load_pretrained_backbone(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                new_state_dict[k[len("net.") :]] = v

        self.backbone.load_state_dict(new_state_dict)
        log.info(f"Loaded pretrained backbone from {ckpt_path}")

    def load_pretrained_decoder(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                continue
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)
        log.info(f"Loaded pretrained decoder from {ckpt_path}")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.info("Backbone frozen")


if __name__ == "__main__":
    from src.models.networks.spunet.spconv_unet_v1m1_base import SpUNetBase

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = SpUNetBase(
        in_channels=3,
        out_channels=768,  # recap-clip
        base_channels=32,
        channels=[32, 64, 128, 256, 256, 128, 96, 96],
        layers=[2, 3, 4, 6, 2, 2, 2, 2],
        out_fpn=True,
        hash_method="ravel",
    )
    model = OpenSegment3D(
        backbone=backbone,
        hidden_dim=128,
        feedforward_dim=1024,
        clip_dim=512,
        num_queries=150,
        num_heads=8,
        decoder_iterations=3,
        max_sample_sizes=[200, 800, 3200, 12800, 51200],
        hlevels=[4],
        backbone_ckpt="results/550723/checkpoints/epoch_684-step_0068500.ckpt",
        decoder_ckpt="ckpts/segment3d_patched.ckpt",
    ).to(device)
    print(">>> OpenSegment3D initialized")

    input_dict = {
        "grid_size": 0.02,
        "coord": torch.rand(10000, 3, device=device),
        "offset": torch.tensor([4000, 10000], dtype=torch.long, device=device),
        "feat": torch.rand(10000, 3, device=device),
    }
    out_dict = model(input_dict)
    print(">>> Forward passed")
    for k, v in out_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}.shape:", v.shape)
        elif isinstance(v, list):
            print(f"    len({k}):", len(v))
            for i, vv in enumerate(v):
                print(f"        {k}[{i}].shape:", vv.shape)
