import os
from typing import Optional
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from src.models.networks.segment3d.mask3d_no_aux import Mask3D_no_aux
from src.models.networks.segment3d.utils.pointnet2_utils import furthest_point_sample
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class Mask3D_no_aux_openvocab(Mask3D_no_aux):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
        clip_dim,
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__(
            config,
            hidden_dim,
            num_queries,
            num_heads,
            dim_feedforward,
            sample_sizes,
            shared_decoder,
            num_classes,
            num_decoders,
            dropout,
            pre_norm,
            positional_encoding_type,
            non_parametric_queries,
            train_on_segments,
            normalize_pos_enc,
            use_level_embed,
            scatter_type,
            hlevels,
            use_np_features,
            voxel_size,
            max_sample_size,
            random_queries,
            gauss_scale,
            random_query_both,
            random_normal,
        )
        self.clip_embed_head = nn.Linear(hidden_dim, clip_dim)
        self.from_pretrained(pretrained_ckpt)

    def forward(
        self, batch_dict: dict, is_eval: bool = False, num_queries: Optional[int] = None
    ) -> dict:
        tfield = batch_dict["tfield"]
        x = batch_dict["stensor"]
        point2segment = batch_dict["voxel2segment"]
        raw_coordinates = batch_dict["voxel_centroids"]
        if num_queries is not None:
            assert (
                self.non_parametric_queries
            ), "num_queries is only supported with non-parametric queries"
        num_queries = num_queries if num_queries is not None else self.num_queries

        pcd_features, aux = self.backbone(x)

        batch_size = len(x.decomposed_coordinates)

        with torch.no_grad():
            coordinates = ME.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=aux[-1].coordinate_manager,
                coordinate_map_key=aux[-1].coordinate_map_key,
                device=aux[-1].device,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)

        if point2segment is not None:
            mask_segments = []
            for i, mask_feature in enumerate(mask_features.decomposed_features):
                mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))

        sampled_coords = None

        if self.non_parametric_queries:
            fps_idx = [
                furthest_point_sample(
                    x.decomposed_coordinates[i][None, ...].float(),
                    num_queries,
                )
                .squeeze(0)
                .long()
                for i in range(len(x.decomposed_coordinates))
            ]

            sampled_coords = torch.stack(
                [
                    coordinates.decomposed_features[i][fps_idx[i].long(), :]
                    for i in range(len(fps_idx))
                ]
            )

            mins = torch.stack(
                [
                    coordinates.decomposed_features[i].min(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )
            maxs = torch.stack(
                [
                    coordinates.decomposed_features[i].max(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )

            query_pos = self.pos_enc(
                sampled_coords.float(), input_range=[mins, maxs]
            )  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)

            if not self.use_np_features:
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = torch.stack(
                    [
                        pcd_features.decomposed_features[i][fps_idx[i].long(), :]
                        for i in range(len(fps_idx))
                    ]
                )
                queries = self.np_feature_projection(queries)
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:
            query_pos = (
                torch.rand(
                    batch_size,
                    self.mask_dim,
                    num_queries,
                    device=x.device,
                )
                - 0.5
            )

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = (
                    torch.rand(
                        batch_size,
                        2 * self.mask_dim,
                        num_queries,
                        device=x.device,
                    )
                    - 0.5
                )
            else:
                query_pos_feat = torch.randn(
                    batch_size,
                    2 * self.mask_dim,
                    num_queries,
                    device=x.device,
                )

            queries = query_pos_feat[:, : self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim :, :].permute((2, 0, 1))
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)

        predictions_class = []
        predictions_mask = []
        predictions_clip = []
        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                if point2segment is not None:
                    output_class, outputs_mask, attn_mask = self.mask_module(
                        queries,
                        mask_features,
                        mask_segments,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=point2segment,
                        coords=coords,
                    )
                else:
                    output_class, outputs_mask, attn_mask = self.mask_module(
                        queries,
                        mask_features,
                        None,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=None,
                        coords=coords,
                    )

                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                curr_sample_size = max([pcd.shape[0] for pcd in decomposed_aux])

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError("only a single point gives nans in cross-attention")

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(pcd_size, device=queries.device)

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(decomposed_aux[k].shape[0], device=queries.device)[
                            :curr_sample_size
                        ]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack(
                    [decomposed_aux[k][rand_idx[k], :] for k in range(len(rand_idx))]
                )

                batched_attn = torch.stack(
                    [decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))]
                )

                batched_pos_enc = torch.stack(
                    [pos_encodings_pcd[hlevel][0][k][rand_idx[k], :] for k in range(len(rand_idx))]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == rand_idx[0].shape[0]
                ] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux.permute((1, 0, 2)))
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute(
                        (0, 2, 1)
                    ),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](output).permute((1, 0, 2))

        if point2segment is not None:
            output_class, outputs_mask, outputs_clip = self.final_mask_module(
                tfield,
                queries,
                mask_features,
                mask_segments,
                point2segment=point2segment,
            )
        else:
            output_class, outputs_mask, outputs_clip = self.final_mask_module(
                tfield,
                queries,
                mask_features,
                None,
                point2segment=None,
            )
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
        predictions_clip.append(outputs_clip)
        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_clip_feat": predictions_clip[-1],
            "sampled_coords": sampled_coords.detach() if sampled_coords is not None else None,
        }

    def mask_module(
        self,
        query_feat,
        mask_features,
        mask_segments,
        num_pooling_steps,
        ret_attn_mask=True,
        point2segment=None,
        coords=None,
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []
        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(mask_features.decomposed_features[i] @ mask_embed[i].T)

        output_masks = torch.cat(output_masks)
        outputs_mask = ME.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = ME.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            if point2segment is not None:
                return outputs_class, output_segments, attn_mask
            else:
                return (
                    outputs_class,
                    outputs_mask.decomposed_features,
                    attn_mask,
                )

        if point2segment is not None:
            return outputs_class, output_segments
        else:
            return outputs_class, outputs_mask.decomposed_features

    def final_mask_module(
        self,
        tensor_field,
        query_feat,
        mask_features,
        mask_segments,
        point2segment=None,
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        outputs_clip = self.clip_embed_head(query_feat)

        outputs_masks = []
        if point2segment is not None:
            for i in range(len(mask_segments)):
                output_segment = mask_segments[i] @ mask_embed[i].T
                outputs_masks.append(output_segment[point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                outputs_masks.append(mask_features.decomposed_features[i] @ mask_embed[i].T)

        output_masks = torch.cat(outputs_masks)
        outputs_mask = ME.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )
        outputs_mask_full = outputs_mask.slice(tensor_field).decomposed_features
        return outputs_class, outputs_mask_full, outputs_clip

    def from_pretrained(self, ckpt_path):
        if ckpt_path is None:
            return

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint {ckpt_path} not found. Please download it using download_ckpt.py"
            )

        # clip_embed_head is not included in the pre-trained checkpoint
        self.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
        log.info(f"Checkpoint except for clip_embed_head loaded from {ckpt_path}")


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra

    config = {
        "backbone": {
            "_target_": "src.models.networks.segment3d.res16unet.Res16UNet34C",
            "in_channels": 3,
            "out_channels": -1,
            "config": {"conv1_kernel_size": 5, "bn_momentum": 0.02},
            "out_fpn": True,
        }
    }
    config = OmegaConf.create(config)
    config = hydra.utils.instantiate(config)

    model_args = dict(
        config=config,
        hidden_dim=128,
        dim_feedforward=1024,
        num_queries=100,
        num_heads=8,
        num_decoders=3,
        dropout=0.0,
        pre_norm=False,
        use_level_embed=False,
        normalize_pos_enc=True,
        positional_encoding_type="fourier",
        gauss_scale=1.0,
        hlevels=[0, 1, 2, 3],
        non_parametric_queries=True,
        random_query_both=False,
        random_normal=False,
        random_queries=False,
        use_np_features=False,
        sample_sizes=[200, 800, 3200, 12800, 51200],
        max_sample_size=False,
        shared_decoder=True,
        num_classes=2,
        clip_dim=512,  # not included in the pre-trained checkpoint
        train_on_segments=False,
        scatter_type="mean",
        voxel_size=0.02,
        pretrained_ckpt="ckpts/segment3d_processed.ckpt",
    )

    model = Mask3D_no_aux_openvocab(**model_args)
    print(model)
