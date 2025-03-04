from typing import Dict, Optional, Tuple
from itertools import chain
import random
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from timm.models.layers import trunc_normal_
from torch_geometric.nn.pool import voxel_grid
import pointops

from src.models.lightning_modules.language_module import DenseLanguageLitModule
from src.models.components.misc import offset2batch
from src.utils.dist_utils import get_world_size


class MaskedDenseLanguageLitModule(DenseLanguageLitModule):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        clip_encoder: Dict,
        compile: bool,
        loss_cfg: Dict,
        best_metric: str,
        eval_cfg: Optional[Dict] = None,
        use_prompt: bool = False,
    ):
        super().__init__(
            net,
            optimizer,
            scheduler,
            scheduler_interval,
            clip_encoder,
            compile,
            loss_cfg,
            best_metric,
            eval_cfg,
            use_prompt,
        )

        msc_loss_cfg = loss_cfg["msc_loss"]
        self.mask_strategy = msc_loss_cfg["mask_strategy"]
        if self.mask_strategy == "random":
            self.mask_generator = self.generate_cross_masks
        elif self.mask_strategy == "captioned":
            self.mask_generator = self.generate_cross_masks_captioned
        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
        assert msc_loss_cfg.mask_rate <= 0.5
        self.nce_criteria = nn.CrossEntropyLoss()

    def configure_model(self) -> None:
        super().configure_model()

        # modules for self-supervised learning
        msc_loss_cfg = self.hparams.loss_cfg.msc_loss
        self.mask_token = nn.Parameter(torch.zeros(1, self.net.in_channels))
        trunc_normal_(self.mask_token, mean=0, std=0.02)
        self.color_head = (
            nn.Linear(self.net.out_channels, 3) if msc_loss_cfg.reconstruct_color else None
        )
        self.normal_head = (
            nn.Linear(self.net.out_channels, 3) if msc_loss_cfg.reconstruct_normal else None
        )

    @torch.no_grad()
    def generate_cross_masks(
        self,
        view1_origin_coord: torch.Tensor,
        view1_offset: torch.Tensor,
        view2_origin_coord: torch.Tensor,
        view2_offset: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = view1_origin_coord.device
        msc_loss_cfg = self.hparams.loss_cfg.msc_loss

        # union origin coord
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()

        view1_origin_coord_split = view1_origin_coord.split(list(view1_batch_count))
        view2_origin_coord_split = view2_origin_coord.split(list(view2_batch_count))
        union_origin_coord = torch.cat(
            list(chain.from_iterable(zip(view1_origin_coord_split, view2_origin_coord_split)))
        )
        union_offset = torch.cat(
            [view1_offset.unsqueeze(-1), view2_offset.unsqueeze(-1)], dim=-1
        ).sum(-1)
        union_batch = offset2batch(union_offset)

        # grid partition
        mask_patch_coord = union_origin_coord.div(msc_loss_cfg.mask_grid_size)
        mask_patch_grid_coord = torch.floor(mask_patch_coord)
        mask_patch_cluster = voxel_grid(
            pos=mask_patch_grid_coord, size=1, batch=union_batch, start=0
        )
        unique, cluster, counts = torch.unique(
            mask_patch_cluster, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        patch_max_point = counts.max().item()
        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
        patch2point_mask = torch.lt(
            torch.arange(patch_max_point, device=device).unsqueeze(0), counts.unsqueeze(-1)
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        patch2point_map[patch2point_mask] = sorted_cluster_indices

        # generate cross masks
        patch_mask = torch.zeros(patch_num, dtype=torch.int32, device=device)
        rand_perm = torch.randperm(patch_num)
        mask_patch_num = int(patch_num * msc_loss_cfg.mask_rate)

        # mask1 tag with 1, mask2 tag with 2
        patch_mask[rand_perm[0:mask_patch_num]] = 1
        patch_mask[rand_perm[mask_patch_num : mask_patch_num * 2]] = 2
        point_mask = torch.zeros(union_origin_coord.shape[0], dtype=torch.int32, device=device)
        point_mask[patch2point_map[patch_mask == 1][patch2point_mask[patch_mask == 1]]] = 1
        point_mask[patch2point_map[patch_mask == 2][patch2point_mask[patch_mask == 2]]] = 2

        # separate mask to view1 and view2
        point_mask_split = point_mask.split(
            list(
                torch.cat(
                    [view1_batch_count.unsqueeze(-1), view2_batch_count.unsqueeze(-1)],
                    dim=-1,
                ).flatten()
            )
        )
        view1_point_mask = torch.cat(point_mask_split[0::2]) == 1
        view2_point_mask = torch.cat(point_mask_split[1::2]) == 2
        return view1_point_mask, view2_point_mask

    @torch.no_grad()
    def generate_cross_masks_captioned(
        self,
        view1_origin_coord: torch.Tensor,
        view1_offset: torch.Tensor,
        view2_origin_coord: torch.Tensor,
        view2_offset: torch.Tensor,
        caption_data: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = view1_origin_coord.device
        msc_loss_cfg = self.hparams.loss_cfg.msc_loss

        # Process batch information
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()

        # Create union of coordinates
        view1_origin_coord_split = view1_origin_coord.split(list(view1_batch_count))
        view2_origin_coord_split = view2_origin_coord.split(list(view2_batch_count))
        union_origin_coord = torch.cat(
            list(chain.from_iterable(zip(view1_origin_coord_split, view2_origin_coord_split)))
        )
        union_offset = torch.cat(
            [view1_offset.unsqueeze(-1), view2_offset.unsqueeze(-1)], dim=-1
        ).sum(-1)
        union_batch = offset2batch(union_offset)

        # Grid partition for masking
        mask_patch_coord = union_origin_coord.div(msc_loss_cfg.mask_grid_size)
        mask_patch_grid_coord = torch.floor(mask_patch_coord)
        mask_patch_cluster = voxel_grid(
            pos=mask_patch_grid_coord, size=1, batch=union_batch, start=0
        )
        unique, cluster, counts = torch.unique(
            mask_patch_cluster, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * msc_loss_cfg.mask_rate)
        patch_max_point = counts.max().item()

        # Create patch to point mapping
        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
        patch2point_mask = torch.lt(
            torch.arange(patch_max_point, device=device).unsqueeze(0), counts.unsqueeze(-1)
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        patch2point_map[patch2point_mask] = sorted_cluster_indices

        # Create mapping from view1 points to union points
        view1_to_union_map = []
        offset = 0
        for b_idx in range(len(view1_batch_count)):
            v1_count = view1_batch_count[b_idx].item()
            v2_count = view2_batch_count[b_idx].item()
            view1_to_union_map.extend([offset + i for i in range(v1_count)])
            offset += v1_count + v2_count
        view1_to_union_map = torch.tensor(view1_to_union_map, device=device, dtype=torch.long)

        # Initialize patch mask and get caption data
        patch_mask = torch.zeros(patch_num, dtype=torch.int32, device=device)
        point_indices = caption_data["point_indices"]
        point_indices_split = point_indices.split(list(caption_data["num_points_per_caption"]))

        # Mask patches based on caption data for view1
        caption_indices = torch.randperm(len(point_indices_split))
        masked_patches_count = 0

        for caption_idx in caption_indices:
            if masked_patches_count >= mask_patch_num:
                break

            # Map caption points to union points
            caption_point_indices = point_indices_split[caption_idx]
            union_indices = view1_to_union_map[caption_point_indices]

            # Filter valid indices
            valid_mask = (union_indices >= 0) & (union_indices < union_origin_coord.shape[0])
            if not valid_mask.any():
                continue

            union_indices = union_indices[valid_mask]

            # Find and mask patches containing these points
            point_patches = cluster[union_indices]
            unique_patches = torch.unique(point_patches)
            new_patches = unique_patches[patch_mask[unique_patches] == 0]
            patch_mask[new_patches] = 1
            masked_patches_count += len(new_patches)

        # Add random patches if needed for view1
        if masked_patches_count < mask_patch_num:
            remaining_patches = torch.where(patch_mask == 0)[0]
            additional_patches = remaining_patches[
                torch.randperm(len(remaining_patches))[: mask_patch_num - masked_patches_count]
            ]
            patch_mask[additional_patches] = 1

        # Mask patches for view2
        remaining_patches = torch.where(patch_mask == 0)[0]
        view2_mask_patches = remaining_patches[
            torch.randperm(len(remaining_patches))[:mask_patch_num]
        ]
        patch_mask[view2_mask_patches] = 2

        # Convert patch masks to point masks
        point_mask = torch.zeros(union_origin_coord.shape[0], dtype=torch.int32, device=device)
        point_mask[patch2point_map[patch_mask == 1][patch2point_mask[patch_mask == 1]]] = 1
        point_mask[patch2point_map[patch_mask == 2][patch2point_mask[patch_mask == 2]]] = 2

        # Split masks for view1 and view2
        point_mask_split = point_mask.split(
            list(
                torch.cat(
                    [view1_batch_count.unsqueeze(-1), view2_batch_count.unsqueeze(-1)], dim=-1
                ).flatten()
            )
        )
        view1_point_mask = torch.cat(point_mask_split[0::2]) == 1
        view2_point_mask = torch.cat(point_mask_split[1::2]) == 2

        return view1_point_mask, view2_point_mask

    @torch.no_grad()
    def match_contrastive_pair(
        self,
        view1_coord: torch.Tensor,
        view1_offset: torch.Tensor,
        view2_coord: torch.Tensor,
        view2_offset: torch.Tensor,
        max_k: int,
        max_radius: float,
    ):
        msc_loss_cfg = self.hparams.loss_cfg.msc_loss

        index, distance = pointops.knn_query(
            max_k,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index = torch.cat(
            [
                torch.arange(index.shape[0], device=index.device, dtype=torch.long)
                .view(-1, 1, 1)
                .expand(-1, max_k, 1),
                index.view(-1, max_k, 1),
            ],
            dim=-1,
        )[distance.squeeze(-1) < max_radius]
        _, count = index[:, 0].unique(return_counts=True)
        select = (
            torch.cumsum(count, dim=0)
            - torch.randint(count.max(), count.shape, device=count.device) % count
            - 1
        )
        index = index[select]
        if index.shape[0] > msc_loss_cfg.matching_max_pair:
            index = index[torch.randperm(index.shape[0])[: msc_loss_cfg.matching_max_pair]]
        return index

    def compute_contrastive_loss(
        self,
        view1_feat: torch.Tensor,
        view1_offset: torch.Tensor,
        view2_feat: torch.Tensor,
        view2_offset: torch.Tensor,
        match_index: torch.Tensor,
    ):
        msc_loss_cfg = self.hparams.loss_cfg.msc_loss

        assert view1_offset.shape == view2_offset.shape

        view1_feat = view1_feat[match_index[:, 0]]
        view2_feat = view2_feat[match_index[:, 1]]
        view1_feat = view1_feat / (torch.norm(view1_feat, p=2, dim=1, keepdim=True) + 1e-7)
        view2_feat = view2_feat / (torch.norm(view2_feat, p=2, dim=1, keepdim=True) + 1e-7)
        sim = torch.mm(view1_feat, view2_feat.transpose(1, 0))

        with torch.no_grad():
            pos_sim = torch.diagonal(sim).mean()
            neg_sim = sim.mean(dim=-1).mean() - pos_sim / match_index.shape[0]
        labels = torch.arange(sim.shape[0], dtype=torch.long, device=view1_feat.device)
        loss = self.nce_criteria(torch.div(sim, msc_loss_cfg.nce_t), labels)

        if get_world_size() > 1:
            dist.all_reduce(loss)
            dist.all_reduce(pos_sim)
            dist.all_reduce(neg_sim)
        return (
            loss / get_world_size(),
            pos_sim / get_world_size(),
            neg_sim / get_world_size(),
        )

    def training_step(self, batch, batch_idx):
        self._train_start = time.time()
        msc_loss_cfg = self.hparams.loss_cfg.msc_loss

        view1_origin_coord = batch["view1_origin_coord"]
        view1_coord = batch["view1_coord"]
        view1_feat = batch["view1_feat"]
        view1_offset = batch["view1_offset"].int()

        view2_origin_coord = batch["view2_origin_coord"]
        view2_coord = batch["view2_coord"]
        view2_feat = batch["view2_feat"]
        view2_offset = batch["view2_offset"].int()

        # mask generation by union original coord (without spatial augmentation)
        view1_point_mask, view2_point_mask = self.mask_generator(
            view1_origin_coord,
            view1_offset,
            view2_origin_coord,
            view2_offset,
            batch["caption_data"],
        )

        view1_mask_tokens = self.mask_token.expand(view1_coord.shape[0], -1)
        view1_weight = view1_point_mask.unsqueeze(-1).type_as(view1_mask_tokens)
        view1_feat = view1_feat * (1 - view1_weight) + view1_mask_tokens * view1_weight

        view2_mask_tokens = self.mask_token.expand(view2_coord.shape[0], -1)
        view2_weight = view2_point_mask.unsqueeze(-1).type_as(view2_mask_tokens)
        view2_feat = view2_feat * (1 - view2_weight) + view2_mask_tokens * view2_weight

        view1_batch = dict(
            origin_coord=view1_origin_coord,
            coord=view1_coord,
            grid_coord=batch["view1_grid_coord"],
            feat=view1_feat,
            offset=view1_offset,
            grid_size=batch["grid_size"],
        )
        view2_batch = dict(
            origin_coord=view2_origin_coord,
            coord=view2_coord,
            grid_coord=batch["view2_grid_coord"],
            feat=view2_feat,
            offset=view2_offset,
            grid_size=batch["grid_size"],
        )
        if "condition" in batch.keys():
            view1_batch["condition"] = batch["condition"]
            view2_batch["condition"] = batch["condition"]

        # view mixing strategy
        if random.random() < msc_loss_cfg.view1_mix_prob:
            view1_batch["offset"] = torch.cat(
                [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0
            )
        if random.random() < msc_loss_cfg.view2_mix_prob:
            view2_batch["offset"] = torch.cat(
                [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            )

        # Time forward pass
        self._forward_start = time.time()
        view1_out_dict = self(view1_batch)
        view1_feat = view1_out_dict["clip_feat"]
        view2_out_dict = self(view2_batch)
        view2_feat = view2_out_dict["clip_feat"]
        forward_time = time.time() - self._forward_start
        self.forward_time(forward_time)

        # Time loss computation
        self._loss_start = time.time()

        # self-supervised contrastive loss
        match_index = self.match_contrastive_pair(
            view1_origin_coord,
            view1_offset,
            view2_origin_coord,
            view2_offset,
            max_k=msc_loss_cfg.matching_max_k,
            max_radius=msc_loss_cfg.matching_max_radius,
        )
        nce_loss, pos_sim, neg_sim = self.compute_contrastive_loss(
            view1_feat, view1_offset, view2_feat, view2_offset, match_index
        )
        loss = nce_loss * self.hparams.loss_cfg.weights.nce_loss
        log_metrics = dict(nce_loss=nce_loss, pos_sim=pos_sim, neg_sim=neg_sim)

        if self.color_head is not None:
            assert "view1_color" in batch.keys()
            assert "view2_color" in batch.keys()
            view1_color = batch["view1_color"]
            view2_color = batch["view2_color"]
            view1_color_pred = self.color_head(view1_feat[view1_point_mask])
            view2_color_pred = self.color_head(view2_feat[view2_point_mask])
            color_loss = (
                torch.sum((view1_color_pred - view1_color[view1_point_mask]) ** 2)
                + torch.sum((view2_color_pred - view2_color[view2_point_mask]) ** 2)
            ) / (view1_color_pred.shape[0] + view2_color_pred.shape[0])
            loss = loss + color_loss * self.hparams.loss_cfg.weights.color_loss
            log_metrics["color_loss"] = color_loss

        if self.normal_head is not None:
            assert "view1_normal" in batch.keys()
            assert "view2_normal" in batch.keys()
            view1_normal = batch["view1_normal"]
            view2_normal = batch["view2_normal"]
            view1_normal_pred = self.normal_head(view1_feat[view1_point_mask])
            view2_normal_pred = self.normal_head(view2_feat[view2_point_mask])

            view1_normal_pred = view1_normal_pred / (
                torch.norm(view1_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            view2_normal_pred = view2_normal_pred / (
                torch.norm(view2_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            normal_loss = (
                torch.sum(view1_normal_pred * view1_normal[view1_point_mask])
                + torch.sum(view2_normal_pred * view2_normal[view2_point_mask])
            ) / (view1_normal_pred.shape[0] + view2_normal_pred.shape[0])
            loss = loss + normal_loss * self.hparams.loss_cfg.weights.normal_loss
            log_metrics["normal_loss"] = normal_loss

        # caption loss on view1
        caption_loss_kwargs = {
            "captions": batch["caption_data"].get("caption", None),
            "embeddings": batch["caption_data"].get("embedding", None),
            "point_indices": batch["caption_data"]["point_indices"],
            "caption_offsets": batch["caption_data"]["caption_offsets"],
            "num_points_per_caption": batch["caption_data"]["num_points_per_caption"],
            "clip_encoder": self.clip_encoder,
        }
        caption_loss = (
            self.caption_loss.loss(view1_feat, **caption_loss_kwargs)
            * self.hparams.loss_cfg.weights.caption_loss
        )
        log_metrics["caption_loss"] = caption_loss
        loss = loss + caption_loss

        log_metrics["loss"] = loss
        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        lr = self.optimizers().param_groups[0]["lr"]
        log_metrics["lr"] = lr

        # useful metadata
        bs = len(batch["view1_offset"]) - 1
        log_metrics["num_points"] = batch["view1_coord"].shape[0] / bs
        log_metrics["num_objects"] = (batch["caption_data"]["caption_offsets"].shape[0] - 1) / bs

        # Calculate training time and mark start of next data loading
        train_time = time.time() - self._train_start
        self.train_time(train_time)
        self._data_load_start = time.time()

        # Add timing metrics to existing logging
        log_metrics.update(
            {
                "time/data_loading": self.data_load_time.compute(),
                "time/forward": self.forward_time.compute(),
                "time/loss": self.loss_time.compute(),
                "time/training": self.train_time.compute(),
            }
        )

        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=self.train_sync_dist,
        )

        return loss
