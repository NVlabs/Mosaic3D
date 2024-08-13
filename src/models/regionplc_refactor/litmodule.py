import functools
from typing import List

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule

import src.models.regionplc.utils.caption_utils as caption_utils
from src.models.components.structure import Point
from src.models.regionplc.text_models import build_text_model
from src.models.regionplc_refactor.backbone import SparseUNet
from src.models.regionplc_refactor.head import BinaryHead, CaptionHead, TextSegHead
from src.models.regionplc_refactor.modules import MLP
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class RegionPLCLitModule(LightningModule):
    def __init__(
        self,
        # in_channel: int,
        # mid_channel: int,
        # block_reps: int,
        # block_residual: bool,
        # custom_sp1x1: bool,
        # num_blocks: int,
        # text_channel: int,
        # adapter_num_layers: int,
        # adapter_last_norm: bool,
        # backbone
        backbone_cfg,
        # binary_head
        binary_head_cfg,
        # enable_binary_head: bool,
        # binary_thresh: float,
        # ignore_label: int,
        # hook_feature_list: List[str],
        # detach: bool,
        # loss_weight: float,
        # voxel_loss: bool,
        # text seg head
        task_head_cfg,
        # text_embed_path: str,
        # feat_norm: bool,
        # text_seg_loss_weight: float,
        # logit_scale: float,
        # logit_learable: bool,
        # eval_only: bool,
        # caption head
        caption_head_cfg,
        # feat_norm: bool,
        # logit_scale: float,
        # logit_learnable: float,
        # loss_weight: float,
        # novel_grad_only: bool,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        # network
        if self.backbone is not None:
            return

        self.backbone = SparseUNet(**self.hparams.backbone_cfg)

        mid_channel, text_channel = (
            self.hparams.backbone_cfg.mid_channel,
            self.hparams.backbone_cfg.text_channel,
        )
        adapter_channel_list = [mid_channel, text_channel]
        if self.hparams.adapter_cfg.num_layers == 2:
            multiplier = int(np.log2(text_channel / mid_channel))
            adapter_channel_list = [mid_channel, mid_channel * multiplier, text_channel]
        self.adapter = MLP(
            adapter_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            **self.hparams.adapter_cfg,
        )

        self.binary_head = None
        if self.hparams.binary_head_cfg.enable_binary_head:
            self.binary_head = BinaryHead(**self.hparams.binary_head_cfg)
            self.binary_head.register_hook_for_binary_head(self.backbone)

        self.task_head = TextSegHead(**self.hparams.task_head_cfg)
        self.caption_head = CaptionHead(**self.hparams.caption_head_cfg)

        # text encoder
        self.text_encoder = build_text_model(self.hparams.text_encoder)

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        dataset = self.trainer.train_dataloader.dataset
        batch = caption_utils.get_caption_batch(
            dataset.caption_cfg,
            {},
            batch,
            self.text_encoder,
            local_rank=self.local_rank,
        )

        point = Point(batch)
        point.sparsify(pad=128)

        sparse_tensor = point.sparse_conv_feat
        backbone_feat = self.backbone(sparse_tensor)
        adapter_feat = self.adapter(backbone_feat)

        binary_head_output = None
        if self.binary_head is not None:
            binary_head_output = self.binary_head(point.v2p_map)

        task_head_output = self.task_head(batch, adapter_feat)
        caption_head_output = self.caption_head(
            batch, point, batch["caption_infos"], point.v2p_map, adapter_feat
        )

        # loss
        binary_loss = self.binary_head.get_loss(
            binary_head_output["binary_scores"], batch["binary_labels"]
        )
        seg_loss = self.task_head.get_loss(task_head_output["seg_scores"], batch["segment"])
        caption_loss = self.caption_head.get_loss(caption_head_output)
        loss = binary_loss + seg_loss + caption_loss

        log_metrics = dict(
            loss=loss,
            binary_loss=binary_loss,
            caption_loss=caption_loss,
            seg_loss=seg_loss,
        )
        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
