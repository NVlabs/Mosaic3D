from typing import Any, Dict, List

import torch

import src.utils.caption_utils as caption_utils
from src.models.components.structure import Point
from src.models.lightning_modules.module_base import LitModuleBase
from src.models.components.clip_models import build_clip_model, download_clip_model
from src.models.losses.clip_alignment_loss import CLIPAlignmentLoss
from src.models.components.evaluator import InstanceSegmentationEvaluator
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class InstanceLanguageLitModule(LitModuleBase):
    def __init__(
        self,
        net,
        clip_encoder: Dict,
        compile: bool,
        loss_cfg: Dict,
        use_prompt: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        assert loss_cfg["seg_loss"]["eval_only"], "Segmentation loss must be eval only"
        self.clip_alignment_loss = CLIPAlignmentLoss(**loss_cfg["seg_loss"])

        self.evaluator = None

    def prepare_data(self) -> None:
        # download clip model on rank 0
        ckpt_path = download_clip_model(self.hparams.clip_encoder)
        log.info(f"Downloaded CLIP model to {ckpt_path}")

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        self.net = self.hparams.net()

        # clip encoder
        self.clip_encoder = build_clip_model(self.hparams.clip_encoder, device=self.device)

        # freeze net
        for params in self.net.parameters():
            params.requires_grad = False

        # freeze clip encoder
        for params in self.clip_encoder.parameters():
            params.requires_grad = False

    def setup(self, stage: str) -> None:
        val_dataloader = self.trainer.datamodule.val_dataloader()
        val_dataset = val_dataloader.dataset
        self.class_names = val_dataset.CLASS_LABELS

        ignore_idx = val_dataset.ignore_label
        segment_ignore_idx = val_dataset.ignore_class_idx.copy()
        segment_ignore_idx.append(ignore_idx)
        self.evaluator = InstanceSegmentationEvaluator(
            class_names=self.class_names,
            segment_ignore_index=segment_ignore_idx,
            instance_ignore_index=ignore_idx,
            subset_mapper=val_dataset.subset_mapper,
        )

    def forward(self, batch: Any) -> List:
        point = self.net(batch)
        list_out_dict = self._output_to_list(point, batch)  # mask embeddings
        return list_out_dict

    def _output_to_list(self, output: Any, batch: Any) -> List:
        # TODO: use overrides
        assert isinstance(output, Point)
        output: Point = output
        point_feat = output.sparse_conv_feat.features[output.v2p_map]

        offset = batch["offset"]
        batch_size = len(offset) - 1
        outputs = []
        for i in range(batch_size):
            pred_point_feat = point_feat[offset[i] : offset[i + 1]]
            gt_classes = batch["segment"][offset[i] : offset[i + 1]]
            gt_instances = batch["instance"][offset[i] : offset[i + 1]]
            pred_masks = batch["masks_binary"][i]

            # mask logits (voting)
            pred_point_logits = self.clip_alignment_loss.predict(
                pred_point_feat, return_logit=True
            )
            pred_point_logits = torch.nn.functional.softmax(pred_point_logits, dim=-1)
            logits = torch.stack([pred_point_logits[mask].mean(dim=0) for mask in pred_masks])

            pred_scores, pred_classes = torch.max(logits, dim=1)

            pred = dict(
                pred_classes=pred_classes.cpu().numpy(),
                pred_scores=pred_scores.cpu().numpy(),
                pred_masks=pred_masks.cpu().numpy(),
            )
            target = dict(
                segment=gt_classes.cpu().numpy(),
                instance=gt_instances.cpu().numpy(),
            )
            outputs.append((pred, target))

        return outputs

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_epoch_start(self):
        class_names = [c if c != "otherfurniture" else "other" for c in self.class_names]
        if self.hparams.use_prompt:
            class_names = [f"a {c} in a scene" if c != "other" else c for c in class_names]
        text_embedding = caption_utils.forward_text_encoder(
            class_names, self.clip_encoder.to(self.device), normalize=True
        )
        self.clip_alignment_loss.set_target_embedding(text_embedding)

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        for output in outputs:
            self.evaluator.update(*output)

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()

        # log metrics only if not sanity checking
        if not self.trainer.sanity_checking:
            for class_name, classwise_metrics in metrics["classes"].items():
                for metric_name, metric_value in classwise_metrics.items():
                    self.log(
                        f"val/{metric_name}_{class_name}",
                        metric_value,
                        sync_dist=True,
                        logger=True,
                    )
            metrics.pop("classes")
            self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True, logger=True)

        self.evaluator.reset()

    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
