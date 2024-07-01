import copy
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetTemplate(Dataset):
    def __init__(
        self,
        dataset_cfg=None,
        class_names=None,
        training=True,
        root_path=None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.root_path = (
            Path(root_path) if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        )

        if self.dataset_cfg is None or class_names is None:
            return

        self.class_names = class_names
        self.ignore_label = dataset_cfg.IGNORE_LABEL
        self.n_classes = len(self.class_names)

        # === open vocab ===
        self.class_mode = (
            "base" if (self.training and hasattr(dataset_cfg, "base_class_idx")) else "all"
        )

        # === open vocab ===
        self.valid_class_idx = np.arange(self.n_classes).tolist()

        if "base_class_idx" in dataset_cfg:
            self.base_class_mapper = self.build_class_mapper(
                dataset_cfg.base_class_idx, self.ignore_label
            )
            self.binary_class_mapper = self.build_binary_class_mapper(
                dataset_cfg.base_class_idx,
                dataset_cfg.novel_class_idx,
                dataset_cfg.ignore_class_idx,
                self.ignore_label,
            )
            self.base_class_idx = dataset_cfg.base_class_idx
            self.novel_class_idx = dataset_cfg.novel_class_idx

        if "ignore_class_idx" in dataset_cfg:
            for c in dataset_cfg.ignore_class_idx:
                self.valid_class_idx.remove(c)
            self.ignore_class_idx = dataset_cfg.ignore_class_idx

        self.valid_class_mapper = self.build_class_mapper(
            self.valid_class_idx, self.ignore_label, squeeze_label=self.training
        )

        # caption config
        if self.training and "CAPTION_INFO" in self.dataset_cfg:
            self.caption_cfg = self.dataset_cfg.CAPTION_INFO
            self.caption_keys = self.dataset_cfg.CAPTION_INFO.KEY
            self.caption = self.get_caption_items(self.caption_cfg)

        self.need_view_caption = (
            hasattr(self, "caption_cfg")
            and self.caption_cfg.get("VIEW", None)
            and self.caption_cfg.VIEW.ENABLED
        )

        # KD keywords
        if self.training:
            self.load_kd_label = self.dataset_cfg.get("LOAD_KD_LABEL_TRAIN", None)
        else:
            self.load_kd_label = self.dataset_cfg.get("LOAD_KD_LABEL_TEST", None)
        self.kd_label_dir = self.dataset_cfg.get("KD_LABEL_DIR", None)
        self.kd_label_norm = self.dataset_cfg.get("KD_LABEL_NORM", False)

        self.mode = "train" if self.training else "test"

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def build_class_mapper(class_idx, ignore_idx, squeeze_label=True):
        remapper = np.ones(256, dtype=np.int64) * ignore_idx
        for i, x in enumerate(class_idx):
            if squeeze_label:
                remapper[x] = i
            else:
                remapper[x] = x
        return remapper

    @staticmethod
    def build_binary_class_mapper(base_class_idx, novel_class_idx, ignore_class_idx, ignore_idx):
        remapper = np.ones(256, dtype=np.int64) * ignore_idx  # base: 1, novel: 0
        for _, x in enumerate(base_class_idx):
            remapper[x] = 1
        for _, x in enumerate(novel_class_idx):
            remapper[x] = 0
        # ignored categories are mapped to novel
        for _, x in enumerate(ignore_class_idx):
            remapper[x] = 0
        return remapper

    def collate_batch_indoor(self, batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        total_inst_num = 0
        for key, val in data_dict.items():
            if key in ["points"]:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ["ids", "pc_count", "batch_size", "inst_num"]:
                ret[key] = data_dict[key]
            elif key in [
                "points_xyz",
                "feats",
                "labels",
                "binary_labels",
                "origin_idx",
                "rgb",
                "pt_offset_label",
                "inst_info",
                "inst_pointnum",
                "kd_labels",
                "pred_mask",
                "kd_labels_mask",
                "adapter_feats",
                "adapter_feats_mask",
                "super_voxel",
                "pt_offset_mask",
                "n_captions_points",
            ]:
                ret[key] = np.concatenate(data_dict[key], axis=0)
            elif key in ["inst_label"]:
                if "inst_num" in data_dict:
                    inst_labels = []
                    for i, il in enumerate(val):
                        il[np.where(il != self.ignore_label)] += total_inst_num
                        total_inst_num += data_dict["inst_num"][i]
                        inst_labels.append(il)
                else:
                    inst_labels = val
                ret[key] = np.concatenate(inst_labels, axis=0)
            elif key in ["points_xyz_voxel_scale"]:
                if data_dict[key][0].shape[1] == 4:  # x4_split
                    assert len(data_dict[key]) == 1
                    ret[key] = np.concatenate(data_dict[key], axis=0)
                    batch_size = int(ret[key][..., 0].max() + 1)  # re-set batch size
                else:
                    ret[key] = np.concatenate(
                        [
                            np.concatenate(
                                [np.full((d.shape[0], 1), i), d.astype(np.int64)],
                                axis=-1,
                            )
                            for i, d in enumerate(data_dict[key])
                        ],
                        axis=0,
                    )
            elif key in ["caption_data"]:
                # ret[key] = tuple(zip(*data_dict[key]))
                if val[0] is None:
                    continue
                ret[key] = {}
                for k in val[0]:
                    ret[key][k] = {}
                    ret[key][k]["idx"] = [val[n][k]["idx"] for n in range(len(val))]
                    ret[key][k]["caption"] = []
                    for n in range(len(val)):
                        ret[key][k]["caption"].extend(val[n][k]["caption"])
            elif key in ["inst_cls"]:
                ret[key] = np.array([j for i in data_dict[key] for j in i], dtype=np.int32)
            else:
                ret[key] = np.stack(val, axis=0)

        ret["spatial_shape"] = np.clip(
            (ret["points_xyz_voxel_scale"].max(0)[1:] + 1),
            self.dataset_cfg.MIN_SPATIAL_SCALE,
            None,
        )

        ret["batch_idxs"] = ret["points_xyz_voxel_scale"][:, 0].astype(np.int32)
        if len(batch_list) == 1:
            ret["offsets"] = np.array([0, ret["batch_idxs"].shape[0]]).astype(np.int32)
        else:
            ret["offsets"] = np.cumsum(np.bincount(ret["batch_idxs"] + 1).astype(np.int32))
            assert len(ret["offsets"]) == batch_size + 1

        ret["batch_size"] = batch_size
        return ret

    def collate_batch_lai(self, batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        # shift idx_recon to support batch wise process
        accumulate_points_num = 0
        total_inst_num = 0
        for i in range(batch_size):
            data_dict["v2p_map"][i] = accumulate_points_num + data_dict["v2p_map"][i]
            accumulate_points_num += data_dict["voxel_coords"][i].shape[0]

        for key, val in data_dict.items():
            if key in ["points"]:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ["voxel_features", "v2p_map"]:
                ret[key] = torch.cat(data_dict[key])
            elif key in [
                "labels",
                "kd_labels",
                "origin_idx",
                "binary_labels",
                "rgb",
                "super_voxel",
                "pred_mask",
                "pt_offset_label",
                "points_xyz",
                "inst_info",
                "inst_pointnum",
                "pt_offset_mask",
            ]:
                ret[key] = np.concatenate(data_dict[key], axis=0)
            elif key in ["inst_label"]:
                if "inst_num" in data_dict:
                    inst_labels = []
                    for i, il in enumerate(val):
                        il[np.where(il != self.ignore_label)] += total_inst_num
                        total_inst_num += data_dict["inst_num"][i]
                        inst_labels.append(il)
                else:
                    inst_labels = val
                ret[key] = np.concatenate(inst_labels, axis=0)
            elif key in ["image_corr_tuple", "image_name_tuple"]:
                ret[key] = tuple(zip(*data_dict[key]))
            elif key in ["ids", "scene_name", "pc_count", "batch_size"]:
                ret[key] = data_dict[key]
            elif key in ["voxel_coords"]:
                if val[0].shape[1] == 4:  # x4_split
                    assert len(val) == 1
                    ret[key] = torch.cat(val)
                    batch_size = int(ret[key][..., 0].max() + 1)  # re-set batch size
                else:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = torch.nn.functional.pad(coor, (1, 0), mode="constant", value=i)
                        coors.append(coor_pad)
                    ret[key] = torch.cat(coors)
            elif key in ["caption_data"]:
                if val[0] is None:
                    continue
                ret[key] = {}
                for k in val[0]:
                    ret[key][k] = {}
                    ret[key][k]["idx"] = [val[n][k]["idx"] for n in range(len(val))]
                    ret[key][k]["caption"] = []
                    for n in range(len(val)):
                        ret[key][k]["caption"].extend(val[n][k]["caption"])
            elif key in ["inst_cls"]:
                ret[key] = np.array([j for i in data_dict[key] for j in i], dtype=np.int32)
            else:
                try:
                    ret[key] = np.stack(val, axis=0)
                except Exception as e:
                    print(e)

        ret["batch_idxs"] = ret["points"][:, 0].astype(np.int32)
        # ret['offsets'] = np.cumsum(np.bincount(ret['batch_idxs'] + 1).astype(np.int32))
        # assert len(ret['offsets']) == batch_size + 1
        if len(batch_list) == 1:
            ret["offsets"] = np.array([0, ret["batch_idxs"].shape[0]]).astype(np.int32)
        else:
            ret["offsets"] = np.cumsum(np.bincount(ret["batch_idxs"] + 1).astype(np.int32))
            assert len(ret["offsets"]) == batch_size + 1

        ret["batch_size"] = batch_size

        # As a data augmentation, follow Lai Xin's code
        if self.training:
            ret["voxel_coords"][:, 1:] += (torch.rand(3) * 2).type_as(ret["voxel_coords"])

        ret["spatial_shape"] = np.clip(
            (ret["voxel_coords"].max(0)[0][1:] + 1).numpy(),
            self.dataset_cfg.MIN_SPATIAL_SCALE,
            None,
        )
        return ret

    def get_caption_items(self, caption_cfg):
        caption_items = {}
        data_path = self.dataset_cfg.DATA_PATH
        for key in caption_cfg:
            if key in self.caption_keys and caption_cfg[key].ENABLED:
                caption_path = os.path.join(data_path, caption_cfg[key].CAPTION_PATH)
                caption_items[key.lower()] = copy.deepcopy(json.load(open(caption_path)))
        return caption_items

    def select_caption_and_idx_all(self, scene_name, image_name_dict, image_corr_dict):
        if not hasattr(self, "caption_cfg"):
            return None

        ret = {}
        for key in self.caption_cfg:
            if key in self.caption_keys and self.caption_cfg[key].ENABLED:
                key_lower = key.lower()
                ret[key_lower] = self.select_caption_and_idx(
                    self.caption[key_lower],
                    self.caption_cfg[key],
                    scene_name,
                    image_name_dict[key_lower],
                    image_corr_dict[key_lower],
                )
        return ret

    @staticmethod
    def select_caption_and_idx(caption, caption_cfg, scene_name, image_names, image_corr_indices):
        if image_corr_indices is None:
            select_captions = [caption[scene_name]]
            select_image_corr = [None]
        else:
            assert len(caption[scene_name]) == len(image_names)
            select_image_names, select_image_corr = DatasetTemplate.select_images(
                caption_cfg, image_names, image_corr_indices
            )  # list (B, K), (B, K, N)
            # (B*K)
            select_captions = [caption[scene_name][n] for n in select_image_names]
        return {"idx": select_image_corr, "caption": select_captions}

    @staticmethod
    def select_images(caption_cfg, image_name, image_corr):
        if caption_cfg.RATIO == 1.0:
            return image_name, image_corr

        if image_name is None or len(image_name) == 0:  # lack 2d data
            selected_idx = None
        else:
            ratio = caption_cfg.RATIO
            selected_idx = np.random.choice(
                len(image_name), max(1, int(len(image_name) * ratio)), replace=False
            )

        if selected_idx is not None:
            selected_image_name = np.array(image_name)[selected_idx].tolist()
            selected_image_corr = np.array(image_corr, dtype=object)[selected_idx].tolist()
        else:
            selected_image_name = []
            selected_image_corr = []

        return selected_image_name, selected_image_corr

    def include_point_caption_idx(self):
        if self.need_view_caption and self.caption_cfg.VIEW.get("IMAGE_CORR_PATH", None):
            corr_path = self.caption_cfg.VIEW.IMAGE_CORR_PATH
            corr_path = self.root_path / corr_path
            point_caption_idx = pickle.load(open(corr_path, "rb"))
        else:
            point_caption_idx = None

        entity_point_caption_idx = None
        return point_caption_idx, entity_point_caption_idx
