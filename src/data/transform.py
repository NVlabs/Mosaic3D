import copy
import numbers
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import List, Optional, Literal
import re

import numpy as np
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.stats
import torch
from jaxtyping import Int
from torch import Tensor

from src.data.caption_transform import CaptionFilter
from src.utils.registry import Registry

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class Collect:
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """e.g. Collect(keys=[coord], feat_keys=[coord, color])"""
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            if key in data_dict.keys():
                data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy:
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class ToTensor:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class Add:
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor:
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord:
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift:
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class MeanShift:
    def __init__(self):
        pass

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift = data_dict["coord"].mean(axis=0)
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class CenterShift:
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class RandomShift:
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip:
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout:
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate:
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle:
    def __init__(self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale:
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter:
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter:
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast:
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation:
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter:
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(noise + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale:
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter:
    """Random Color Jitter for 3D point cloud (refer torchvision)"""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (ratio * color1 + (1.0 - ratio) * color2).clip(0, bound).astype(color1.dtype)

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = None if brightness is None else np.random.uniform(brightness[0], brightness[1])
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = None if saturation is None else np.random.uniform(saturation[0], saturation[1])
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_brightness(data_dict["color"], brightness_factor)
            elif fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_contrast(data_dict["color"], contrast_factor)
            elif fn_id == 2 and saturation_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_saturation(data_dict["color"], saturation_factor)
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation:
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop:
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return f"RandomColorDrop(color_augment: {self.color_augment}, p: {self.p})"


@TRANSFORMS.register_module()
class ElasticDistortion:
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


class BasePointCloudTransform:
    """
    Base class for point cloud transformations that handle data indexing and remapping.

    This class provides common functionality for point cloud operations that need to:
    - Apply indexing to multiple data fields
    - Remap caption data indices after point selection
    - Handle clip point indices
    - Filter empty captions

    All point cloud transforms that modify point indices should inherit from this class
    to ensure consistent data handling across different fields.
    """

    def _apply_point_indexing(self, data_dict, idx_selected, keys=None):
        """
        Apply point indexing to data_dict based on selected indices.

        This method applies the same index selection to multiple point cloud data fields,
        ensuring all per-point data remains synchronized after point selection operations.

        Args:
            data_dict (dict): Dictionary containing point cloud data with per-point arrays
            idx_selected (np.ndarray): 1D array of indices to keep (shape: [N])
            keys (tuple, optional): Tuple of keys to apply indexing to. If None,
                applies to common point cloud keys: coord, color, normal, segment,
                binary, instance, displacement, strength, origin_coord, grid_coord, origin_idx

        Returns:
            dict: Modified data_dict with indexed data. Only the specified keys are modified.

        Example:
            >>> data_dict = {"coord": np.random.rand(1000, 3), "color": np.random.rand(1000, 3)}
            >>> indices = np.array([0, 10, 20, 30])  # Select 4 points
            >>> data_dict = self._apply_point_indexing(data_dict, indices)
            >>> data_dict["coord"].shape  # (4, 3)
        """
        if keys is None:
            # Default keys for point cloud data
            keys = (
                "coord",
                "color",
                "normal",
                "segment",
                "binary",
                "instance",
                "displacement",
                "strength",
                "origin_coord",
                "grid_coord",
                "origin_idx",
            )

        # Apply indexing to all specified keys that exist in data_dict
        for key in keys:
            if key in data_dict:
                data_dict[key] = data_dict[key][idx_selected]

        return data_dict

    def _remap_caption_data(self, data_dict, idx_selected, original_size):
        """
        Remap caption data indices after point sampling/cropping.

        When points are selected from a point cloud, any associated caption data that
        references point indices must be updated to reflect the new indexing. This method
        handles the complex task of remapping caption indices and filtering out captions
        that no longer have associated points.

        Args:
            data_dict (dict): Dictionary containing point cloud data with caption_data field
            idx_selected (np.ndarray): Array of indices that were selected (shape: [N])
            original_size (int): Original number of points before selection

        Returns:
            dict: Modified data_dict with updated caption_data indices

        Note:
            - Creates a mapping from original indices to new indices
            - Filters out captions that have no points after selection
            - Preserves both "caption" and "embedding" data types
            - Handles both list and tensor indices
        """
        if "caption_data" not in data_dict:
            return data_dict

        caption_dict = data_dict["caption_data"]
        target_key = "caption" if "caption" in caption_dict else "embedding"
        assert target_key in caption_dict
        captions_or_embeddings = caption_dict[target_key]
        # List of point indices for each caption
        caption_point_indices: List[Int[Tensor, "*"]] = caption_dict["idx"]  # noqa: F722
        assert len(captions_or_embeddings) == len(caption_point_indices)

        # Create mapping from old to new indices
        new_index = np.arange(len(idx_selected))
        to_new_index = np.ones(original_size, dtype=int) * -1
        to_new_index[idx_selected] = new_index

        # Remap caption indices
        new_caption_index = [
            to_new_index[point_indices] for point_indices in caption_point_indices
        ]
        # Remove -1 index (points that were filtered out)
        new_caption_index = [
            point_indices[point_indices != -1] for point_indices in new_caption_index
        ]
        # Get indices of captions that still have points
        valid_caption_indices = [
            i for i, point_indices in enumerate(new_caption_index) if len(point_indices) > 0
        ]
        # Filter out empty arrays
        new_caption_index = [new_caption_index[i] for i in valid_caption_indices]
        captions_or_embeddings = [captions_or_embeddings[i] for i in valid_caption_indices]

        data_dict["caption_data"] = {
            target_key: captions_or_embeddings,
            "idx": new_caption_index,
        }

        return data_dict

    def _remap_clip_point_indices(self, data_dict, idx_selected, original_size):
        """
        Remap clip point indices after point sampling/cropping.

        Clip point indices are special indices that mark important points that should
        be preserved during transformations. This method updates these indices to
        reflect the new point ordering after selection operations.

        Args:
            data_dict (dict): Dictionary containing point cloud data with clip_point_indices field
            idx_selected (np.ndarray): Array of indices that were selected (shape: [N])
            original_size (int): Original number of points before selection

        Returns:
            dict: Modified data_dict with updated clip_point_indices

        Note:
            - Uses torch tensors for index operations
            - Filters out clip indices that are no longer present
            - Concatenates multiple clip point groups into a single tensor
        """
        if "clip_point_indices" not in data_dict:
            return data_dict

        clip_point_indices = data_dict["clip_point_indices"]
        new_index = torch.arange(len(idx_selected))
        to_new_index = torch.full((original_size,), -1, dtype=torch.long)
        to_new_index[idx_selected] = new_index
        new_clip_point_indices = [
            to_new_index[clip_point_index] for clip_point_index in clip_point_indices
        ]
        # Remove -1 index
        new_clip_point_indices = [
            new_clip_point_index[new_clip_point_index != -1]
            for new_clip_point_index in new_clip_point_indices
        ]
        new_clip_point_indices = torch.cat(new_clip_point_indices)
        data_dict["clip_point_indices"] = new_clip_point_indices

        return data_dict

    def _filter_empty_captions(self, data_dict):
        """
        Filter out captions that have no associated points.

        After point selection operations, some captions may end up with no associated
        points. This method removes such empty captions to maintain data consistency
        and prevent errors in downstream processing.

        Args:
            data_dict (dict): Dictionary containing point cloud data with caption_data field

        Returns:
            dict: Modified data_dict with empty captions removed

        Note:
            - Validates that all remaining captions have at least one point
            - Preserves the order of remaining captions
            - Works with both "caption" and "embedding" data types

        Raises:
            AssertionError: If any caption ends up with no points after filtering
        """
        if "caption_data" not in data_dict:
            return data_dict

        caption_dict = data_dict["caption_data"]
        target_key = "caption" if "caption" in caption_dict else "embedding"
        assert target_key in caption_dict
        captions_or_embeddings = caption_dict[target_key]
        # List of point indices for each caption
        caption_point_indices: List[Int[Tensor, "*"]] = caption_dict["idx"]  # noqa: F722

        # Filter out captions that have no points
        valid_caption_indices = [
            i for i, point_indices in enumerate(caption_point_indices) if len(point_indices) > 0
        ]
        # Filter out captions that have no points
        if len(valid_caption_indices) != len(captions_or_embeddings):
            caption_point_indices = [caption_point_indices[i] for i in valid_caption_indices]
            captions_or_embeddings = [captions_or_embeddings[i] for i in valid_caption_indices]
        # Assert that all captions have points
        assert all(len(point_indices) > 0 for point_indices in caption_point_indices)
        data_dict["caption_data"] = {
            target_key: captions_or_embeddings,
            "idx": caption_point_indices,
        }

        return data_dict


@TRANSFORMS.register_module()
class GridSample(BasePointCloudTransform):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        original_size = len(data_dict["coord"])
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]

            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(np.append(idx_unique, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]

            # Use base class methods for data remapping
            data_dict = self._remap_caption_data(data_dict, idx_unique, original_size)

            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]

            # Apply point indexing using base class method
            data_dict = self._apply_point_indexing(data_dict, idx_unique, self.keys)
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """Ravel the coordinates after subtracting the min coordinates."""
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
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


class BaseCrop(BasePointCloudTransform):
    """
    Base class for point cloud cropping operations with optional voxel downsampling.

    This class provides a flexible framework for point cloud cropping that combines:
    1. Optional voxel-based downsampling for efficiency
    2. Spatial cropping (spherical, rectangular, etc.)
    3. Sample rate and point count constraints

    The processing pipeline follows this order:
    1. Calculate target number of points from original size (sample_rate/point_max takes precedence)
    2. Apply voxel downsampling if voxel_size is specified (preprocessing for efficiency)
    3. Apply spatial cropping to reach the target point count
    4. Apply final data transformation with proper index remapping

    Parameters:
        point_max (int): Maximum number of points to keep (default: 80000)
        sample_rate (float, optional): Fraction of original points to keep (takes precedence over voxel_size)
        mode (str): How to select center point ("random", "center", "captioned")
        voxel_size (float, optional): Voxel size for grid downsampling preprocessing
        hash_type (str): Hash function type for voxelization ("fnv" or "ravel")
    """

    def __init__(
        self,
        point_max=80000,
        sample_rate=None,
        mode: Literal["random", "center", "captioned"] = "random",
        voxel_size=None,
        hash_type="fnv",
    ):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "captioned"]
        self.mode = mode
        self.voxel_size = voxel_size
        # Set up hash function for grid downsampling
        if voxel_size is not None:
            self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec

    def _get_point_max(self, original_num_points):
        """
        Calculate the maximum number of points to keep based on original point cloud size.

        This method implements the precedence rule where sample_rate and point_max are calculated
        from the original point cloud size, giving users predictable control over the final
        output regardless of any preprocessing steps like voxel downsampling.

        Args:
            original_num_points (int): Number of points in the original point cloud (before any processing)

        Returns:
            int: Maximum number of points to keep

        Examples:
            >>> # With sample_rate only
            >>> crop = BaseCrop(sample_rate=0.1)
            >>> crop._get_point_max(100000)  # Returns 10000 (10% of 100K)

            >>> # With both sample_rate and point_max
            >>> crop = BaseCrop(sample_rate=0.2, point_max=15000)
            >>> crop._get_point_max(100000)  # Returns 15000 (min of 20K and 15K)
        """
        if self.sample_rate is not None:
            sample_based_max = int(self.sample_rate * original_num_points)
            # If both sample_rate and point_max are specified, take the minimum
            return (
                min(sample_based_max, self.point_max)
                if self.point_max is not None
                else sample_based_max
            )
        else:
            return self.point_max

    def _get_center_point(self, data_dict):
        """Get the center point for cropping based on mode."""
        if self.mode == "random":
            return data_dict["coord"][np.random.randint(data_dict["coord"].shape[0])]
        elif self.mode == "center":
            return data_dict["coord"][data_dict["coord"].shape[0] // 2]
        elif self.mode == "captioned":
            assert "caption_data" in data_dict, "Caption data is required for captioned mode"
            point_indices = data_dict["caption_data"]["idx"]
            sel_point_indices = np.random.randint(len(point_indices))
            random_idx = np.random.choice(point_indices[sel_point_indices])
            return data_dict["coord"][random_idx]
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported")

    def _crop_data_dict(self, data_dict, idx_crop):
        """
        Apply cropping to data_dict based on idx_crop indices.

        Args:
            data_dict: Dictionary containing point cloud data
            idx_crop: Indices to keep after cropping

        Returns:
            Cropped data_dict
        """
        original_size = data_dict["coord"].shape[0]

        # Use base class methods for data remapping and indexing
        data_dict = self._apply_point_indexing(data_dict, idx_crop)
        data_dict = self._remap_caption_data(data_dict, idx_crop, original_size)
        data_dict = self._remap_clip_point_indices(data_dict, idx_crop, original_size)
        data_dict = self._filter_empty_captions(data_dict)

        return data_dict

    def _grid_downsample(self, data_dict):
        """
        Apply grid downsampling to the point cloud for computational efficiency.

        This method performs voxel-based downsampling where the point cloud is divided
        into a regular 3D grid and one representative point is randomly selected from
        each occupied voxel. This preprocessing step can significantly reduce
        computational cost for subsequent spatial operations.

        Args:
            data_dict (dict): Dictionary containing point cloud data with "coord" field

        Returns:
            tuple: (downsampled_indices, original_data_dict)
                - downsampled_indices (np.ndarray): Indices of selected points (shape: [M])
                - original_data_dict (dict): Unchanged input data_dict

        Note:
            - If voxel_size is None, returns all point indices (no downsampling)
            - Uses random selection within each voxel for diversity
            - Coordinate scaling: coord / voxel_size
            - Grid origin is shifted to ensure non-negative indices

        Examples:
            >>> # With voxel_size specified
            >>> crop = BaseCrop(voxel_size=0.05)
            >>> indices, data = crop._grid_downsample({"coord": points})
            >>> len(indices) < len(points)  # True - points reduced

            >>> # Without voxel_size
            >>> crop = BaseCrop()
            >>> indices, data = crop._grid_downsample({"coord": points})
            >>> len(indices) == len(points)  # True - no downsampling
        """
        if self.voxel_size is None:
            # No downsampling, return all indices
            return np.arange(len(data_dict["coord"])), data_dict

        scaled_coord = data_dict["coord"] / np.array(self.voxel_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord

        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # Use train mode logic - randomly select one point per voxel
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]

        return idx_unique, data_dict

    def _apply_clip_point_indices(self, idx_crop, data_dict):
        """Apply clip point indices if present in data_dict."""
        if "clip_point_indices" in data_dict.keys():
            clip_point_indices = data_dict["clip_point_indices"]
            num_replace = len(clip_point_indices)
            idx_crop[-num_replace:] = clip_point_indices.numpy()
        return idx_crop

    @staticmethod
    def ravel_hash_vec(arr):
        """Ravel the coordinates after subtracting the min coordinates."""
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
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

    def __call__(self, data_dict):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement __call__ method")


@TRANSFORMS.register_module()
class SphereCrop(BaseCrop):
    """
    Spherical cropping around a center point with optional voxel downsampling.

    This transform selects points within a sphere around a chosen center point.
    The center can be selected randomly, from the middle of the point cloud, or
    from points associated with captions. Optionally applies voxel downsampling
    first for computational efficiency.

    The spherical selection is based on Euclidean distance from the center point,
    choosing the closest points up to the specified limit.

    Args:
        point_max (int): Maximum number of points to keep (default: 80000)
        sample_rate (float, optional): Fraction of original points to keep
        mode (str): Center selection method:
            - "random": Random point as center
            - "center": Middle point of sorted coordinates as center
            - "captioned": Random point from caption annotations as center
        voxel_size (float, optional): Voxel size for preprocessing downsampling
        hash_type (str): Hash function for voxelization ("fnv" or "ravel")

    Examples:
        >>> # Basic spherical cropping
        >>> transform = SphereCrop(point_max=50000, mode="random")

        >>> # With voxel downsampling + spherical cropping
        >>> transform = SphereCrop(
        ...     sample_rate=0.1,
        ...     voxel_size=0.05,
        ...     mode="center"
        ... )

        >>> # Caption-guided spherical cropping
        >>> transform = SphereCrop(point_max=80000, mode="captioned")
    """

    def __init__(
        self,
        point_max=80000,
        sample_rate=None,
        mode: Literal["random", "center", "captioned"] = "random",
        voxel_size=None,
        hash_type="fnv",
    ):
        super().__init__(point_max, sample_rate, mode, voxel_size, hash_type)

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        original_size = len(data_dict["coord"])

        # Step 1: Calculate target number of points based on original size
        # sample_rate and point_max take precedence over voxel_size
        target_point_max = self._get_point_max(original_size)

        # Step 2: Apply grid downsampling if voxel_size is specified
        idx_downsample, data_dict = self._grid_downsample(data_dict)

        # Step 3: Apply spherical cropping to reach target_point_max
        if len(idx_downsample) > target_point_max:
            # Get center point using downsampled coordinates
            downsampled_coords = data_dict["coord"][idx_downsample]
            center = self._get_center_point_from_coords(
                downsampled_coords, data_dict, idx_downsample
            )

            # Find closest points to center among downsampled points
            distances = np.sum(np.square(downsampled_coords - center), 1)
            idx_crop_local = np.argsort(distances)[:target_point_max]

            # Map back to original indices
            idx_final = idx_downsample[idx_crop_local]
        else:
            # Use all downsampled points (fewer than target)
            idx_final = idx_downsample

        # Apply clip point indices if present
        idx_final = self._apply_clip_point_indices(idx_final, data_dict)

        # Apply final cropping using combined indices
        data_dict = self._crop_data_dict(data_dict, idx_final)

        return data_dict

    def _get_center_point_from_coords(self, coords, data_dict, coord_indices):
        """Get center point for cropping from given coordinates."""
        if self.mode == "random":
            return coords[np.random.randint(len(coords))]
        elif self.mode == "center":
            return coords[len(coords) // 2]
        elif self.mode == "captioned":
            assert "caption_data" in data_dict, "Caption data is required for captioned mode"
            point_indices = data_dict["caption_data"]["idx"]
            sel_point_indices = np.random.randint(len(point_indices))
            caption_point_idx = np.random.choice(point_indices[sel_point_indices])

            # Find the index in coord_indices that matches caption_point_idx
            coord_mask = np.isin(coord_indices, caption_point_idx)
            if coord_mask.any():
                local_idx = np.where(coord_mask)[0][0]
                return coords[local_idx]
            else:
                # Fallback to random if caption point not in downsampled set
                return coords[np.random.randint(len(coords))]
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported")


@TRANSFORMS.register_module()
class RectCrop(BaseCrop):
    """
    Rectangular/cubic cropping around a center point with optional voxel downsampling.

    This transform selects points within a rectangular bounding box around a chosen
    center point. The bounding box can have uniform or different dimensions for each
    axis. If the rectangular region contains fewer points than requested, falls back
    to distance-based selection.

    Args:
        point_max (int): Maximum number of points to keep (default: 80000)
        sample_rate (float, optional): Fraction of original points to keep
        size (float or tuple, optional): Size of rectangular crop region
            - float: Uniform size for all dimensions (creates a cube)
            - tuple: (x_size, y_size, z_size) for different dimensions
            - None: Falls back to distance-based selection
        mode (str): Center selection method:
            - "random": Random point as center
            - "center": Middle point of sorted coordinates as center
            - "captioned": Random point from caption annotations as center
        voxel_size (float, optional): Voxel size for preprocessing downsampling
        hash_type (str): Hash function for voxelization ("fnv" or "ravel")

    Examples:
        >>> # Cubic cropping
        >>> transform = RectCrop(point_max=50000, size=10.0, mode="center")

        >>> # Non-uniform rectangular cropping
        >>> transform = RectCrop(
        ...     sample_rate=0.15,
        ...     size=(20, 20, 5),  # 20x20x5 box
        ...     mode="random"
        ... )

        >>> # With voxel preprocessing
        >>> transform = RectCrop(
        ...     point_max=80000,
        ...     size=15.0,
        ...     voxel_size=0.02,
        ...     mode="captioned"
        ... )
    """

    def __init__(
        self,
        point_max=80000,
        sample_rate=None,
        size=None,
        mode: Literal["random", "center", "captioned"] = "random",
        voxel_size=None,
        hash_type="fnv",
    ):
        super().__init__(point_max, sample_rate, mode, voxel_size, hash_type)
        self.size = size  # Size of the rectangular crop (can be tuple for different dimensions)

    def _get_rectangular_indices(self, coords, center, size, point_max):
        """Get indices for points within rectangular bounds."""
        if size is None:
            # Fallback to taking closest points if no size specified
            distances = np.sum(np.square(coords - center), 1)
            return np.argsort(distances)[:point_max]

        # Convert size to array for broadcasting
        if isinstance(size, (int, float)):
            size = np.array([size, size, size])
        else:
            size = np.array(size)

        # Find points within rectangular bounds
        diff = np.abs(coords - center)
        within_bounds = np.all(diff <= size / 2, axis=1)
        valid_indices = np.where(within_bounds)[0]

        # If we have more points than needed, randomly sample
        if len(valid_indices) > point_max:
            valid_indices = np.random.choice(valid_indices, point_max, replace=False)
        # If we don't have enough points, fall back to distance-based selection
        elif len(valid_indices) < point_max:
            distances = np.sum(np.square(coords - center), 1)
            valid_indices = np.argsort(distances)[:point_max]

        return valid_indices

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        original_size = len(data_dict["coord"])

        # Step 1: Calculate target number of points based on original size
        # sample_rate and point_max take precedence over voxel_size
        target_point_max = self._get_point_max(original_size)

        # Step 2: Apply grid downsampling if voxel_size is specified
        idx_downsample, data_dict = self._grid_downsample(data_dict)

        # Step 3: Apply rectangular cropping to reach target_point_max
        if len(idx_downsample) > target_point_max:
            # Get center point using downsampled coordinates
            downsampled_coords = data_dict["coord"][idx_downsample]
            center = self._get_center_point_from_coords(
                downsampled_coords, data_dict, idx_downsample
            )

            # Get rectangular indices among downsampled points
            idx_crop_local = self._get_rectangular_indices(
                downsampled_coords, center, self.size, target_point_max
            )

            # Map back to original indices
            idx_final = idx_downsample[idx_crop_local]
        else:
            # Use all downsampled points (fewer than target)
            idx_final = idx_downsample

        # Apply clip point indices if present
        idx_final = self._apply_clip_point_indices(idx_final, data_dict)

        # Apply final cropping using combined indices
        data_dict = self._crop_data_dict(data_dict, idx_final)

        return data_dict

    def _get_center_point_from_coords(self, coords, data_dict, coord_indices):
        """Get center point for cropping from given coordinates."""
        if self.mode == "random":
            return coords[np.random.randint(len(coords))]
        elif self.mode == "center":
            return coords[len(coords) // 2]
        elif self.mode == "captioned":
            assert "caption_data" in data_dict, "Caption data is required for captioned mode"
            point_indices = data_dict["caption_data"]["idx"]
            sel_point_indices = np.random.randint(len(point_indices))
            caption_point_idx = np.random.choice(point_indices[sel_point_indices])

            # Find the index in coord_indices that matches caption_point_idx
            coord_mask = np.isin(coord_indices, caption_point_idx)
            if coord_mask.any():
                local_idx = np.where(coord_mask)[0][0]
                return coords[local_idx]
            else:
                # Fallback to random if caption point not in downsampled set
                return coords[np.random.randint(len(coords))]
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported")


@TRANSFORMS.register_module()
class ShufflePoint:
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "displacement" in data_dict.keys():
            data_dict["displacement"] = data_dict["displacement"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][shuffle_index]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][shuffle_index]
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary:
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][mask]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][mask]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][mask]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][mask]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][mask]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][mask]
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator:
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser:
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class FilterCaption:
    """Transform wrapper for CaptionFilter.

    Args:
        min_words (int): Minimum words required
        max_words (int): Maximum words allowed
        min_letter_ratio (float): Minimum letter ratio
        max_repetition_ratio (float): Maximum word repetition ratio
        min_unique_ratio (float): Minimum unique word ratio
        max_consecutive (int): Maximum consecutive repeats
    """

    def __init__(
        self,
        min_words: int = 3,
        max_words: int = 50,
        min_letter_ratio: float = 0.5,
        max_repetition_ratio: float = 0.4,
        min_unique_ratio: float = 0.3,
        max_consecutive: int = 3,
    ):
        self.filter = CaptionFilter(
            min_words=min_words,
            max_words=max_words,
            min_letter_ratio=min_letter_ratio,
            max_repetition_ratio=max_repetition_ratio,
            min_unique_ratio=min_unique_ratio,
            max_consecutive=max_consecutive,
        )

    def __call__(self, data_dict):
        if "caption_data" not in data_dict:
            return data_dict

        caption_dict = data_dict["caption_data"]
        captions = caption_dict["caption"]
        caption_point_indices = caption_dict["idx"]

        # Get valid flags for each caption
        valid_flags = self.filter(captions)

        # Filter captions and their corresponding point indices
        if not all(valid_flags):  # Only update if we have invalid captions
            filtered_captions = [cap for cap, valid in zip(captions, valid_flags) if valid]
            filtered_indices = [
                idx for idx, valid in zip(caption_point_indices, valid_flags) if valid
            ]

            data_dict["caption_data"] = {"caption": filtered_captions, "idx": filtered_indices}

        return data_dict

    def __repr__(self):
        return f"FilterCaption(filter={self.filter})"


@TRANSFORMS.register_module()
class AugmentCaption:
    def __init__(self):
        self.location_prompts = [
            # Original patterns
            "{}, located at {}",
            "{}, placed at {}",
            "{}, positioned at {}",
            "{}, set at {}",
            "{}, situated at {}",
            "{}, anchored at {}",
            "{}, fixed at {}",
            "{}, installed at {}",
            "{}, mounted at {}",
            # Position and arrangement patterns
            "{}, arranged at {}",
            "{}, aligned at {}",
            "{}, centered at {}",
            "{}, oriented at {}",
            "{}, stationed at {}",
            "{}, established at {}",
            "{}, secured at {}",
            "{}, deployed at {}",
            "{}, housed at {}",
            # Geographic and spatial patterns
            "{}, found at {}",
            "{}, discovered at {}",
            "{}, spotted at {}",
            "{}, observed at {}",
            "{}, detected at {}",
            "{}, present at {}",
            "{}, visible at {}",
            "{}, appearing at {}",
            "{}, residing at {}",
            # Placement and fixture patterns
            "{}, affixed at {}",
            "{}, attached at {}",
            "{}, fastened at {}",
            "{}, secured at {}",
            "{}, embedded at {}",
            "{}, integrated at {}",
            "{}, incorporated at {}",
            "{}, settled at {}",
            "{}, rooted at {}",
            # Location and presence patterns
            "{}, standing at {}",
            "{}, resting at {}",
            "{}, remaining at {}",
            "{}, contained at {}",
            "{}, confined at {}",
            "{}, preserved at {}",
            "{}, maintained at {}",
            "{}, stored at {}",
            "{}, kept at {}",
        ]

    def __call__(self, data_dict):
        if "caption_data" not in data_dict:
            return data_dict

        caption_dict = data_dict["caption_data"]
        coord = data_dict["coord"]
        captions = caption_dict["caption"]
        caption_point_indices = caption_dict["idx"]
        new_captions, new_idx = [], []
        for caption, idx in zip(captions, caption_point_indices):
            if len(idx) > 1:
                centroid = coord[idx].mean(axis=0)
                centroid_str = f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
                prompt = random.choice(self.location_prompts)
                augmented_caption = prompt.format(caption, centroid_str)
                new_captions.append(augmented_caption)
                new_idx.append(idx)
        data_dict["caption_data"] = {"caption": new_captions, "idx": new_idx}
        return data_dict


@TRANSFORMS.register_module()
class ClassNameAnonymizer:
    """Replace class names in captions with 'object' to prevent label leakage.

    Args:
        dataset (str): Name of the dataset ('scannet', 'scannetpp', 'matterport', 'structured3d')
        replacement (str): Word to replace class names with (default: 'object')
        case_sensitive (bool): Whether to match class names case-sensitively (default: False)
        handle_plurals (bool): Whether to handle plural forms of class names (default: True)
    """

    def __init__(
        self,
        dataset: str,
        replacement: str = "object",
        case_sensitive: bool = False,
        handle_plurals: bool = True,
    ):
        self.dataset = dataset
        self.replacement = replacement
        self.case_sensitive = case_sensitive
        self.handle_plurals = handle_plurals

        # Get class names from metadata based on dataset name
        self.class_names = self._get_class_names_from_dataset(dataset)

        # Prepare regex patterns for each class name
        self.patterns = []
        for class_name in self.class_names:
            # Create word boundary pattern to match whole words only
            pattern = r"\b" + re.escape(class_name) + r"\b"

            # Add pattern for plural form if enabled
            if self.handle_plurals:
                # Handle special plural cases
                if class_name.endswith("y"):
                    plural_pattern = r"\b" + re.escape(class_name[:-1]) + r"ies\b"
                elif class_name.endswith(("s", "x", "z", "ch", "sh")):
                    plural_pattern = r"\b" + re.escape(class_name) + r"es\b"
                else:
                    plural_pattern = r"\b" + re.escape(class_name) + r"s\b"

                # Combine singular and plural patterns
                pattern = f"({pattern}|{plural_pattern})"

            if not case_sensitive:
                self.patterns.append((re.compile(pattern, re.IGNORECASE), class_name))
            else:
                self.patterns.append((re.compile(pattern), class_name))

    def _get_class_names_from_dataset(self, dataset: str):
        """Get class names from the appropriate metadata based on dataset name."""
        if dataset.lower() == "scannet":
            from src.data.metadata.scannet import CLASS_LABELS_200

            return CLASS_LABELS_200
        elif dataset.lower() == "scannetpp":
            from src.data.metadata.scannetpp import CLASS_LABELS

            return CLASS_LABELS
        elif dataset.lower() == "matterport":
            from src.data.metadata.matterport3d import CLASS_LABELS_160

            return CLASS_LABELS_160
        elif dataset.lower() == "structured3d":
            from src.data.metadata.structured3d import CLASS_LABELS_25

            return CLASS_LABELS_25
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    def __call__(self, data_dict):
        if "caption_data" not in data_dict:
            return data_dict

        caption_dict = data_dict["caption_data"]
        captions = caption_dict["caption"]

        # Process each caption
        anonymized_captions = []
        for caption in captions:
            # Apply all patterns to the caption
            anonymized_caption = caption
            for pattern, class_name in self.patterns:
                # Use a function to determine if replacement should be singular or plural
                def replace_with_proper_form(match):
                    matched_text = match.group(0)
                    # If the matched text is plural form, use plural replacement
                    if self.handle_plurals and matched_text.lower() != class_name.lower():
                        return self.replacement + "s"
                    return self.replacement

                anonymized_caption = pattern.sub(replace_with_proper_form, anonymized_caption)

            anonymized_captions.append(anonymized_caption)

        # Update the caption data
        data_dict["caption_data"]["caption"] = anonymized_captions
        return data_dict

    def __repr__(self):
        return f"ClassNameAnonymizer(dataset='{self.dataset}', num_classes={len(self.class_names)}, replacement='{self.replacement}', handle_plurals={self.handle_plurals})"


class Compose:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


if __name__ == "__main__":
    # Test ClassNameAnonymizer
    print("Testing ClassNameAnonymizer...")

    # Create test data
    original_captions = [
        "There is a chair next to the table.",
        "A sofa and two chairs are in the living room.",
        "The kitchen has a refrigerator and an oven.",
        "A bed with a nightstand is in the bedroom.",
    ]

    # Test with ScanNet dataset
    dataset = "scannet"
    print(f"\nTesting with {dataset} dataset:")
    anonymizer = ClassNameAnonymizer(dataset=dataset)
    print(f"Loaded {len(anonymizer.class_names)} class names")

    # Create test data
    test_data = {
        "caption_data": {
            "caption": original_captions.copy(),
            "idx": [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        }
    }

    # Process the data
    result = anonymizer(test_data)

    # Print original vs anonymized captions
    print("Original vs Anonymized:")
    for orig, anon in zip(original_captions, result["caption_data"]["caption"]):
        print(f"Original: {orig}")
        print(f"Anonymized: {anon}")
        print("-" * 50)
