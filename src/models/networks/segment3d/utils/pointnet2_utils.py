# Copyright (c) Facebook, Inc. and its affiliates.
""" Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch """
import torch
from torch.autograd import Function
import torch.nn as nn
import src.models.networks.segment3d.utils.pytorch_utils as pt_utils
import sys

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(grad_out.contiguous(), idx, weight, m)

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(
        self,
        radius,
        nsample,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        # type: (QueryAndGroup, float, int, bool) -> None
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0,
                        num_unique,
                        (self.nsample - num_unique,),
                        dtype=torch.long,
                    )
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """Groups all features.

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
