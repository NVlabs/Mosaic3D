import numpy as np


def project_3d_point_to_image(xyz_world, pose, depth, depth_hw, depth_intrinsic):
    fx, fy = depth_intrinsic[0, 0], depth_intrinsic[1, 1]
    cx, cy = depth_intrinsic[0, 2], depth_intrinsic[1, 2]
    bx, by = depth_intrinsic[0, 3], depth_intrinsic[1, 3]

    # == 3D to camera coordination ===
    xyz = np.hstack((xyz_world[..., :3], np.ones((xyz_world.shape[0], 1))))
    xyz = np.dot(xyz, np.linalg.inv(np.transpose(pose)))

    # == camera to image coordination ===
    u = (xyz[..., 0] - bx) * fx / xyz[..., 2] + cx
    v = (xyz[..., 1] - by) * fy / xyz[..., 2] + cy
    d = xyz[..., 2]
    u = (u + 0.5).astype(np.int32)
    v = (v + 0.5).astype(np.int32)

    # filter out invalid points
    valid_mask = (d >= 0) & (u < depth_hw[1]) & (v < depth_hw[0]) & (u >= 0) & (v >= 0)
    valid_idx = np.where(valid_mask)[0]
    uv_1d = v * depth_hw[1] + u
    uv_1d = uv_1d[valid_idx]

    depth_1d = depth.reshape(-1)
    depth_mask_1d = depth_1d != 0

    image_depth = depth_1d[uv_1d.astype(np.int64)]
    depth_mask_1d = depth_mask_1d[uv_1d.astype(np.int64)]
    projected_depth = d[valid_idx]
    depth_valid_mask = depth_mask_1d & (np.abs(image_depth - projected_depth) <= 0.2 * image_depth)

    # corresponding image coords
    uv_1d = uv_1d[depth_valid_mask]
    valid_u = uv_1d % depth_hw[1]  # (width, long)
    valid_v = uv_1d // depth_hw[1]  # (height, short)
    uv = np.concatenate([valid_u[:, None], valid_v[:, None]], axis=-1)
    valid_idx = valid_idx[depth_valid_mask]  # corresponding point idx
    return valid_idx, uv
