# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def normalize(x, eps=1e-9):
    return x / torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=eps)


def quat_apply(quat, vec):
    orig_shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, :3]
    w = quat[:, 3:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return (vec + w * t + torch.cross(xyz, t, dim=-1)).reshape(orig_shape)


def quat_rotate_inverse(quat, vec):
    quat_conj = quat.reshape(-1, 4).clone()
    quat_conj[:, :3] = -quat_conj[:, :3]
    return quat_apply(quat_conj, vec)


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).requires_grad_(requires_grad)
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float32, n_dims=3):
    zs = np.zeros((n_dims,), dtype=dtype)
    assert axis_idx < n_dims, 'the axis dim should be within the vector dimensions'
    zs[axis_idx] = value
    zs[0] = x_value
    return zs
