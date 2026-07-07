# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2w_config import GO2WRoughCfg


class Go2w(LeggedRobot):
    cfg: GO2WRoughCfg

    def get_current_obs(self):
        return self._compute_current_actor_obs()

    def _reward_hip_action_l2(self):
        return torch.sum(self.actions[:, [0, 4, 8, 12]] ** 2, dim=1)
