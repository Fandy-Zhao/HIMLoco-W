# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaacgym import gymtorch
from legged_gym.utils.torch_compat import quat_rotate_inverse, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2w_config import GO2WRoughCfg


class Go2w(LeggedRobot):
    cfg: GO2WRoughCfg

    def _create_envs(self):
        super()._create_envs()
        wheel_names = []
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([dof_name for dof_name in self.dof_names if name in dof_name])
        if not wheel_names:
            raise RuntimeError(f'No wheel joints matched {self.cfg.asset.wheel_name} in DOFs: {self.dof_names}')
        self.wheel_dof_names = wheel_names
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(wheel_names):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], name)
        self.wheel_dof_indices = self.wheel_indices

        wheel_dof_set = set(int(idx) for idx in self.wheel_dof_indices.detach().cpu().tolist())
        self.leg_dof_names = [name for i, name in enumerate(self.dof_names) if i not in wheel_dof_set]
        self.leg_dof_indices = torch.tensor(
            [i for i in range(len(self.dof_names)) if i not in wheel_dof_set],
            dtype=torch.long,
            device=self.device,
        )
        if self.leg_dof_indices.numel() == 0:
            raise RuntimeError(f'No leg joints remain after wheel DOF split. DOFs: {self.dof_names}, wheels: {wheel_names}')

        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.body_names = rigid_body_names
        wheel_body_names = []
        for name in getattr(self.cfg.asset, 'wheel_body_name', self.cfg.asset.foot_name if isinstance(self.cfg.asset.foot_name, list) else [self.cfg.asset.foot_name]):
            wheel_body_names.extend([body_name for body_name in rigid_body_names if name in body_name])
        if not wheel_body_names:
            wheel_body_names = [body_name for body_name in rigid_body_names if self.cfg.asset.foot_name in body_name]
        if not wheel_body_names:
            raise RuntimeError(f'No wheel bodies matched {self.cfg.asset.foot_name} in bodies: {rigid_body_names}')
        self.wheel_body_names = wheel_body_names
        self.wheel_body_indices = torch.zeros(len(wheel_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(wheel_body_names):
            self.wheel_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        wheel_body_set = set(int(idx) for idx in self.wheel_body_indices.detach().cpu().tolist())
        leg_foot_names = [
            name for i, name in enumerate(rigid_body_names)
            if i not in wheel_body_set and ('foot' in name.lower() or 'calf' in name.lower())
        ]
        if not leg_foot_names:
            leg_foot_names = [
                name for i, name in enumerate(rigid_body_names)
                if i not in wheel_body_set and ('thigh' in name.lower() or 'calf' in name.lower())
            ]
        self.leg_foot_names = leg_foot_names
        self.leg_foot_indices = torch.zeros(len(leg_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(leg_foot_names):
            self.leg_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

    def _init_buffers(self):
        super()._init_buffers()
        self.init_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i, name in enumerate(self.dof_names):
            self.init_dof_pos[i] = self.cfg.init_state.init_joint_angles[name]
        self.init_dof_pos = self.init_dof_pos.unsqueeze(0)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.init_dof_pos
        self.dof_vel[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _compute_torques(self, actions):
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:, self.wheel_indices] = 0.

        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.wheel_indices] = 0.

        vel_ref = torch.zeros_like(actions_scaled)
        vel_ref[:, self.wheel_indices] = (actions * self.cfg.control.vel_scale)[:, self.wheel_indices]

        control_type = self.cfg.control.control_type
        if control_type == 'P':
            torques = (
                self.p_gains * self.Kp_factors * (actions_scaled + dof_err)
                + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)
            )
        elif control_type == 'V':
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == 'T':
            torques = actions_scaled
        else:
            raise NameError(f'Unknown controller type: {control_type}')
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_one_step_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.
        noise_vec[9:25] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[25:41] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[41:57] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[57:73] = 0.
        return noise_vec

    def _compute_current_actor_obs(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0.
        dof_pos_obs = self.dof_pos.clone()
        dof_pos_obs[:, self.wheel_indices] = 0.
        current_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            dof_err * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            dof_pos_obs,
            self.actions,
        ), dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec
        return current_obs

    def _compute_current_privileged_obs(self, current_obs=None):
        if current_obs is None:
            current_obs = self._compute_current_actor_obs()
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1,
            1.,
        ) * self.obs_scales.height_measurements
        return torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, heights), dim=-1)

    def compute_observations(self):
        current_obs = self._compute_current_actor_obs()
        self.obs_buf = torch.cat((current_obs, self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = self._compute_current_privileged_obs(current_obs)

    def get_current_obs(self):
        return self._compute_current_actor_obs()

    def compute_termination_observations(self, env_ids):
        return self._compute_current_privileged_obs()[env_ids]

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        if self.cfg.terrain.measure_heights:
            contact_flag = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
            self.reset_buf |= contact_flag < 0.20

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        reward = torch.square(base_height - self.cfg.rewards.base_height_target)
        relaxed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, 'env_has_goals'):
            relaxed_mask |= self.env_has_goals
        for name in ['terrain_idx_Slope', 'terrain_idx_BridgeA', 'terrain_idx_BridgeB']:
            if hasattr(self.cfg.terrain, name):
                relaxed_mask |= self._terrain_is(getattr(self.cfg.terrain, name))
        reward[relaxed_mask] *= 0.5
        return reward

    def _reward_dof_vel(self):
        if not hasattr(self, 'leg_dof_indices'):
            dof_vel = self.dof_vel.clone()
            dof_vel[:, self.wheel_indices] = 0.
            return torch.sum(torch.square(dof_vel), dim=1)
        return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        if not hasattr(self, 'leg_dof_indices'):
            dof_err = self.dof_pos - self.default_dof_pos
            dof_err[:, self.wheel_indices] = 0.
        else:
            dof_err = self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_hip_action_l2(self):
        return torch.sum(self.actions[:, [0, 4, 8, 12]] ** 2, dim=1)
