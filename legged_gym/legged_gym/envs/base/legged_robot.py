# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from legged_gym.utils.torch_compat import (
    get_axis_params,
    quat_apply,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.history_length = int(self.num_obs / self.num_one_step_obs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.final_goal_reset_buf = self._check_final_goal_termination()
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.final_goal_reset_buf
        if self.cfg.terrain.measure_heights and bool(getattr(self.cfg.terrain, 'terminate_on_ground_contact', False)):
            base_clearance = self._get_base_heights()
            threshold = float(getattr(self.cfg.terrain, 'ground_contact_height_threshold', 0.16))
            self.reset_buf |= base_clearance < threshold

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        episode_details = self._collect_episode_health_details(env_ids) if hasattr(self, '_collect_episode_health_details') else None
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        self._randomize_terrain_cell_on_reset(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        goal_success_rate = None
        if hasattr(self, 'cur_goal_idx') and hasattr(self, 'reached_goal'):
            valid_goal_eps = self.env_has_goals[env_ids] if hasattr(self, 'env_has_goals') else torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            if torch.any(valid_goal_eps):
                final_goal_idx = max(int(getattr(self.cfg.terrain, 'num_goals', 1)) - 1, 0)
                goal_success = (self.cur_goal_idx[env_ids] >= final_goal_idx) & self.reached_goal[env_ids]
                goal_success_rate = torch.mean(goal_success[valid_goal_eps].float())

        self._reset_goals(env_ids)
        self._update_goals()
        self._update_goal_yaw_command()
        if hasattr(self, 'prev_goal_dist'):
            self.prev_goal_dist[env_ids] = self._compute_goal_distance(env_ids)
        if hasattr(self, 'prev_abs_delta_yaw') and self.commands.shape[1] >= 3:
            self.prev_abs_delta_yaw[env_ids] = torch.abs(self.commands[env_ids, 2])
        if hasattr(self, 'final_goal_reset_buf'):
            self.final_goal_reset_buf[env_ids] = False

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        if hasattr(self, 'leg_feet_air_time'):
            self.leg_feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if goal_success_rate is not None:
            self.extras["episode"]["goal_success_rate"] = goal_success_rate
            if hasattr(self, '_update_continuous_speed_curriculum'):
                self._update_continuous_speed_curriculum(goal_success_rate)
        if hasattr(self, 'speed_curriculum_ratio'):
            self.extras["episode"]["curriculum_ratio"] = self.speed_curriculum_ratio.clone()
            self.extras["episode"]["current_speed_max"] = torch.tensor(float(self.command_ranges["lin_vel_x"][1]), device=self.device)
            self.extras["episode"]["success_rate_ema"] = self.speed_curriculum_success_ema.clone()
            if hasattr(self, 'speed_curriculum_progress_ema'):
                self.extras["episode"]["progress_ema"] = self.speed_curriculum_progress_ema.clone()
            if hasattr(self, 'speed_curriculum_intermediate_goal_ema'):
                self.extras["episode"]["intermediate_goal_rate_ema"] = self.speed_curriculum_intermediate_goal_ema.clone()
        if episode_details is not None:
            self.extras["episode_details"] = episode_details
        if len(env_ids) > 0 and self.commands.shape[1] >= 3:
            heading_error = self._get_heading_error()
            current_yaw = self._get_base_yaw()
            self.extras["episode"]["command_vx"] = torch.mean(self.commands[env_ids, 0])
            self.extras["episode"]["command_vy"] = torch.mean(self.commands[env_ids, 1])
            self.extras["episode"]["delta_yaw_cmd"] = torch.mean(self.commands[env_ids, 2])
            self.extras["episode"]["current_yaw"] = torch.mean(current_yaw[env_ids])
            self.extras["episode"]["heading_error"] = torch.mean(heading_error[env_ids])
            self.extras["episode"]["base_yaw_rate"] = torch.mean(self.base_ang_vel[env_ids, 2])
            self.extras["episode"]["yaw_rate_l2"] = torch.mean(torch.square(self.base_ang_vel[env_ids, 2]))
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        if hasattr(self, '_prepare_goal_alignment_rewards'):
            self._prepare_goal_alignment_rewards()
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        if hasattr(self, '_finalize_goal_alignment_rewards'):
            self._finalize_goal_alignment_rewards()
    
    def _compute_current_actor_obs(self):
        """ Computes current actor observation (proprioceptive only, no privileged info).

        Order: ang_vel → gravity → commands → dof_err → dof_vel → actions
        """
        dof_err = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        if hasattr(self, 'wheel_indices'):
            dof_err[:, self.wheel_indices] = 0.
        current_obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,      # 3
            self.projected_gravity,                           # 3
            self._get_command_obs(),                          # 3
            dof_err,                                          # num_actions
            self.dof_vel * self.obs_scales.dof_vel,           # num_actions
            self.actions,                                     # num_actions
        ), dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec
        return current_obs

    def _compute_current_privileged_obs(self, current_obs=None):
        """ Computes privileged observation (actor obs + privileged info).
        Appends lin_vel, optionally disturbance and height measurements.
        """
        if current_obs is None:
            current_obs = self._compute_current_actor_obs()
        if self.cfg.domain_rand.disturbance:
            current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        else:
            current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.,
            ) * self.obs_scales.height_measurements
            current_obs = torch.cat((current_obs, heights), dim=-1)
        return current_obs

    def compute_observations(self):
        """ Computes observations """
        current_obs = self._compute_current_actor_obs()
        current_priv_obs = self._compute_current_privileged_obs(current_obs)

        self.obs_buf = torch.cat(
            (current_obs, self.obs_buf[:, :-self.num_one_step_obs]),
            dim=-1,
        )

        self.privileged_obs_buf = torch.cat(
            (current_priv_obs, self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]),
            dim=-1,
        )

    def get_current_obs(self):
        return self._compute_current_privileged_obs()

    def compute_termination_observations(self, env_ids):
        """ Computes termination observations (privileged obs with history) """
        priv_obs = self._compute_current_privileged_obs()
        return torch.cat((priv_obs, self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    


    def _init_env_terrain_idx(self):
        if getattr(self.cfg.terrain, 'use_terrain_idx', False) and hasattr(self, 'terrain'):
            if hasattr(self.terrain, 'terrain_idx_map'):
                self.terrain_idx_map = torch.from_numpy(self.terrain.terrain_idx_map).to(self.device).long()
            elif hasattr(self.terrain, 'terrain_type'):
                self.terrain_idx_map = torch.from_numpy(self.terrain.terrain_type).to(self.device).long()
            else:
                self.terrain_idx_map = torch.full((self.cfg.terrain.num_rows, self.cfg.terrain.num_cols), self.cfg.terrain.terrain_idx_unknown, device=self.device, dtype=torch.long)
            self.env_terrain_idx = self.terrain_idx_map[self.terrain_levels, self.terrain_types]
        else:
            self.env_terrain_idx = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

    def _sync_env_terrain_idx(self, env_ids=None):
        if not hasattr(self, 'env_terrain_idx') or not hasattr(self, 'terrain_idx_map'):
            return
        if env_ids is None:
            self.env_terrain_idx[:] = self.terrain_idx_map[self.terrain_levels, self.terrain_types]
        else:
            self.env_terrain_idx[env_ids] = self.terrain_idx_map[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _terrain_is(self, idx):
        if not hasattr(self, 'env_terrain_idx'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.env_terrain_idx == idx

    def _terrain_in(self, idx_list):
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, 'env_terrain_idx'):
            return mask
        for idx in idx_list:
            mask |= self.env_terrain_idx == idx
        return mask

    def _indices_from_names(self, all_names, names, label):
        missing = [name for name in names if name not in all_names]
        if missing:
            raise RuntimeError(f'Missing {label} names {missing}. Available names: {all_names}')
        return torch.tensor([all_names.index(name) for name in names], dtype=torch.long, device=self.device)

    def _matching_names(self, all_names, patterns):
        matched = []
        for pattern in patterns:
            matched.extend([name for name in all_names if pattern in name])
        return matched

    def _get_base_yaw(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _get_heading_error(self):
        return torch.clamp(self.commands[:, 2], -torch.pi, torch.pi)

    def _get_delta_yaw_to_goal(self):
        return self._get_heading_error()

    def _get_command_obs(self):
        delta_yaw_scale = float(getattr(self.obs_scales, 'delta_yaw', getattr(self.obs_scales, 'heading_error', 1.0)))
        return torch.cat((
            self.commands[:, :2] * self.commands_scale[:2],
            self._get_delta_yaw_to_goal().unsqueeze(1) * delta_yaw_scale,
        ), dim=1)

    def debug_terrain_idx(self, prefix='[terrain_idx]'):
        if not hasattr(self, 'env_terrain_idx'):
            print(prefix, 'env_terrain_idx missing')
            return
        unique_ids, counts = torch.unique(self.env_terrain_idx, return_counts=True)
        print(prefix, 'unique:', unique_ids.detach().cpu().tolist())
        print(prefix, 'counts:', counts.detach().cpu().tolist())
        print(prefix, 'first 16:', self.env_terrain_idx[:16].detach().cpu().tolist())

    def _goal_commands_enabled(self):
        return bool(getattr(self.cfg.commands, 'use_goal_yaw_command', False)) and bool(getattr(self.cfg.terrain, 'use_parkour_goals', False))

    def _init_goal_buffers(self):
        self.num_goal_waypoints = max(int(getattr(self.cfg.terrain, 'num_goals', 0)), 0)
        self.num_future_goal_obs = max(int(getattr(self.cfg.terrain, 'num_future_goal_obs', 0)), 0)
        goal_slots = max(self.num_goal_waypoints + self.num_future_goal_obs, 1)
        self.terrain_goals = None
        self.env_goals = torch.zeros(self.num_envs, goal_slots, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.cur_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.env_has_goals = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.target_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.goal_yaw_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_pos_rel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.curr_goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.goal_progress_delta_raw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_abs_delta_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.goal_reach_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.final_goal_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.goal_success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.goal_reached_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.speed_curriculum_ratio = torch.tensor(float(getattr(self.cfg.commands, 'curriculum_min_ratio', 0.0)), dtype=torch.float, device=self.device)
        self.speed_curriculum_success_ema = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.speed_curriculum_progress_ema = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.speed_curriculum_intermediate_goal_ema = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.last_speed_curriculum_update = 0
        if bool(getattr(self.cfg.commands, 'use_continuous_speed_curriculum', False)):
            self._apply_continuous_speed_curriculum_limit()
        if not self._goal_commands_enabled() or self.num_goal_waypoints <= 0:
            return
        if not hasattr(self, 'terrain') or not hasattr(self.terrain, 'goals'):
            return
        terrain_goals = torch.as_tensor(self.terrain.goals, dtype=torch.float, device=self.device)
        if terrain_goals.ndim != 4 or terrain_goals.shape[-1] != 3 or terrain_goals.shape[2] == 0:
            return
        self.terrain_goals = terrain_goals
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._refresh_env_goals(env_ids)
        self.prev_goal_dist[:] = self._compute_goal_distance(env_ids)

    def _refresh_env_goals(self, env_ids):
        if env_ids is None or len(env_ids) == 0 or self.terrain_goals is None:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        levels = self.terrain_levels[env_ids].clamp(0, self.terrain_goals.shape[0] - 1)
        types = self.terrain_types[env_ids].clamp(0, self.terrain_goals.shape[1] - 1)
        goals = self.terrain_goals[levels, types]
        valid = torch.isfinite(goals).view(goals.shape[0], -1).all(dim=1)
        valid &= torch.norm(goals[:, -1, :2] - goals[:, 0, :2], dim=1) > 1e-4
        slots = self.env_goals.shape[1]
        copy_count = min(goals.shape[1], slots)
        self.env_goals[env_ids] = 0.
        self.env_goals[env_ids, :copy_count] = goals[:, :copy_count]
        if copy_count < slots:
            self.env_goals[env_ids, copy_count:] = goals[:, copy_count - 1:copy_count]
        self.env_has_goals[env_ids] = True

    def _collect_episode_health_details(self, env_ids):
        if not hasattr(self, 'cur_goal_idx') or not hasattr(self, 'reached_goal'):
            return None
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        final_goal_idx = max(int(getattr(self.cfg.terrain, 'num_goals', 1)) - 1, 0)
        raw_success = (self.cur_goal_idx[env_ids] >= final_goal_idx) & self.reached_goal[env_ids]
        goal_dist = self._compute_goal_distance(env_ids) if hasattr(self, '_compute_goal_distance') else torch.zeros(len(env_ids), device=self.device)
        yaw_error = torch.abs(self._get_delta_yaw_to_goal()[env_ids]) if hasattr(self, '_get_delta_yaw_to_goal') else torch.zeros(len(env_ids), device=self.device)
        base_height = self.root_states[env_ids, 2]
        roll_pitch_proxy = torch.norm(self.projected_gravity[env_ids, :2], dim=1)
        wheel_slip = torch.abs(self._reward_wheel_slip()[env_ids]) if hasattr(self, '_reward_wheel_slip') else torch.zeros(len(env_ids), device=self.device)
        collision = self._reward_collision()[env_ids] > 0 if hasattr(self, '_reward_collision') else torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        timeout = self.time_out_buf[env_ids].bool() if hasattr(self, 'time_out_buf') else torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        yaw_ok = yaw_error < float(getattr(self.cfg.rewards, 'healthy_yaw_threshold', 0.6))
        height_ok = (base_height > float(getattr(self.cfg.rewards, 'healthy_base_height_min', 0.18))) & (base_height < float(getattr(self.cfg.rewards, 'healthy_base_height_max', 0.65)))
        roll_pitch_ok = roll_pitch_proxy < float(getattr(self.cfg.rewards, 'healthy_roll_pitch_proxy_max', 0.65))
        slip_ok = wheel_slip < float(getattr(self.cfg.rewards, 'healthy_wheel_slip_max', 0.8))
        healthy_success = raw_success & yaw_ok & height_ok & roll_pitch_ok & slip_ok & (~collision) & (~timeout)
        return {
            'env_ids': env_ids.detach().cpu(),
            'raw_success': raw_success.detach().cpu(),
            'healthy_success': healthy_success.detach().cpu(),
            'goal_dist_final': goal_dist.detach().cpu(),
            'yaw_error_final': yaw_error.detach().cpu(),
            'base_height_final': base_height.detach().cpu(),
            'roll_pitch_proxy_final': roll_pitch_proxy.detach().cpu(),
            'wheel_slip_final': wheel_slip.detach().cpu(),
            'collision_or_stumble': collision.detach().cpu(),
            'timeout': timeout.detach().cpu(),
            'episode_length': self.episode_length_buf[env_ids].detach().cpu(),
        }

    def _reset_goals(self, env_ids):
        if not hasattr(self, 'env_goals'):
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0.
        self.reached_goal[env_ids] = False
        if hasattr(self, 'goal_reach_event'):
            self.goal_reach_event[env_ids] = False
        if hasattr(self, 'final_goal_event'):
            self.final_goal_event[env_ids] = False
        if hasattr(self, 'goal_success_event'):
            self.goal_success_event[env_ids] = False
        if hasattr(self, 'goal_reached_buf'):
            self.goal_reached_buf[env_ids] = False
        if hasattr(self, 'success_latched'):
            self.success_latched[env_ids] = False
        self._refresh_env_goals(env_ids)
        if hasattr(self, 'prev_goal_dist'):
            self.prev_goal_dist[env_ids] = self._compute_goal_distance(env_ids)
        if hasattr(self, 'prev_abs_delta_yaw'):
            self.prev_abs_delta_yaw[env_ids] = torch.abs(self.commands[env_ids, 2])

    def _gather_current_goals(self, future=0):
        idx = torch.clamp(self.cur_goal_idx + int(future), 0, self.env_goals.shape[1] - 1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.env_goals[env_ids, idx]

    def _compute_goal_distance(self, env_ids=None):
        if not hasattr(self, 'env_goals') or not hasattr(self, 'cur_goal_idx'):
            if env_ids is None:
                return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            return torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        cur_goal = self.env_goals[env_ids, self.cur_goal_idx[env_ids]]
        return torch.norm(self.root_states[env_ids, :2] - cur_goal[:, :2], dim=1)

    def _goal_alignment_active(self):
        if not hasattr(self, 'env_goals') or not hasattr(self, 'cur_goal_idx'):
            return False
        if not hasattr(self, 'prev_goal_dist') or not hasattr(self, 'curr_goal_dist'):
            return False
        if hasattr(self, 'env_has_goals') and not torch.any(self.env_has_goals):
            return False
        return True

    def _prepare_goal_alignment_rewards(self):
        if hasattr(self, 'goal_success_event') and hasattr(self, 'final_goal_event'):
            self.goal_success_event[:] = self.final_goal_event

        if not self._goal_alignment_active():
            return

        self.curr_goal_dist[:] = self._compute_goal_distance()
        max_delta = float(getattr(self.cfg.rewards, 'goal_progress_delta_max', 1.0))
        self.goal_progress_delta_raw[:] = torch.clamp(
            self.prev_goal_dist - self.curr_goal_dist,
            -max_delta,
            max_delta,
        )
        if hasattr(self, 'env_has_goals'):
            self.goal_progress_delta_raw[~self.env_has_goals] = 0.0

    def _finalize_goal_alignment_rewards(self):
        if not self._goal_alignment_active():
            return

        if hasattr(self, 'env_has_goals'):
            self.prev_goal_dist[self.env_has_goals] = self.curr_goal_dist[self.env_has_goals]
        else:
            self.prev_goal_dist[:] = self.curr_goal_dist

    def _apply_continuous_speed_curriculum_limit(self):
        ratio = float(self.speed_curriculum_ratio.item()) if hasattr(self, 'speed_curriculum_ratio') else 0.0
        start = float(getattr(self.cfg.commands, 'curriculum_start_speed', self.command_ranges['lin_vel_x'][1]))
        goal = float(getattr(self.cfg.commands, 'curriculum_goal_speed', getattr(self.cfg.commands, 'curriculum_target_speed', self.command_ranges['lin_vel_x'][1])))
        self.command_ranges['lin_vel_x'][1] = start + ratio * (goal - start)

    def _update_continuous_speed_curriculum(self, current_success_rate):
        if not bool(getattr(self.cfg.commands, 'use_continuous_speed_curriculum', False)):
            return
        interval = int(getattr(self.cfg.commands, 'curriculum_update_interval', 100))
        if self.common_step_counter - self.last_speed_curriculum_update < interval:
            return
        self.last_speed_curriculum_update = self.common_step_counter
        alpha = float(getattr(self.cfg.commands, 'curriculum_ema_alpha', 0.1))
        self.speed_curriculum_success_ema = alpha * current_success_rate + (1.0 - alpha) * self.speed_curriculum_success_ema

        progress_signal = torch.clamp(self.goal_progress_delta_raw, min=0.0).mean() if hasattr(self, 'goal_progress_delta_raw') else torch.tensor(0.0, device=self.device)
        self.speed_curriculum_progress_ema = alpha * progress_signal + (1.0 - alpha) * self.speed_curriculum_progress_ema
        if hasattr(self, 'cur_goal_idx'):
            final_goal_idx = max(int(getattr(self.cfg.terrain, 'num_goals', 1)) - 1, 0)
            intermediate_goal_rate = ((self.cur_goal_idx > 0) & (self.cur_goal_idx < final_goal_idx)).float().mean()
        else:
            intermediate_goal_rate = torch.tensor(0.0, device=self.device)
        self.speed_curriculum_intermediate_goal_ema = alpha * intermediate_goal_rate + (1.0 - alpha) * self.speed_curriculum_intermediate_goal_ema

        ratio = float(self.speed_curriculum_ratio.item())
        metric = str(getattr(self.cfg.commands, 'curriculum_metric', 'success'))
        if metric == 'progress':
            high = float(getattr(self.cfg.commands, 'curriculum_progress_high', 0.045))
            low = float(getattr(self.cfg.commands, 'curriculum_progress_low', 0.015))
            intermediate_high = float(getattr(self.cfg.commands, 'curriculum_intermediate_goal_high', 0.25))
            advance = float(self.speed_curriculum_progress_ema.item()) > high or float(self.speed_curriculum_intermediate_goal_ema.item()) > intermediate_high
            retreat = float(self.speed_curriculum_progress_ema.item()) < low
        else:
            high = float(getattr(self.cfg.commands, 'curriculum_success_high', 0.75))
            low = float(getattr(self.cfg.commands, 'curriculum_success_low', 0.35))
            advance = float(self.speed_curriculum_success_ema.item()) > high
            retreat = float(self.speed_curriculum_success_ema.item()) < low
        if advance:
            ratio += float(getattr(self.cfg.commands, 'curriculum_increase_step', 0.03))
        elif retreat and not bool(getattr(self.cfg.commands, 'curriculum_hold_on_drop', True)):
            ratio -= float(getattr(self.cfg.commands, 'curriculum_decrease_step', 0.01))
        min_ratio = float(getattr(self.cfg.commands, 'curriculum_min_ratio', 0.0))
        max_ratio = float(getattr(self.cfg.commands, 'curriculum_max_ratio', 1.0))
        ratio = max(min_ratio, min(max_ratio, ratio))
        self.speed_curriculum_ratio.fill_(ratio)
        self._apply_continuous_speed_curriculum_limit()

    def _update_goals(self):
        if hasattr(self, 'goal_reach_event'):
            self.goal_reach_event[:] = False
        if hasattr(self, 'final_goal_event'):
            self.final_goal_event[:] = False
        if hasattr(self, 'goal_success_event'):
            self.goal_success_event[:] = False
        if not hasattr(self, 'env_goals') or self.terrain_goals is None:
            return
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_goals = self._gather_current_goals()
        reached = active & (torch.norm(self.root_states[:, :2] - current_goals[:, :2], dim=1) < self.cfg.terrain.next_goal_threshold)
        self.reached_goal[:] = reached
        self.reach_goal_timer[reached] += 1
        self.reach_goal_timer[active & ~reached] = 0
        goal_delay = getattr(self.cfg.terrain, 'goal_reach_delay', getattr(self.cfg.terrain, 'reach_goal_delay', 0.1))
        delay_steps = max(1.0, float(goal_delay) / self.dt)
        final_goal_idx = max(self.num_goal_waypoints - 1, 0)
        ready = active & reached & (self.reach_goal_timer > delay_steps)
        advance = ready & (self.cur_goal_idx < final_goal_idx)
        final_ready = ready & (self.cur_goal_idx >= final_goal_idx)
        final_event = final_ready & (~self.success_latched) if hasattr(self, 'success_latched') else final_ready
        if hasattr(self, 'goal_reach_event'):
            self.goal_reach_event[:] = advance | final_event
        if hasattr(self, 'final_goal_event'):
            self.final_goal_event[:] = final_event
        if hasattr(self, 'success_latched'):
            self.success_latched[final_event] = True
        self.cur_goal_idx[advance] += 1
        self.reach_goal_timer[advance] = 0
        self._update_goal_commands()

    def _update_goal_commands(self):
        if not hasattr(self, 'env_goals') or not hasattr(self, 'cur_goal_idx') or self.terrain_goals is None:
            return
        env_ids = torch.arange(self.num_envs, device=self.device)
        cur_goal = self.env_goals[env_ids, self.cur_goal_idx]
        target_vec = cur_goal[:, :2] - self.root_states[:, :2]
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        delta_yaw = wrap_to_pi(target_yaw - self._get_base_yaw())

        self.target_pos_rel[:] = target_vec
        self.target_yaw[:] = target_yaw
        self.goal_yaw_error[:] = delta_yaw
        self.commands[:, 2] = delta_yaw
        if hasattr(self, 'env_has_goals'):
            self.env_has_goals[:] = True

    def _update_goal_yaw_command(self):
        self._update_goal_commands()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._update_goals()
        self._update_goal_commands()

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self._update_goal_commands()

        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)

        # set y commands of high vel envs to zero
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            For wheeled robots: position PD for legs, velocity PD for wheels.
            For pure legged robots: position PD for all DOFs.
        """
        actions_scaled = actions * self.cfg.control.action_scale

        if hasattr(self, 'wheel_indices'):
            dof_err = self.default_dof_pos - self.dof_pos
            dof_err[:, self.wheel_indices] = 0.
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
        else:
            actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_reduction
            self.joint_pos_target = self.default_dof_pos + actions_scaled

            control_type = self.cfg.control.control_type
            if control_type == "P":
                torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            elif control_type == "V":
                torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            elif control_type == "T":
                torques = actions_scaled
            else:
                raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions (or default only when configured).
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if bool(getattr(self.cfg.init_state, 'reset_to_default_pos', False)):
            self.dof_pos[env_ids] = self.init_dof_pos
        else:
            rng = self.cfg.domain_rand.initial_joint_pos_range
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                rng[0], rng[1], (len(env_ids), self.num_dof), device=self.device
            )
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._sync_env_terrain_idx(env_ids)

    def _randomize_terrain_cell_on_reset(self, env_ids):
        if not getattr(self.cfg.terrain, 'randomize_terrain_on_reset', False):
            return
        if len(env_ids) == 0 or not getattr(self, 'custom_origins', False):
            return
        if not hasattr(self, 'terrain_levels') or not hasattr(self, 'terrain_types'):
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if getattr(self.cfg.terrain, 'randomize_terrain_levels_on_reset', True):
            self.terrain_levels[env_ids] = torch.randint(
                0,
                self.max_terrain_level,
                (len(env_ids),),
                device=self.device,
                dtype=self.terrain_levels.dtype,
            )
        if getattr(self.cfg.terrain, 'randomize_terrain_types_on_reset', True):
            self.terrain_types[env_ids] = torch.randint(
                0,
                self.cfg.terrain.num_cols,
                (len(env_ids),),
                device=self.device,
                dtype=self.terrain_types.dtype,
            )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._sync_env_terrain_idx(env_ids)

    def _check_final_goal_termination(self):
        if not bool(getattr(self.cfg.terrain, 'terminate_after_reaching_final_goal', False)):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, 'cur_goal_idx') or not hasattr(self, 'reached_goal'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        final_goal_idx = max(int(getattr(self.cfg.terrain, 'num_goals', 0)) - 1, 0)
        if final_goal_idx <= 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (self.cur_goal_idx >= final_goal_idx) & self.reached_goal
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if "tracking_lin_vel" not in self.episode_sums or "tracking_lin_vel" not in self.reward_scales:
            return
        if self.reward_scales["tracking_lin_vel"] <= 0.0:
            return
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to actor observations.
            Matches order: ang_vel(3) → gravity(3) → commands(3) → dof_err(N) → dof_vel(N) → actions(N)
        """
        noise_vec = torch.zeros(self.cfg.env.num_one_step_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands: no noise
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0.  # actions: no noise
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, target yaw
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, getattr(self.obs_scales, 'delta_yaw', 1.0)], device=self.device, requires_grad=False)
        self._init_goal_buffers()
        self.final_goal_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        if hasattr(self, 'leg_foot_indices') and self.leg_foot_indices.numel() > 0:
            self.leg_feet_air_time = torch.zeros(self.num_envs, self.leg_foot_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
            self.leg_last_contacts = torch.zeros(self.num_envs, len(self.leg_foot_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # store default DOF positions for reset (used by subclasses)
        self.init_dof_pos = self.default_dof_pos.clone()


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self._validate_reward_scales()
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _validate_reward_scales(self):
        missing = []
        for name, scale in self.reward_scales.items():
            if scale == 0:
                continue
            fn_name = '_reward_' + name
            if not hasattr(self, fn_name):
                missing.append((name, fn_name, scale))
        if missing:
            msg = "\n".join([f"{name}: {fn_name}, scale={scale}" for name, fn_name, scale in missing])
            raise RuntimeError(f"Missing reward functions for non-zero scales:\n{msg}")

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # --- Auto-detect wheel DOFs and bodies ---
        wheel_names = list(getattr(self.cfg.asset, 'wheel_dof_names', []) or [])
        if not wheel_names:
            wheel_patterns = getattr(self.cfg.asset, 'wheel_name', None)
            if wheel_patterns:
                wheel_patterns = wheel_patterns if isinstance(wheel_patterns, list) else [wheel_patterns]
                wheel_names = self._matching_names(self.dof_names, wheel_patterns)
        if wheel_names:
            self.wheel_dof_names = wheel_names
            self.wheel_indices = self._indices_from_names(self.dof_names, wheel_names, 'wheel DOF')
            self.wheel_dof_indices = self.wheel_indices
            wheel_forward_sign = getattr(self.cfg.asset, 'wheel_forward_sign', None)
            if wheel_forward_sign is None:
                self.wheel_forward_sign = torch.ones(len(self.wheel_dof_indices), dtype=torch.float, device=self.device, requires_grad=False)
            else:
                self.wheel_forward_sign = torch.tensor(wheel_forward_sign, dtype=torch.float, device=self.device, requires_grad=False)

            wheel_dof_set = set(int(idx) for idx in self.wheel_dof_indices.detach().cpu().tolist())
            self.leg_dof_names = list(getattr(self.cfg.asset, 'leg_dof_names', []) or [])
            if self.leg_dof_names:
                self.leg_dof_indices = self._indices_from_names(self.dof_names, self.leg_dof_names, 'leg DOF')
            else:
                self.leg_dof_names = [name for i, name in enumerate(self.dof_names) if i not in wheel_dof_set]
                self.leg_dof_indices = torch.tensor(
                    [i for i in range(len(self.dof_names)) if i not in wheel_dof_set],
                    dtype=torch.long, device=self.device,
                )

            rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
            wheel_body_names = list(getattr(self.cfg.asset, 'wheel_body_names', []) or [])
            if not wheel_body_names:
                wheel_body_patterns = getattr(self.cfg.asset, 'wheel_body_name', None)
                if wheel_body_patterns:
                    wheel_body_patterns = wheel_body_patterns if isinstance(wheel_body_patterns, list) else [wheel_body_patterns]
                    wheel_body_names = self._matching_names(rigid_body_names, wheel_body_patterns)
            if wheel_body_names:
                self.wheel_body_names = wheel_body_names
                self.wheel_body_indices = torch.tensor(
                    [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name) for name in wheel_body_names],
                    dtype=torch.long, device=self.device,
                )

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

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self._init_env_terrain_idx()
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            self.env_terrain_idx = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    def _get_heights_at_points(self, points_xy):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(points_xy.shape[:2], dtype=torch.float, device=self.device)
        if not hasattr(self, 'height_samples') or not hasattr(self, 'terrain'):
            return torch.mean(self.measured_heights, dim=1, keepdim=True).repeat(1, points_xy.shape[1])

        points = points_xy + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].reshape(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].reshape(-1), 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(torch.min(heights1, heights2), heights3)
        return heights.view(points_xy.shape[0], points_xy.shape[1]) * self.terrain.cfg.vertical_scale

    def _get_obstacle_ahead_mask(self):
        distances = getattr(self.cfg.rewards, 'obstacle_probe_distances', [0.25, 0.40, 0.55, 0.70])
        distances = torch.tensor(distances, dtype=torch.float, device=self.device)
        if distances.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        heading = self._get_base_yaw() + self.commands[:, 2]
        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        points_xy = self.root_states[:, None, :2] + forward[:, None, :] * distances[None, :, None]
        ahead_heights = self._get_heights_at_points(points_xy)
        terrain_height_now = self._get_heights_at_points(self.root_states[:, None, :2])[:, 0]

        return self._compute_obstacle_mask_from_heights(terrain_height_now, ahead_heights)

    def _compute_obstacle_mask_from_heights(self, h_now, h_ahead):
        height_diff = h_ahead - h_now.unsqueeze(1)
        obstacle_threshold = float(getattr(self.cfg.rewards, 'obstacle_height_threshold', 0.04))
        gap_threshold = float(getattr(self.cfg.rewards, 'gap_height_threshold', 0.06))
        step_or_wall = torch.any(height_diff > obstacle_threshold, dim=1)
        gap = torch.any(height_diff < -gap_threshold, dim=1)
        mask = step_or_wall | gap
        if hasattr(self, 'env_terrain_idx') and mask.shape[0] == self.env_terrain_idx.shape[0]:
            mask &= self.env_terrain_idx >= 0
        return mask

    #------------ reward functions----------------
    def _sum_leg_wheel_penalty(self, per_dof_value, wheel_weight=1.0):
        """Sum per-DOF penalty with leg/wheel separation.

        Args:
            per_dof_value: Tensor of shape (num_envs, num_dof) with per-DOF values
            wheel_weight: Weight multiplier for wheel DOF penalties
        Returns:
            Tensor of shape (num_envs,) with summed penalties
        """
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum(per_dof_value, dim=1)

        leg_value = per_dof_value[:, self.leg_dof_indices]
        wheel_value = per_dof_value[:, self.wheel_dof_indices]

        reward = torch.sum(leg_value, dim=1)
        if wheel_value.numel() > 0:
            reward = reward + wheel_weight * torch.sum(wheel_value, dim=1)
        return reward

    def _reward_goal_delta_progress(self):
        if not hasattr(self, 'prev_goal_dist') or not hasattr(self, 'curr_goal_dist'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        curr_dist = self._compute_goal_distance()
        delta = self.prev_goal_dist - curr_dist
        progress_vel = delta / max(self.dt, 1e-6)

        max_progress = float(getattr(self.cfg.rewards, 'goal_progress_delta_max', 1.0))
        reward = torch.clamp(progress_vel, 0.0, max_progress)

        self.curr_goal_dist[:] = curr_dist
        self.prev_goal_dist[:] = curr_dist
        # goal_progress_delta_raw is written by _prepare_goal_alignment_rewards

        return reward

    def _reward_delta_yaw_progress(self):
        if not hasattr(self, 'prev_abs_delta_yaw') or self.commands.shape[1] < 3:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        curr = torch.abs(self.commands[:, 2])
        delta = self.prev_abs_delta_yaw - curr
        self.prev_abs_delta_yaw[:] = curr
        return torch.clamp(delta, -0.2, 0.2)

    def _get_goal_progress_state(self):
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        target_dir = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        if not hasattr(self, 'env_goals') or not hasattr(self, 'cur_goal_idx'):
            return target_dir, zeros, zeros, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        cur_goal = self.env_goals[env_ids, self.cur_goal_idx]
        target_vec = cur_goal[:, :2] - self.root_states[:, :2]
        target_dir = target_vec / (torch.norm(target_vec, dim=1, keepdim=True) + 1e-6)
        progress_vel = torch.sum(target_dir * self.root_states[:, 7:9], dim=1)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        stop_cmd_threshold = float(getattr(self.cfg.rewards, 'stop_cmd_threshold', 0.05))
        moving_cmd = cmd_speed > stop_cmd_threshold
        return target_dir, progress_vel, cmd_speed, moving_cmd

    def _reward_goal_progress(self):
        _, vel_to_goal, cmd_speed, _ = self._get_goal_progress_state()

        stop_cmd_threshold = float(getattr(self.cfg.rewards, 'stop_cmd_threshold', 0.05))
        min_goal_speed = float(getattr(self.cfg.rewards, 'min_goal_speed', 0.0))
        max_goal_speed = float(getattr(self.cfg.rewards, 'max_goal_speed', 0.8))
        if bool(getattr(self.cfg.commands, 'use_continuous_speed_curriculum', False)):
            max_goal_speed = min(max_goal_speed, float(self.command_ranges['lin_vel_x'][1]))
        target_speed = torch.clamp(cmd_speed, min=min_goal_speed, max=max_goal_speed)
        target_speed = torch.where(cmd_speed <= stop_cmd_threshold, torch.zeros_like(target_speed), target_speed)
        reward = torch.exp(-torch.square(vel_to_goal - target_speed) / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_goal_bonus(self):
        """Reward for reaching goal waypoints and final goal.

        Uses goal_reach_event (intermediate waypoints) and final_goal_event
        (final goal) which are set in _update_goals().
        """
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        if hasattr(self, 'goal_reach_event'):
            reward += self.goal_reach_event.float()

        if hasattr(self, 'final_goal_event'):
            final_bonus = float(getattr(self.cfg.rewards, 'final_goal_bonus', 5.0))
            reward += final_bonus * self.final_goal_event.float()

        if hasattr(self, 'env_has_goals'):
            reward[~self.env_has_goals] = 0.0

        return reward

    def _mask_invalid_terrain_reward(self, reward):
        if hasattr(self, 'env_terrain_idx'):
            reward[self.env_terrain_idx < 0] = 0.0
        return reward

    def _compute_wheel_lateral_slip(self, wheel_vel_world, wheel_contact):
        """ Compute lateral slip penalty for wheel bodies.

        Args:
            wheel_vel_world: (num_envs, num_wheels, 3) wheel velocities in world frame
            wheel_contact: (num_envs, num_wheels) boolean wheel contact mask
        Returns:
            (num_envs,) lateral slip penalty per environment
        """
        base_quat = self.base_quat.unsqueeze(1).repeat(1, wheel_vel_world.shape[1], 1)
        wheel_vel_body = quat_rotate_inverse(base_quat, wheel_vel_world)
        lateral_vel = wheel_vel_body[..., 1]
        return torch.sum(torch.square(lateral_vel) * wheel_contact.float(), dim=1)

    def _reward_wheel_slip(self):
        if not hasattr(self, 'wheel_body_indices') or not hasattr(self, 'rigid_body_states'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rigid_body_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_vel_world = rigid_body_state[:, self.wheel_body_indices, 7:10]
        wheel_contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0
        reward = self._compute_wheel_lateral_slip(wheel_vel_world, wheel_contact)
        return self._mask_invalid_terrain_reward(reward)

    def _reward_wheel_clearance(self):
        if not hasattr(self, 'wheel_body_indices') or not hasattr(self, 'rigid_body_states'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        rigid_body_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_pos = rigid_body_state[:, self.wheel_body_indices, :3]
        terrain_h = self._get_heights_at_points(wheel_pos[..., :2])
        margin = float(getattr(self.cfg.rewards, 'wheel_clearance_margin', 0.04))
        target_z = terrain_h + float(getattr(self.cfg.asset, 'wheel_radius', 0.05)) + margin
        clearance_error = torch.relu(target_z - wheel_pos[..., 2])

        contact_threshold = float(getattr(self.cfg.rewards, 'contact_force_thresh', 1.0))
        wheel_contact = torch.norm(self.contact_forces[:, self.wheel_body_indices, :], dim=-1) > contact_threshold
        obstacle_gate = self._get_obstacle_ahead_mask().float().unsqueeze(1)
        reward = torch.mean(obstacle_gate * (~wheel_contact).float() * torch.square(clearance_error), dim=1)
        return self._mask_invalid_terrain_reward(reward)

    def _reward_wheel_climb_drive(self):
        if not hasattr(self, 'wheel_dof_indices') or not hasattr(self, 'wheel_body_indices'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        wheel_vel = self.dof_vel[:, self.wheel_dof_indices]
        wheel_forward_sign = getattr(self, 'wheel_forward_sign', torch.ones(len(self.wheel_dof_indices), dtype=torch.float, device=self.device))
        positive_wheel_spin = torch.clamp(wheel_vel * wheel_forward_sign.unsqueeze(0), min=0.0)
        _, progress_vel, _, moving_cmd = self._get_goal_progress_state()
        progress_vel = torch.clamp(progress_vel, min=0.0).unsqueeze(1)
        moving_cmd = moving_cmd.float().unsqueeze(1)
        wheel_contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0

        reward = torch.sum(positive_wheel_spin * progress_vel * moving_cmd * wheel_contact.float(), dim=1)
        reward *= self._get_obstacle_ahead_mask().float()
        return self._mask_invalid_terrain_reward(reward)

    def _reward_wheel_spin_without_progress(self):
        if not hasattr(self, 'wheel_dof_indices'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        wheel_vel = torch.abs(self.dof_vel[:, self.wheel_dof_indices]).mean(dim=1)
        _, progress_vel, _, moving_cmd = self._get_goal_progress_state()
        min_progress_speed = float(getattr(self.cfg.rewards, 'min_progress_speed', 0.05))
        stuck = (progress_vel < min_progress_speed) & moving_cmd
        reward = stuck.float() * torch.square(wheel_vel)
        return self._mask_invalid_terrain_reward(reward)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        if hasattr(self, 'env_has_goals'):
            reward[self.env_has_goals] *= 0.2
        return reward
    
    def _reward_tracking_delta_yaw(self):
        delta_yaw = self.commands[:, 2]
        return 0.5 * (torch.cos(delta_yaw) + 1.0)

    def _reward_yaw_rate_l2(self):
        yaw_rate = self.base_ang_vel[:, 2]
        if self.commands.shape[1] >= 3 and bool(getattr(self.cfg.rewards, 'yaw_rate_gate', False)):
            delta_yaw = torch.abs(self.commands[:, 2])
            gate = torch.clamp(1.0 - delta_yaw / float(getattr(self.cfg.rewards, 'yaw_rate_gate_scale', 0.5)), 0.0, 1.0)
            return gate * torch.square(yaw_rate)
        return torch.square(yaw_rate)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return self._sum_leg_wheel_penalty(
            torch.square(dof_acc),
            self.cfg.rewards.wheel_acc_weight
        )
    
    def _reward_joint_power(self):
        #Penalize high power
        power = torch.abs(self.dof_vel) * torch.abs(self.torques)
        wheel_weight = getattr(self.cfg.rewards, 'wheel_drive_torque_weight', self.cfg.rewards.wheel_torque_weight)
        return self._sum_leg_wheel_penalty(power, wheel_weight)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        base_target = float(getattr(self.cfg.rewards, 'base_height_target', 0.34))
        if hasattr(self, '_get_obstacle_ahead_mask'):
            obstacle_mask = self._get_obstacle_ahead_mask()
        else:
            obstacle_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        target = base_target + float(getattr(self.cfg.rewards, 'obstacle_height_offset', 0.0)) * obstacle_mask.float()
        return torch.square(base_height - target)
    
    def _get_reward_feet_body_indices(self):
        if hasattr(self, 'leg_foot_indices') and self.leg_foot_indices.numel() > 0:
            return self.leg_foot_indices
        return self.feet_indices

    def _get_reward_feet_state(self):
        feet_body_indices = self._get_reward_feet_body_indices()
        if hasattr(self, 'leg_foot_indices') and feet_body_indices is self.leg_foot_indices:
            rigid_body_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
            return rigid_body_state[:, feet_body_indices, 0:3], rigid_body_state[:, feet_body_indices, 7:10]
        return self.feet_pos, self.feet_vel

    def _reward_foot_clearance(self):
        feet_pos, feet_vel = self._get_reward_feet_state()
        cur_footpos_translated = feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, feet_pos.shape[1], 3, device=self.device)
        cur_footvel_translated = feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, feet_pos.shape[1], 3, device=self.device)
        for i in range(feet_pos.shape[1]):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        action_rate = self.last_actions - self.actions
        return self._sum_leg_wheel_penalty(
            torch.square(action_rate),
            self.cfg.rewards.wheel_action_rate_weight
        )
    
    def _reward_torques(self):
        # Penalize torques
        wheel_weight = getattr(self.cfg.rewards, 'wheel_drive_torque_weight', self.cfg.rewards.wheel_torque_weight)
        return self._sum_leg_wheel_penalty(torch.square(self.torques), wheel_weight)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        if not hasattr(self, 'leg_dof_indices'):
            return torch.sum(torch.square(self.dof_vel), dim=1)
        leg_dof_vel = self.dof_vel[:, self.leg_dof_indices]
        return torch.sum(torch.square(leg_dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        failed_reset = self.reset_buf * ~self.time_out_buf
        if hasattr(self, 'final_goal_reset_buf'):
            failed_reset = failed_reset & ~self.final_goal_reset_buf
        return failed_reset
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        if not hasattr(self, 'leg_dof_indices'):
            out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
            out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
            return torch.sum(out_of_limits, dim=1)
        leg_pos = self.dof_pos[:, self.leg_dof_indices]
        leg_limits = self.dof_pos_limits[self.leg_dof_indices]
        out_of_limits = -(leg_pos - leg_limits[:, 0]).clip(max=0.)
        out_of_limits += (leg_pos - leg_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if not hasattr(self, 'leg_dof_indices'):
            return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
        leg_dof_vel = self.dof_vel[:, self.leg_dof_indices]
        leg_dof_vel_limits = self.dof_vel_limits[self.leg_dof_indices]
        return torch.sum((torch.abs(leg_dof_vel) - leg_dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        over_limit = (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.)
        wheel_weight = getattr(self.cfg.rewards, 'wheel_drive_torque_weight', self.cfg.rewards.wheel_torque_weight)
        return self._sum_leg_wheel_penalty(over_limit, wheel_weight)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        feet_body_indices = self._get_reward_feet_body_indices()
        contact = self.contact_forces[:, feet_body_indices, 2] > 1.
        if hasattr(self, 'leg_foot_indices') and feet_body_indices is self.leg_foot_indices:
            contact_filt = torch.logical_or(contact, self.leg_last_contacts)
            self.leg_last_contacts = contact
            first_contact = (self.leg_feet_air_time > 0.) * contact_filt
            self.leg_feet_air_time += self.dt
            rew_airTime = torch.sum((self.leg_feet_air_time - 0.5) * first_contact, dim=1)
            self.leg_feet_air_time *= ~contact_filt
        else:
            contact_filt = torch.logical_or(contact, self.last_contacts)
            self.last_contacts = contact
            first_contact = (self.feet_air_time > 0.) * contact_filt
            self.feet_air_time += self.dt
            rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
            self.feet_air_time *= ~contact_filt
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        feet_body_indices = self._get_reward_feet_body_indices()
        return torch.any(torch.norm(self.contact_forces[:, feet_body_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, feet_body_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        if not hasattr(self, 'leg_dof_indices'):
            dof_error = torch.abs(self.dof_pos - self.default_dof_pos)
        else:
            dof_error = torch.abs(self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices])
        return torch.sum(dof_error, dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        feet_body_indices = self._get_reward_feet_body_indices()
        return torch.sum((torch.norm(self.contact_forces[:, feet_body_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
