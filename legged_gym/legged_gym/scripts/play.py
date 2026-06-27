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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils.cuda_compat import check_cuda_runtime_compat
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils import webviewer

import numpy as np
import torch


def _sample_play_x_commands(env, x_range):
    low, high = float(x_range[0]), float(x_range[1])
    return low + (high - low) * torch.rand(env.num_envs, device=env.device)


def _apply_play_commands(env, x_vel, y_vel, delta_yaw=0.0, env_ids=None):
    if env_ids is None:
        env.commands[:, 0] = x_vel if torch.is_tensor(x_vel) else float(x_vel)
        env.commands[:, 1] = y_vel
    else:
        env.commands[env_ids, 0] = x_vel[env_ids] if torch.is_tensor(x_vel) else float(x_vel)
        env.commands[env_ids, 1] = y_vel
    if hasattr(env, '_update_goal_commands'):
        env._update_goal_commands()
    else:
        env.commands[:, 2] = delta_yaw


def _sync_play_command_obs(env):
    if not hasattr(env, 'obs_buf') or not hasattr(env, 'commands_scale'):
        return env.get_observations()

    one_step_obs = int(getattr(env, 'num_one_step_obs', 0))
    if one_step_obs <= 9 or env.obs_buf.shape[1] < 9:
        return env.get_observations()

    command_obs = env._get_command_obs() if hasattr(env, '_get_command_obs') else env.commands[:, :3] * env.commands_scale
    for start in range(0, env.obs_buf.shape[1], one_step_obs):
        if start + 9 <= env.obs_buf.shape[1]:
            env.obs_buf[:, start + 6:start + 9] = command_obs
    return env.get_observations()


def play(args, x_vel=1.0, y_vel=0.0, delta_yaw=0.0, x_vel_range=(1.0, 3.0)):
    check_cuda_runtime_compat()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    # env_cfg.terrain.mesh_type = 'plane'
    # prepare environment
    env_cfg.commands.ranges.lin_vel_x = [float(x_vel_range[0]), float(x_vel_range[1])]
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if args.web:
        web_viewer = webviewer.WebViewer()
        web_viewer.setup(env)
    play_x_vel = _sample_play_x_commands(env, x_vel_range)
    _apply_play_commands(env, play_x_vel, y_vel, delta_yaw)
    print(f"[play] sampling command x velocity in [{x_vel_range[0]}, {x_vel_range[1]}] m/s")
    env.compute_observations()

    obs = _sync_play_command_obs(env)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
    
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        done_env_ids = dones.nonzero(as_tuple=False).flatten() if torch.is_tensor(dones) else torch.tensor([], device=env.device, dtype=torch.long)
        if len(done_env_ids) > 0:
            play_x_vel[done_env_ids] = _sample_play_x_commands(env, x_vel_range)[done_env_ids]
        _apply_play_commands(env, play_x_vel, y_vel, delta_yaw)
        obs = _sync_play_command_obs(env)

        if args.web:
            web_viewer.render(fetch_results=True,
                              step_graphics=True,
                              render_all_camera_sensors=True,
                              wait_for_page_load=True)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'delta_yaw_cmd': env.commands[robot_index, 2].item(),
                    'current_yaw': env._get_base_yaw()[robot_index].item() if hasattr(env, '_get_base_yaw') else 0.0,
                    'heading_error': env._get_heading_error()[robot_index].item() if hasattr(env, '_get_heading_error') else 0.0,
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=1.0, y_vel=0.0, delta_yaw=0.0, x_vel_range=(1.0, 3.0))
