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

import os
from typing import Tuple, Any
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: Any, train_cfg: Any):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[Any, Any]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def _reward_name(self, *parts):
        return "".join(parts)

    @property
    def LEGACY_GO2W_REWARD_SCALES(self):
        r = self._reward_name
        return [
            r("tracking_", "lin_vel"),
            r("tracking_", "ang_vel"),
            r("tracking_", "heading"),
            r("tracking_", "goal_", "yaw"),
            r("tracking_", "goal_", "vel"),
            r("tracking_", "goal_", "vel_", "cmd_scaled"),
            r("reach_", "goal"),
            r("finish_", "course"),
            r("wheel_", "lateral_", "slip"),
            r("wheel_", "clearance_", "near_", "obstacle"),
            r("base_", "height_", "over_", "obstacle"),
            r("wheel_", "torque"),
            r("joint_", "power"),
            r("smooth", "ness"),
            r("wheel_", "vel_", "smooth"),
            r("feet_", "air_", "time"),
            r("foot_", "clearance"),
            r("feet_", "clearance"),
            r("feet_", "st", "umble"),
            r("st", "umble"),
            r("feet_", "contact_", "forces"),
        ]

    GO2W_CORE_REWARD_NAMES = [
        "goal_progress",
        "tracking_delta_yaw",
        "goal_bonus",
        "wheel_clearance",
        "wheel_climb_drive",
        "wheel_spin_without_progress",
        "wheel_slip",
        "base_height",
        "orientation",
        "lin_vel_z",
        "ang_vel_xy",
        "yaw_rate_l2",
        "torques",
        "dof_vel",
        "dof_acc",
        "action_rate",
        "dof_pos_limits",
        "hip_action_l2",
        "stand_still",
        "collision",
        "termination",
    ]

    def remove_reward_scale(self, scales, name):
        if name in vars(scales):
            delattr(scales, name)

    def remove_legacy_go2w_reward_scales(self, env_cfg):
        for name in self.LEGACY_GO2W_REWARD_SCALES:
            self.remove_reward_scale(env_cfg.rewards.scales, name)
        # go2W scales inherit many base rewards through class inheritance.
        # Replace the scale container so class_to_dict()/TensorBoard only see
        # the explicit core rewards set by the current stage.
        env_cfg.rewards.scales = type("Go2WCoreRewardScales", (), {})
        return env_cfg

    def apply_go2w_stage0_scales(self, env_cfg):
        env_cfg = self.remove_legacy_go2w_reward_scales(env_cfg)
        scales = env_cfg.rewards.scales

        scales.goal_progress = 2.0
        scales.tracking_delta_yaw = 0.5
        scales.goal_bonus = 0.0

        scales.base_height = -0.5
        scales.orientation = -1.0
        scales.lin_vel_z = -2.0
        scales.ang_vel_xy = -0.05
        scales.yaw_rate_l2 = -0.01

        scales.torques = -1e-5
        scales.dof_vel = -1e-4
        scales.dof_acc = -2.5e-7
        scales.action_rate = -0.01
        scales.dof_pos_limits = -0.9
        scales.hip_action_l2 = -0.1

        scales.wheel_slip = -0.1
        scales.stand_still = -0.01
        scales.collision = -0.5
        scales.termination = -0.8

        scales.wheel_clearance = 0.0
        scales.wheel_climb_drive = 0.0
        scales.wheel_spin_without_progress = 0.0

        env_cfg.rewards.min_goal_speed = 0.0
        env_cfg.rewards.max_goal_speed = 0.6
        env_cfg.rewards.stop_cmd_threshold = 0.05
        env_cfg.rewards.final_goal_bonus = 0.0
        env_cfg.rewards.obstacle_height_offset = 0.0
        env_cfg.rewards.obstacle_height_threshold = 0.04
        env_cfg.rewards.gap_height_threshold = 0.06
        env_cfg.rewards.wheel_clearance_target = 0.08
        env_cfg.asset.penalize_contacts_on = ["base", "trunk", "thigh", "calf"]
        env_cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        return env_cfg

    def apply_go2w_stage2_scales(self, env_cfg):
        env_cfg = self.remove_legacy_go2w_reward_scales(env_cfg)
        scales = env_cfg.rewards.scales

        scales.goal_progress = 3.0
        scales.tracking_delta_yaw = 0.8
        scales.goal_bonus = 1.5

        scales.wheel_clearance = 1.0
        scales.wheel_climb_drive = 0.1
        scales.wheel_spin_without_progress = -0.02

        scales.base_height = -0.25
        scales.orientation = -0.2
        scales.lin_vel_z = -0.8
        scales.ang_vel_xy = -0.03
        scales.yaw_rate_l2 = -0.02

        scales.torques = -1e-5
        scales.dof_vel = -5e-5
        scales.dof_acc = -1e-7
        scales.action_rate = -0.005
        scales.dof_pos_limits = -0.5
        scales.hip_action_l2 = -0.05

        scales.wheel_slip = -0.3
        scales.stand_still = -0.01
        scales.collision = -0.3
        scales.termination = -0.8

        env_cfg.rewards.min_goal_speed = 0.15
        env_cfg.rewards.max_goal_speed = 0.8
        env_cfg.rewards.stop_cmd_threshold = 0.05
        env_cfg.rewards.final_goal_bonus = 5.0
        env_cfg.rewards.obstacle_height_offset = 0.06
        env_cfg.rewards.obstacle_height_threshold = 0.04
        env_cfg.rewards.gap_height_threshold = 0.06
        env_cfg.rewards.wheel_clearance_target = 0.10
        env_cfg.rewards.only_positive_rewards = False
        env_cfg.asset.penalize_contacts_on = ["base", "trunk", "thigh", "calf"]
        env_cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]

        print("[go2w stage2 core reward scales]")
        for name in self.GO2W_CORE_REWARD_NAMES:
            if hasattr(scales, name):
                print(f"  {name}: {getattr(scales, name)}")
        print(f"  penalize_contacts_on: {env_cfg.asset.penalize_contacts_on}")
        print(f"  terminate_after_contacts_on: {env_cfg.asset.terminate_after_contacts_on}")
        return env_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, Any]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        if getattr(args, "web", False) and hasattr(env_cfg, "viewer"):
            env_cfg.viewer.web = True
        if getattr(args, "stage", None) is not None:
            env_cfg, _ = self.set_stage(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def set_stage(self, env_cfg, train_cfg, args):
        stage = getattr(args, "stage", None)
        if stage is None:
            return env_cfg, train_cfg

        unsupported_depth_stages = (3, 5)
        if stage in unsupported_depth_stages:
            raise NotImplementedError(
                f"Parkour stage {stage} requires depth camera / depth_encoder support, "
                "which has not been migrated to HIMLoco-W."
            )

        terrain_stage_props = {
            0: {'parkour_flat': 1.0},
            1: {'parkour_hurdle': 2.0, 'parkour_flat': 1.5, 'parkour_step': 1.0, 'parkour_gap': 1.0},
            2: {
                'parkour_hurdle': 0.2, 'parkour_flat': 0.2, 'parkour_step': 0., 'parkour_gap': 0., 'parkour_wall': 0.4,
                'T_step_stl': 0.4, 'Slope': 0.4, 'BridgeA': 0.4, 'BridgeB': 0.,
            },
            4: {'parkour_flat': 1.0},
        }
        train_iterations = {0: 7000, 1: 15000, 2: 10000, 4: 15000}
        if stage not in terrain_stage_props:
            raise ValueError(f"Unsupported parkour stage {stage}. Supported HIM-compatible stages are 0, 1, 2, and 4.")

        if env_cfg is not None:
            print("Set env learning stage to {}".format(stage))
            env_cfg.terrain.terrain_proportions = [0.] * 8
            extra_props = dict(getattr(env_cfg.terrain, 'terrain_extra_proportions', {}) or {})
            for name in extra_props:
                extra_props[name] = 0.
            for name, value in terrain_stage_props[stage].items():
                if name not in extra_props:
                    raise KeyError(f"Stage {stage} terrain '{name}' is not defined in terrain_extra_proportions")
                extra_props[name] = value
            env_cfg.terrain.terrain_extra_proportions = extra_props

            if stage == 0 and getattr(env_cfg.asset, 'name', '') == 'go2w':
                env_cfg = self.apply_go2w_stage0_scales(env_cfg)
                env_cfg.commands.ranges.lin_vel_x = [0.0, 0.8]
            elif stage == 4 and getattr(env_cfg.asset, 'name', '') == 'go2w':
                env_cfg = self.apply_go2w_stage2_scales(env_cfg)
                env_cfg.rewards.scales.goal_progress = 2.0
                env_cfg.rewards.scales.tracking_delta_yaw = 0.8
                env_cfg.rewards.scales.goal_bonus = 1.5
                env_cfg.rewards.tracking_sigma = 0.05
                env_cfg.commands.ranges.lin_vel_x = [0.0, 0.8]
            elif stage == 2 and getattr(env_cfg.asset, 'name', '') == 'go2w':
                env_cfg = self.apply_go2w_stage2_scales(env_cfg)
                env_cfg.commands.ranges.lin_vel_x = [0.0, 1.5]

        if train_cfg is not None:
            print("Set train learning stage to {}".format(stage))
            if getattr(args, "max_iterations", None) is None:
                train_cfg.runner.max_iterations = train_iterations[stage]

        return env_cfg, train_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, Any]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        if getattr(args, "stage", None) is not None:
            _, train_cfg = self.set_stage(None, train_cfg, args)

        run_name = train_cfg.runner.run_name if train_cfg.runner.run_name else 'default'
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
