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

    GO2W_ACTIVE_REWARD_NAMES = [
        "goal_progress",
        "goal_progress_delta",
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

    @property
    def GO2W_LEGACY_REWARD_NAMES(self):
        return self.LEGACY_GO2W_REWARD_SCALES + [
            "success_bonus",
            "early_success",
            "time_penalty",
        ]

    def remove_reward_scale(self, scales, name):
        if name in vars(scales):
            delattr(scales, name)

    def remove_legacy_go2w_reward_scales(self, env_cfg):
        for name in self.GO2W_LEGACY_REWARD_NAMES:
            self.remove_reward_scale(env_cfg.rewards.scales, name)
        # go2W scales inherit many base rewards through class inheritance.
        # Replace the scale container so class_to_dict()/TensorBoard only see
        # the explicit core rewards set by the current stage.
        env_cfg.rewards.scales = type("Go2WCoreRewardScales", (), {})
        return env_cfg

    TERRAIN_PROFILES = {
        0: {'parkour_flat': 1.0},
        1: {'parkour_hurdle': 2.0, 'parkour_flat': 1.5, 'parkour_step': 1.0, 'parkour_gap': 1.0},
        2: {
            'parkour_hurdle': 0.2,
            'parkour_flat': 0.2,
            'parkour_step': 0.0,
            'parkour_gap': 0.0,
            'parkour_wall': 0.4,
            'T_step_stl': 0.4,
            'Slope': 0.4,
            'BridgeA': 0.4,
            'BridgeB': 0.0,
        },
        4: {'parkour_flat': 1.0},
    }

    TRAIN_ITERATIONS = {0: 7000, 1: 15000, 2: 10000, 4: 15000}

    GO2W_STAGE_PROFILES = {
        0: ("stage0", "stage0"),
        1: ("stage1", "stage1"),
        2: ("stage2", "stage2"),
        4: ("stage4", "stage4"),
    }

    GO2W_REWARD_PROFILES = {
        "stage0": {
            "scales": {
                "goal_progress": 2.0,  # 基础速度跟踪：奖励沿当前目标方向达到命令速度。
                "goal_progress_delta": 0.0,  # 关闭距离差分推进；stage0 不要求按路点持续推进。
                "tracking_delta_yaw": 0.5,  # 弱朝向对齐：鼓励机身朝向当前目标方向。
                "goal_bonus": 0.0,  # 关闭到点事件奖励，避免平地基础阶段刷 waypoint bonus。
                "wheel_clearance": 0.0,  # 关闭轮子抬高奖励；stage0 不训练越障动作。
                "wheel_climb_drive": 0.0,  # 关闭障碍附近轮子驱动奖励；stage0 只学基础行驶。
                "wheel_spin_without_progress": 0.0,  # 关闭空转惩罚；基础阶段先避免过多轮足约束。
                "wheel_slip": -0.1,  # 轻惩罚轮子横向滑移，减少侧滑但不过度限制探索。
                "base_height": -0.5,  # 惩罚基座高度偏离目标高度，保持基础站姿。
                "orientation": -1.0,  # 惩罚 roll/pitch 倾斜，平地阶段保持身体稳定。
                "lin_vel_z": -2.0,  # 惩罚垂直速度，抑制跳动和弹跳。
                "ang_vel_xy": -0.05,  # 惩罚 roll/pitch 角速度，减少身体晃动。
                "yaw_rate_l2": -0.01,  # 轻惩罚 yaw 角速度，避免无意义快速自旋。
                "torques": -1e-5,  # 轻能耗惩罚，限制关节/轮子力矩过大。
                "dof_vel": -1e-4,  # 惩罚腿部关节速度，提升动作平滑性。
                "dof_acc": -2.5e-7,  # 惩罚关节加速度，降低抖动。
                "action_rate": -0.01,  # 惩罚连续 action 差分，减少控制突变。
                "dof_pos_limits": -0.9,  # 惩罚腿部关节接近限位，保护可行姿态空间。
                "hip_action_l2": -0.1,  # 抑制髋关节大幅动作，减少外摆和不稳定姿态。
                "stand_still": -0.01,  # 零/低速命令下惩罚腿部偏离默认姿态。
                "collision": -0.5,  # 惩罚非允许部位接触，减少躯干/腿部碰撞。
                "termination": -0.8,  # 失败终止惩罚，区分正常超时和跌倒/碰撞终止。
            },
            "params": {
                "reward_align_stage2": False,  # 不计算 stage2 专用的路点距离差分缓存。
                "use_delta_goal_progress": False,  # `_reward_goal_progress_delta` 即使存在也返回 0。
                "min_goal_speed": 0.0,  # 允许低速/静止命令，适合基础跟踪预训练。
                "max_goal_speed": 2.0,  # 限制 reward 中目标速度上界，低于命令采样最大值以稳定早期训练。
                "stop_cmd_threshold": 0.05,  # 命令速度低于该值时按停止目标处理。
                "final_goal_bonus": 0.0,  # 关闭最终路点额外奖励。
                "obstacle_height_offset": 0.0,  # 不因前方障碍提高基座高度目标。
                "obstacle_height_threshold": 0.04,  # 前方高度升高超过该值才算障碍；stage0 仅保留默认阈值。
                "gap_height_threshold": 0.06,  # 前方高度下降超过该值才算 gap；stage0 不启用相关奖励。
                "wheel_clearance_target": 0.08,  # 轮子离地目标高度；stage0 scale 为 0，仅保留默认值。
                "only_positive_rewards": True,  # 裁剪总 reward 到非负，降低基础阶段早期崩溃风险。
            },
        },
        "stage1": {
            "inherits": "stage0",
            "scales": {
                "wheel_slip": -0.15,
            },
        },
        "stage2": {
            "scales": {
                "goal_progress": 0.5,  # 辅助速度约束：保持朝目标速度合理，不作为主推进信号。
                "goal_progress_delta": 8.0,  # 主推进奖励：按当前步到目标距离减少量给奖。
                "tracking_delta_yaw": 0.8,  # 强化朝向目标，帮助复杂地形上对准障碍入口。
                "goal_bonus": 2.0,  # 一次性路点到达奖励，配合 final_goal_bonus 奖励完成整段路线。
                "wheel_clearance": 1.0,  # 障碍附近奖励轮子达到目标离地高度，辅助跨越台阶/墙/gap。
                "wheel_climb_drive": 0.1,  # 障碍附近奖励接触轮正向驱动且产生目标方向进展。
                "wheel_spin_without_progress": -0.02,  # 障碍附近惩罚轮子高速转但目标方向进展不足。
                "wheel_slip": -0.3,  # 强惩罚轮子横向滑移，提升复杂地形牵引稳定性。
                "base_height": -0.25,  # 弱化基座高度惩罚，允许越障时抬高或压低身体。
                "orientation": -0.2,  # 弱化姿态惩罚，避免越障动作被过度压制。
                "lin_vel_z": -0.8,  # 保留垂直速度惩罚，但比 stage0 弱以允许上台阶/落地。
                "ang_vel_xy": -0.03,  # 轻惩罚 roll/pitch 角速度，允许必要的身体摆动。
                "yaw_rate_l2": -0.02,  # 惩罚无效快速转向，避免在障碍前抖动自旋。
                "torques": -1e-5,  # 保持能耗正则，防止靠极端力矩过障。
                "dof_vel": -5e-5,  # 比 stage0 更弱的腿部速度惩罚，给越障动作留自由度。
                "dof_acc": -1e-7,  # 比 stage0 更弱的加速度惩罚，减少对快速调整的限制。
                "action_rate": -0.005,  # 比 stage0 更弱的 action 平滑惩罚，允许越障时快速修正。
                "dof_pos_limits": -0.5,  # 保留限位保护，但弱于 stage0 以允许大幅姿态变化。
                "hip_action_l2": -0.05,  # 轻惩罚髋关节大动作，避免越障时过度外摆。
                "stand_still": 0.0,  # 关闭低速站立惩罚，避免与持续路点推进目标冲突。
                "collision": -0.3,  # 碰撞惩罚弱于 stage0，避免越障探索被过早压死。
                "termination": -0.8,  # 失败终止惩罚保持不变。
            },
            "params": {
                "reward_align_stage2": True,  # 启用路点距离差分和一次性完成事件缓存。
                "use_delta_goal_progress": True,  # 允许 `_reward_goal_progress_delta` 输出正向距离进展。
                "goal_progress_delta_max": 1.0,  # 限制单步距离差分上界，防止 reset/瞬移造成异常大奖励。
                "min_goal_speed": 0.15,  # 移动命令下 reward 目标速度下限，避免慢挪刷进展。
                "max_goal_speed": 4.0,  # reward 目标速度上限，与连续速度课程目标一致。
                "stop_cmd_threshold": 0.05,  # 命令速度低于该值时视为停止，轮子驱动/空转项也按停止处理。
                "final_goal_bonus": 5.0,  # 到达最终路点时在 goal_bonus 基础上追加的一次性奖励。
                "obstacle_height_offset": 0.06,  # 前方检测到障碍时提高基座高度目标，辅助上障碍。
                "obstacle_height_threshold": 0.04,  # 前方高度升高超过该值判定为 step/wall 类障碍。
                "gap_height_threshold": 0.06,  # 前方高度下降超过该值判定为 gap 类障碍。
                "wheel_clearance_target": 0.10,  # stage2 轮子越障离地目标高度，高于基础阶段默认值。
                "only_positive_rewards": False,  # 保留负奖励，确保滑移/碰撞/空转等约束真实生效。
            },
        },
        "stage4": {
            "inherits": "stage2",
            "scales": {
                "goal_progress": 2.0,
                "goal_progress_delta": 0.0,
                "tracking_delta_yaw": 0.8,
                "goal_bonus": 1.5,
                "stand_still": -0.01,
                "wheel_clearance": 0.0,
                "wheel_climb_drive": 0.0,
                "wheel_spin_without_progress": 0.0,
            },
            "params": {
                "reward_align_stage2": False,
                "use_delta_goal_progress": False,
                "tracking_sigma": 0.05,
                "min_goal_speed": 0.0,
                "max_goal_speed": 0.8,
                "obstacle_height_offset": 0.0,
                "wheel_clearance_target": 0.08,
            },
        },
    }

    GO2W_SPEED_PROFILES = {
        "stage0": {
            "lin_vel_x": [0.0, 3.0],  # 命令采样前进速度范围；reward 侧再用 max_goal_speed 限制目标速度。
            "lin_vel_y": [0.0, 0.0],  # 关闭横向速度命令，训练轮足机器人主要前向行驶。
            "use_continuous_speed_curriculum": False,  # 基础阶段不启用速度课程，直接覆盖完整采样范围。
        },
        "stage1": {
            "lin_vel_x": [0.0, 3.0],
            "lin_vel_y": [0.0, 0.0],
            "use_continuous_speed_curriculum": False,
        },
        "stage2": {
            "lin_vel_x": [0.0, 3.0],  # 初始命令采样范围；实际上界会被连续速度课程动态覆盖。
            "lin_vel_y": [0.0, 0.0],  # 复杂地形仍关闭横向速度命令，避免目标推进语义混乱。
            "use_continuous_speed_curriculum": True,  # 根据训练表现逐步提高命令速度上限。
            "curriculum_start_speed": 2.0,  # 速度课程起点；训练初期 command_ranges['lin_vel_x'][1] 被设为该值。
            "curriculum_goal_speed": 4.0,  # 速度课程最终目标上限。
            "curriculum_target_speed": 4.0,  # 兼容旧字段；与 curriculum_goal_speed 保持一致。
            "curriculum_min_ratio": 0.0,  # 速度课程比例下限。
            "curriculum_max_ratio": 1.0,  # 速度课程比例上限。
            "curriculum_success_high": 0.75,  # success 指标模式下的提速阈值；当前 progress 模式不使用。
            "curriculum_success_low": 0.35,  # success 指标模式下的降速阈值；当前 progress 模式不使用。
            "curriculum_update_interval": 100,  # 每隔多少 policy step 更新一次速度课程。
            "curriculum_increase_step": 0.03,  # 达到提速条件时课程比例增加量。
            "curriculum_decrease_step": 0.01,  # 允许回退时课程比例下降量。
            "curriculum_hold_on_drop": True,  # 指标变差时保持当前速度，不主动降速。
            "curriculum_metric": "progress",  # 使用距离进展/中间路点达成率驱动速度课程，而不是最终成功率。
            "curriculum_progress_high": 0.045,  # 平均正向距离进展超过该值时提速。
            "curriculum_progress_low": 0.015,  # 平均正向距离进展低于该值时可触发降速逻辑。
            "curriculum_intermediate_goal_high": 0.25,  # 中间路点达成率超过该值也允许提速。
        },
        "stage4": {
            "lin_vel_x": [0.0, 0.8],
            "lin_vel_y": [0.0, 0.0],
            "use_continuous_speed_curriculum": False,
        },
    }

    def _resolve_profile(self, profiles, name):
        profile = dict(profiles[name])
        parent_name = profile.pop("inherits", None)
        if parent_name is None:
            return profile
        parent = self._resolve_profile(profiles, parent_name)
        for key, value in profile.items():
            if isinstance(value, dict) and isinstance(parent.get(key), dict):
                merged = dict(parent[key])
                merged.update(value)
                parent[key] = merged
            else:
                parent[key] = value
        return parent

    def apply_terrain_profile(self, env_cfg, stage):
        env_cfg.terrain.terrain_proportions = [0.] * 8
        extra_props = dict(getattr(env_cfg.terrain, 'terrain_extra_proportions', {}) or {})
        for name in extra_props:
            extra_props[name] = 0.
        for name, value in self.TERRAIN_PROFILES[stage].items():
            if name not in extra_props:
                raise KeyError(f"Stage {stage} terrain '{name}' is not defined in terrain_extra_proportions")
            extra_props[name] = value
        env_cfg.terrain.terrain_extra_proportions = extra_props
        return env_cfg

    def apply_go2w_reward_profile(self, env_cfg, profile_name):
        env_cfg = self.remove_legacy_go2w_reward_scales(env_cfg)
        scales = env_cfg.rewards.scales
        profile = self._resolve_profile(self.GO2W_REWARD_PROFILES, profile_name)
        for name, value in profile.get("scales", {}).items():
            setattr(scales, name, value)
        for name, value in profile.get("params", {}).items():
            setattr(env_cfg.rewards, name, value)
        env_cfg.asset.penalize_contacts_on = ["base", "trunk", "thigh", "calf"]
        env_cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
        print(f"[go2w {profile_name} active reward scales]")
        for name in self.GO2W_ACTIVE_REWARD_NAMES:
            if hasattr(scales, name):
                print(f"  {name}: {getattr(scales, name)}")
        print(f"  penalize_contacts_on: {env_cfg.asset.penalize_contacts_on}")
        print(f"  terminate_after_contacts_on: {env_cfg.asset.terminate_after_contacts_on}")
        return env_cfg

    def apply_go2w_speed_profile(self, env_cfg, profile_name):
        profile = self.GO2W_SPEED_PROFILES[profile_name]
        env_cfg.commands.ranges.lin_vel_x = list(profile["lin_vel_x"])
        env_cfg.commands.ranges.lin_vel_y = list(profile["lin_vel_y"])
        for name, value in profile.items():
            if name in ("lin_vel_x", "lin_vel_y"):
                continue
            setattr(env_cfg.commands, name, value)
        return env_cfg

    def apply_go2w_stage0_scales(self, env_cfg):
        return self.apply_go2w_reward_profile(env_cfg, "stage0")

    def apply_go2w_stage2_scales(self, env_cfg):
        return self.apply_go2w_reward_profile(env_cfg, "stage2")
    
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

        if stage not in self.TERRAIN_PROFILES:
            raise ValueError(f"Unsupported parkour stage {stage}. Supported HIM-compatible stages are 0, 1, 2, and 4.")

        if env_cfg is not None:
            print("Set env learning stage to {}".format(stage))
            env_cfg = self.apply_terrain_profile(env_cfg, stage)

            if getattr(env_cfg.asset, 'name', '') == 'go2w':
                reward_profile, speed_profile = self.GO2W_STAGE_PROFILES[stage]
                env_cfg = self.apply_go2w_reward_profile(env_cfg, reward_profile)
                env_cfg = self.apply_go2w_speed_profile(env_cfg, speed_profile)

        if train_cfg is not None:
            print("Set train learning stage to {}".format(stage))
            if getattr(args, "max_iterations", None) is None:
                train_cfg.runner.max_iterations = self.TRAIN_ITERATIONS[stage]

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
        resume_path = None
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        runner.resume_path = resume_path
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
