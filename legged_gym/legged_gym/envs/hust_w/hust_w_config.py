# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class HUSTWRoughCfg(LeggedRobotCfg):
    class env(GO2WRoughCfg.env):
        pass

    class terrain(GO2WRoughCfg.terrain):
        pass

    class commands(GO2WRoughCfg.commands):
        pass

    class init_state(GO2WRoughCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        default_joint_angles = {
            'FR_HIP_JOINT': -0.1,
            'FR_THIGH_JOINT': 0.8,
            'FR_CALF_JOINT': 1.8,
            'FR_WHEEL_JOINT': 0.0,
            'FL_HIP_JOINT': 0.1,
            'FL_THIGH_JOINT': 0.8,
            'FL_CALF_JOINT': 1.8,
            'FL_WHEEL_JOINT': 0.0,
            'LR_HIP_JOINT': -0.1,
            'LR_THIGH_JOINT': 0.8,
            'LR_CALF_JOINT': 1.6,
            'LR_WHEEL_JOINT': 0.0,
            'LL_HIP_JOINT': 0.1,
            'LL_THIGH_JOINT': 0.6,
            'LL_CALF_JOINT': 1.8,
            'LL_WHEEL_JOINT': 0.0,
        }
        init_joint_angles = default_joint_angles.copy()

    class control(GO2WRoughCfg.control):
        stiffness = {'HIP_JOINT': 40., 'THIGH_JOINT': 40., 'CALF_JOINT': 40., 'WHEEL_JOINT': 40.}
        damping = {'HIP_JOINT': 1., 'THIGH_JOINT': 1., 'CALF_JOINT': 1., 'WHEEL_JOINT': 0.5}

    class asset(GO2WRoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hust_w/urdf/hust_w.urdf'
        name = 'hust_w'
        foot_name = 'WHEEL'
        wheel_name = ['WHEEL_JOINT']
        wheel_radius = 0.111
        leg_dof_names = [
            'FR_HIP_JOINT', 'FR_THIGH_JOINT', 'FR_CALF_JOINT',
            'FL_HIP_JOINT', 'FL_THIGH_JOINT', 'FL_CALF_JOINT',
            'LR_HIP_JOINT', 'LR_THIGH_JOINT', 'LR_CALF_JOINT',
            'LL_HIP_JOINT', 'LL_THIGH_JOINT', 'LL_CALF_JOINT',
        ]
        wheel_dof_names = [
            'FR_WHEEL_JOINT', 'FL_WHEEL_JOINT', 'LR_WHEEL_JOINT', 'LL_WHEEL_JOINT',
        ]
        wheel_body_names = [
            'FR_WHEEL', 'FL_WHEEL', 'LR_WHEEL', 'LL_WHEEL',
        ]
        wheel_forward_sign = [-1.0, 1.0, -1.0, 1.0]
        penalize_contacts_on = ['THIGH', 'CALF', 'trunk']
        terminate_after_contacts_on = []
        self_collisions = 1
        collapse_fixed_joints = False
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class domain_rand(GO2WRoughCfg.domain_rand):
        pass

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.4
        yaw_rate_gate = True
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.40
        max_contact_force = 100.
        wheel_drive_torque_weight = 0.5
        wheel_acc_weight = 0.2
        wheel_action_rate_weight = 0.5
        wheel_clearance_target = 0.111
        wheel_clearance_margin = 0.04
        contact_force_thresh = 1.0
        obstacle_height_offset = 0.06
        wheel_spin_progress_threshold = 0.05
        min_progress_speed = 0.05
        min_goal_speed = 0.2
        max_goal_speed = 0.8
        stop_cmd_threshold = 0.05
        final_goal_bonus = 5.0
        goal_dist_thresh = 0.3
        goal_yaw_thresh = 0.4
        obstacle_height_threshold = 0.06
        gap_height_threshold = 0.06
        obstacle_probe_distances = [0.25, 0.40, 0.55, 0.70]
        heading_sigma = 0.25
        delta_yaw_sigma = 0.25
        use_fixed_goal_speed = False
        fixed_goal_speed = 0.4
        # reward_align_stage2 = False
        use_delta_goal_progress = False
        success_bonus_once = True
        use_time_penalty = False
        reduce_alive_reward_stage2 = False
        log_reward_terms_detail = False
        goal_progress_delta_max = 1.0
        wheel_slip_scale_multiplier = 1.0
        healthy_yaw_threshold = 0.6
        healthy_base_height_min = 0.18
        healthy_base_height_max = 0.65
        healthy_roll_pitch_proxy_max = 0.65
        healthy_wheel_slip_max = 0.8

        class scales:
            termination = -0.8
            goal_progress = 1.0
            goal_delta_progress = 2.0
            tracking_delta_yaw = 1.0
            delta_yaw_progress = 0.0
            goal_bonus = 3.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -0.5
            yaw_rate_l2 = -0.01
            torques = -1e-5
            action_rate = -0.01
            dof_pos_limits = -0.9
            dof_vel = -5e-5
            dof_acc = 0.0
            hip_action_l2 = -0.05
            wheel_slip = -0.1
            stand_still = 0.0
            collision = -0.5
            wheel_clearance = 0.0
            wheel_climb_drive = 0.0
            wheel_spin_without_progress = -1e-4


class HUSTWRoughCfgPPO(GO2WRoughCfgPPO):
    class runner(GO2WRoughCfgPPO.runner):
        run_name = 'hust_w_cad_go2w_baseline'
        experiment_name = 'rough_hust_w'
