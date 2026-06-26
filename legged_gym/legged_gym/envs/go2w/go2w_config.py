# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2WRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 6000
        num_actions = 16
        num_one_step_observations = 73
        num_observations = num_one_step_observations * 5
        num_one_step_privileged_obs = 263
        num_privileged_obs = 263

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 3
        resampling_time = 10.
        heading_command = False
        use_goal_yaw_command = True
        goal_yaw_fallback_random = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0, 5]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [0, 0]  # legacy unused yaw-rate range
            heading = [-3.1416, 3.1416]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 18.
        terrain_width = 4.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0]
        terrain_extra_proportions = {
            'parkour': .0,
            'parkour_hurdle': 0.,
            'parkour_flat': 0.,
            'parkour_step': 0.,
            'parkour_gap': 0.,
            'T_step_stl': 0.,
            'Slope': 0.,
            'BridgeA': 0.,
            'BridgeB': 0.,
            'parkour_wall': 0.,
        }
        slope_treshold = 0.75
        use_parkour_goals = True
        num_goals = 8
        num_future_goal_obs = 2
        next_goal_threshold = 0.2
        goal_reach_delay = 0.1
        terminate_after_reaching_final_goal = True
        parkour_terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        randomize_terrain_on_reset = True
        randomize_terrain_levels_on_reset = True
        randomize_terrain_types_on_reset = True
        random_difficulty_range = [0.0, 1.0]
        y_range = [-0.4, 0.4]
        hust_terrain = True
        slim_hurdle = False
        height = [0.02, 0.06]
        downsampled_scale = 0.075

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.67,
            'RL_thigh_joint': 0.67,
            'FR_thigh_joint': 0.67,
            'RR_thigh_joint': 0.67,
            'FL_calf_joint': -1.3,
            'RL_calf_joint': -1.3,
            'FR_calf_joint': -1.3,
            'RR_calf_joint': -1.3,
            'FL_foot_joint': 0.0,
            'RL_foot_joint': 0.0,
            'FR_foot_joint': 0.0,
            'RR_foot_joint': 0.0,
        }
        init_joint_angles = default_joint_angles.copy()

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_joint': 50., 'thigh_joint': 50., 'calf_joint': 50., 'foot_joint': 40.}
        damping = {'hip_joint': 1., 'thigh_joint': 1., 'calf_joint': 1., 'foot_joint': 0.5}
        action_scale = 0.25
        vel_scale = 10.0
        decimation = 4
        wheel_speed = 1
        hip_reduction = 1.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf'
        name = 'go2w'
        foot_name = 'foot'
        wheel_name = ['foot_joint']
        penalize_contacts_on = ['thigh', 'calf', 'base']
        terminate_after_contacts_on = []
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_link_mass = False
        randomize_friction = True
        friction_range = [0.25, 1.25]
        randomize_restitution = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = False
        disturbance = False
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        delay = False
        randomize_base_mass = False
        added_mass_range = [-1., 1.]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.4
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.34
        max_contact_force = 100.
        wheel_drive_torque_weight = 0.5
        wheel_acc_weight = 0.2
        wheel_action_rate_weight = 0.5
        wheel_clearance_target = 0.10
        obstacle_height_offset = 0.06
        heading_sigma = 0.25
        delta_yaw_sigma = 0.25
        min_goal_speed = 0.2
        max_goal_speed = 0.8
        stop_cmd_threshold = 0.05
        final_goal_bonus = 5.0
        use_fixed_goal_speed = False
        fixed_goal_speed = 0.4
        obstacle_height_threshold = 0.06
        gap_height_threshold = 0.06
        obstacle_probe_distances = [0.25, 0.40, 0.55, 0.70]

        class scales:
            termination = -0.8
            goal_progress = 2.0
            tracking_delta_yaw = 0.5
            goal_bonus = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -0.5
            yaw_rate_l2 = -0.01
            torques = -1e-5
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -0.9
            hip_action_l2 = -0.1
            wheel_slip = -0.1
            stand_still = -0.01
            collision = -0.5
            wheel_clearance = 0.0
            wheel_climb_drive = 0.0
            wheel_spin_without_progress = 0.0


class GO2WRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        run_name = '50_1_40_0.5_stair_up_privileged_frac=0.8'
        experiment_name = 'rough_go2w'
        num_steps_per_env = 48
        max_iterations = 30000
        load_run = -1
        checkpoint = -1
