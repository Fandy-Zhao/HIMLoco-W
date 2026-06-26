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
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        use_goal_yaw_command = True
        goal_yaw_fallback_random = True
        goal_yaw_kp = 0.5
        goal_yaw_rate_clip = 2.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0, 5]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [0, 0]
            heading = [-3.14, 3.14]

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
        reach_goal_delay = 0.1
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
        wheel_torque_weight = 0.5
        wheel_acc_weight = 0.2
        wheel_action_rate_weight = 0.5
        wheel_clearance_target = 0.10
        obstacle_height_offset = 0.06

        class scales(LeggedRobotCfg.rewards.scales):
            # Stage 0 is meant to learn stable flat-ground go2w locomotion first.
            # Keep command tracking dominant here; waypoint progress rewards can
            # be re-enabled later for parkour/navigation stages after speed
            # tracking and posture are reliable.
            termination = -0.8
            tracking_lin_vel = 10.0    # Main flat-ground objective: follow commands[:, :2]. Keep high to prevent overspeed.
            tracking_ang_vel = 1.5    # Tracks commands[:, 2]; with goal yaw enabled this helps rotate toward the waypoint smoothly.
            lin_vel_z = -2.0          # Suppress bouncing/hopping; increase if base vertical velocity grows.
            ang_vel_xy = -0.05        # Mild roll/pitch angular-rate penalty; keep mild unless posture becomes visibly unstable.
            orientation = -1.0        # Posture stabilization. Current play metrics look stable, so avoid over-stiffening it.
            torques = -1e-5           # Very light energy regularization; increasing too much can weaken wheel drive.
            dof_vel = -1e-4           # Mild leg joint velocity regularization, wheel DOFs are excluded in go2w_robot.py.
            dof_acc = -2.5e-7         # Mild smoothness term; keep small to avoid suppressing useful gait transitions.
            base_height = -0.5        # Keeps chassis near base_height_target without dominating velocity tracking.
            feet_air_time = 0.0
            foot_clearance = 0.0
            feet_clearance = 0.0
            collision = -0.5          # Penalizes thigh/calf/base contacts. Current collision rate is low, so this is enough.
            feet_stumble = 0.0
            stumble = 0.0
            action_rate = -0.01       # Smooths policy outputs; increase only if actions are visibly jittery.
            stand_still = -0.01       # Only affects near-zero speed commands; low impact for current forward-walk training.
            dof_pos_limits = -0.9     # Strong guard against joint-limit exploitation.
            dof_vel_limits = -0.0
            torque_limits = -0.0
            arm_pos = -0.0
            hip_action_l2 = -0.1      # Discourages excessive hip swing while still allowing leg posture adjustment.
            tracking_goal_vel = 1.0   # Keep disabled in stage 0: current implementation rewards unbounded progress speed, not command tracking.
            tracking_goal_yaw = 0.6   # Small waypoint-facing bias. Lower/disable if it fights straight-line yaw-rate tracking.
            reach_goal = 0.5          # Sparse waypoint bonus. Acceptable on parkour_flat; reduce if it encourages rushing.
            finish_course = 0.0
            wheel_torque = 0.0        # Disabled for now; torque penalties can underpower wheel acceleration.
            wheel_vel_smooth = -1e-7  # Very light wheel speed smoothing; stronger values may make the robot drag wheels.
            wheel_slip = -0.1         # Important for go2w: discourages high-speed wheel spin/sliding without blocking motion.


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
