# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class HUSTWRoughCfg(GO2WRoughCfg):
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
            'LR_CALF_JOINT': 1.8,
            'LR_WHEEL_JOINT': 0.0,
            'LL_HIP_JOINT': 0.1,
            'LL_THIGH_JOINT': 0.8,
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

    class rewards(GO2WRoughCfg.rewards):
        base_height_target = 0.49
        wheel_clearance_target = 0.111


class HUSTWRoughCfgPPO(GO2WRoughCfgPPO):
    class runner(GO2WRoughCfgPPO.runner):
        run_name = 'hust_w_cad_go2w_baseline'
        experiment_name = 'rough_hust_w'
