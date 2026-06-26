# go2W 当前奖励函数说明与实现代码

本文档基于 40 服务器 /home/zhaozhuofan/test/HIMLoco-W 当前代码导出，覆盖 go2W 任务实际可用的 reward 函数、Stage0/Stage2 覆盖关系，以及完整 _reward_* 实现代码。

## 1. Command 与 Reward 总体设计

当前 go2W/HIM 的 command 语义为：

`python
commands[:, 0] = vx_cmd              # m/s
commands[:, 1] = vy_cmd              # m/s
commands[:, 2] = delta_yaw_to_goal   # rad, wrap_to_pi(target_yaw - current_base_yaw)
`

因此：

- 	racking_delta_yaw 负责朝向目标方向，使用 cosine reward。
- goal_progress 负责沿当前 waypoint 方向推进，目标速度由 x/vy command 的模长决定。
- yaw_rate_l2 只是稳定正则，惩罚实际 yaw rate 过大，不是 command tracking。
- 	racking_ang_vel 已废弃，不能再把 commands[:, 2] 和 ase_ang_vel[:, 2] 相减。
- Stage0/Stage2 均关闭 	racking_lin_vel，主要依赖 goal_progress + tracking_delta_yaw。

## 2. 当前 Reward Scale 摘要

以下内容来自远端当前配置 dump：

`	ext
Importing module 'gym_37' (/home/zhaozhuofan/test/HIMLoco-W/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_37.so)
Setting GYM_USD_PLUG_INFO_PATH to /home/zhaozhuofan/test/HIMLoco-W/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
PyTorch version 1.13.1+cu117
Device count 8
/home/zhaozhuofan/test/HIMLoco-W/isaacgym/python/isaacgym/_bindings/src/gymtorch
Using /home/zhaozhuofan/.cache/torch_extensions/py37_cu117 as PyTorch extensions root...
Emitting ninja build file /home/zhaozhuofan/.cache/torch_extensions/py37_cu117/gymtorch/build.ninja...
Building extension module gymtorch...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module gymtorch...
## default
action_rate = -0.01
ang_vel_xy = -0.05
base_height = -0.5
collision = -0.5
dof_acc = -2.5e-07
dof_pos_limits = -0.9
dof_vel = -0.0001
hip_action_l2 = -0.1
lin_vel_z = -2.0
orientation = -1.0
reach_goal = 0.5
stand_still = -0.01
termination = -0.8
torques = -1e-05
tracking_ang_vel = 1.5
tracking_goal_vel = 1.0
tracking_goal_yaw = 0.6
tracking_lin_vel = 10.0
wheel_slip = -0.1
wheel_vel_smooth = -1e-07
params min/max/stop/obstacle: 0.2 0.8 0.05 0.06 0.04 0.06
Set env learning stage to 0
Set train learning stage to 0
## stage0
action_rate = -0.01
ang_vel_xy = -0.05
base_height = -0.5
collision = -0.5
dof_acc = -2.5e-07
dof_pos_limits = -0.9
dof_vel = -0.0001
goal_progress = 2.0
hip_action_l2 = -0.1
lin_vel_z = -2.0
orientation = -1.0
stand_still = -0.01
termination = -0.8
torques = -1e-05
tracking_delta_yaw = 0.5
wheel_slip = -0.1
wheel_vel_smooth = -1e-07
yaw_rate_l2 = -0.01
params min/max/stop/obstacle: 0.0 0.6 0.05 0.0 0.04 0.06
Set env learning stage to 2
[go2w stage2 delta-yaw reward scales]
  tracking_delta_yaw: 0.8
  goal_progress: 3.0
  reach_goal: 1.5
  finish_course: 8.0
  tracking_ang_vel: 0.0
  tracking_heading: 0.0
  tracking_goal_yaw: 0.0
  tracking_goal_vel: 0.0
  tracking_goal_vel_cmd_scaled: 0.0
  wheel_clearance: 1.0
  base_height_over_obstacle: 0.0
  wheel_climb_drive: 0.1
  wheel_spin_without_progress: -0.02
  wheel_lateral_slip: 0.0
  wheel_slip: -0.3
  orientation: -0.2
  lin_vel_z: -0.8
  base_height: -0.25
  action_rate: -0.005
  penalize_contacts_on: ['base', 'trunk', 'thigh', 'calf']
  terminate_after_contacts_on: ['base', 'trunk']
Set train learning stage to 2
## stage2
action_rate = -0.005
ang_vel_xy = -0.03
base_height = -0.25
collision = -0.3
dof_acc = -1e-07
dof_pos_limits = -0.5
dof_vel = -5e-05
finish_course = 8.0
goal_progress = 3.0
hip_action_l2 = -0.05
lin_vel_z = -0.8
orientation = -0.2
reach_goal = 1.5
stand_still = -0.01
termination = -0.8
torques = -1e-05
tracking_delta_yaw = 0.8
wheel_clearance = 1.0
wheel_climb_drive = 0.1
wheel_slip = -0.3
wheel_spin_without_progress = -0.02
wheel_vel_smooth = -1e-07
yaw_rate_l2 = -0.01
params min/max/stop/obstacle: 0.15 0.8 0.05 0.06 0.04 0.06
EXIT:True

`

## 3. Stage0/Stage2 训练重点

### Stage0

Stage0 是基础 goal-following / flat parkour 阶段：

- 	racking_lin_vel = 0.0
- 	racking_delta_yaw = 0.5
- goal_progress = 2.0
- each_goal = 0.0
- inish_course = 0.0
- min_goal_speed = 0.0
- max_goal_speed = 0.6
- stop_cmd_threshold = 0.05

Stage0 支持 stop/slow command：当 cmd_speed <= stop_cmd_threshold 时，goal_progress 的目标速度为 0。

### Stage2

Stage2 是当前越障训练阶段：

- 	racking_lin_vel = 0.0
- 	racking_delta_yaw = 0.8
- goal_progress = 3.0
- each_goal = 1.5
- inish_course = 8.0
- wheel_clearance = 1.0
- wheel_climb_drive = 0.1
- wheel_spin_without_progress = -0.02
- wheel_slip = -0.3
- yaw_rate_l2 = -0.01

Stage2 明确禁用旧 yaw-rate/普通四足冗余项：

- 	racking_ang_vel = 0.0
- 	racking_heading = 0.0
- 	racking_goal_yaw = 0.0
- 	racking_goal_vel = 0.0
- 	racking_goal_vel_cmd_scaled = 0.0
- eet_air_time = 0.0
- oot_clearance = 0.0
- eet_clearance = 0.0
- eet_stumble = 0.0
- stumble = 0.0
- wheel_torque = 0.0

## 4. Reward 函数分类说明

### Goal / Heading 类

- _reward_goal_progress: 当前主推进 reward，计算机器人沿当前 goal 方向的速度与 command speed 目标的差。
- _reward_tracking_delta_yaw: 当前主朝向 reward，delta_yaw=0 时为 1，pi/2 时为 0.5，pi 时为 0。
- _reward_reach_goal: 到达当前 waypoint 的稀疏奖励。
- _reward_finish_course: 到达最后一个 waypoint 的课程完成奖励。
- _reward_tracking_goal_vel / _reward_tracking_goal_vel_cmd_scaled: 兼容旧 scale 名称，当前转发到 _reward_goal_progress。
- _reward_tracking_goal_yaw / _reward_tracking_heading: 兼容旧名称，当前转发到 _reward_tracking_delta_yaw。
- _reward_tracking_ang_vel: legacy，当前返回 0。

### 轮足越障类

- _reward_wheel_clearance: 前方存在障碍/沟时，鼓励轮子离地高度接近 wheel_clearance_target。
- _reward_wheel_climb_drive: 前方有障碍、轮子接触、command 非停止、且沿 goal 方向有进展时，奖励轮子正向驱动。
- _reward_wheel_spin_without_progress: 前方有障碍但机器人几乎没有前进时，惩罚轮子空转。
- _reward_wheel_slip: 当前只惩罚轮子横向滑移，不惩罚正常前向滚动。
- _reward_wheel_lateral_slip: go2W 中是 _reward_wheel_slip 的别名。

### 姿态 / 稳定 / 能耗类

- _reward_lin_vel_z: 惩罚 base 垂向速度。
- _reward_ang_vel_xy: 惩罚 roll/pitch 角速度。
- _reward_orientation: 惩罚 base 姿态偏离水平。
- _reward_base_height: base 高度目标，当前可根据 obstacle mask 加 obstacle_height_offset。
- _reward_yaw_rate_l2: 惩罚实际 yaw rate 过大，防止原地快速旋转。
- _reward_torques, _reward_dof_vel, _reward_dof_acc, _reward_action_rate: 能耗与平滑正则。
- _reward_collision, _reward_termination, _reward_dof_pos_limits, _reward_dof_vel_limits, _reward_torque_limits: 接触、终止和限位类惩罚。

### 普通四足遗留类

以下函数保留是为了兼容 Stage1/其他任务或旧配置，但 Stage2 中应保持 0：

- _reward_feet_air_time
- _reward_foot_clearance
- _reward_stumble
- _reward_feet_contact_forces
- _reward_feet_stumble

## 5. 完整实现代码

以下源码由当前远端仓库直接抽取。

`python
### legged_gym/legged_gym/envs/base/legged_robot.py
@@ _reward_tracking_goal_vel 1346-1347
    def _reward_tracking_goal_vel(self):
        return self._reward_goal_progress()
@@END
@@ _reward_tracking_goal_vel_cmd_scaled 1349-1350
    def _reward_tracking_goal_vel_cmd_scaled(self):
        return self._reward_goal_progress()
@@END
@@ _reward_goal_progress 1352-1369
    def _reward_goal_progress(self):
        if not hasattr(self, 'env_goals') or not hasattr(self, 'cur_goal_idx'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        cur_goal = self.env_goals[env_ids, self.cur_goal_idx]
        target_vec = cur_goal[:, :2] - self.root_states[:, :2]
        target_dir = target_vec / (torch.norm(target_vec, dim=1, keepdim=True) + 1e-6)
        vel_to_goal = torch.sum(target_dir * self.root_states[:, 7:9], dim=1)

        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        stop_cmd_threshold = float(getattr(self.cfg.rewards, 'stop_cmd_threshold', 0.05))
        min_goal_speed = float(getattr(self.cfg.rewards, 'min_goal_speed', 0.0))
        max_goal_speed = float(getattr(self.cfg.rewards, 'max_goal_speed', 0.8))
        target_speed = torch.clamp(cmd_speed, min=min_goal_speed, max=max_goal_speed)
        target_speed = torch.where(cmd_speed <= stop_cmd_threshold, torch.zeros_like(target_speed), target_speed)
        reward = torch.exp(-torch.square(vel_to_goal - target_speed) / self.cfg.rewards.tracking_sigma)
        return reward
@@END
@@ _reward_tracking_goal_yaw 1371-1372
    def _reward_tracking_goal_yaw(self):
        return self._reward_tracking_delta_yaw()
@@END
@@ _reward_reach_goal 1374-1380
    def _reward_reach_goal(self):
        if not hasattr(self, 'reached_goal'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward = self.reached_goal.float()
        if hasattr(self, 'env_has_goals'):
            reward[~self.env_has_goals] = 0.0
        return reward
@@END
@@ _reward_finish_course 1382-1389
    def _reward_finish_course(self):
        if not hasattr(self, 'cur_goal_idx') or not hasattr(self, 'reached_goal') or not hasattr(self.cfg.terrain, 'num_goals'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        final_goal_reached = (self.cur_goal_idx >= self.cfg.terrain.num_goals - 1) & self.reached_goal
        reward = final_goal_reached.float()
        if hasattr(self, 'env_has_goals'):
            reward[~self.env_has_goals] = 0.0
        return reward
@@END
@@ _reward_wheel_torque 1396-1401
    def _reward_wheel_torque(self):
        if not hasattr(self, 'wheel_dof_indices'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        wheel_torque = self.torques[:, self.wheel_dof_indices]
        reward = torch.sum(torch.square(wheel_torque), dim=1)
        return self._mask_invalid_terrain_reward(reward)
@@END
@@ _reward_wheel_vel_smooth 1403-1410
    def _reward_wheel_vel_smooth(self):
        if not hasattr(self, 'wheel_dof_indices') or not hasattr(self, 'last_dof_vel'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        wheel_vel = self.dof_vel[:, self.wheel_dof_indices]
        last_wheel_vel = self.last_dof_vel[:, self.wheel_dof_indices]
        wheel_acc = (wheel_vel - last_wheel_vel) / self.dt
        reward = torch.sum(torch.square(wheel_acc), dim=1)
        return self._mask_invalid_terrain_reward(reward)
@@END
@@ _reward_wheel_slip 1412-1422
    def _reward_wheel_slip(self):
        if not hasattr(self, 'wheel_body_indices') or not hasattr(self, 'rigid_body_states'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rigid_body_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_vel_world = rigid_body_state[:, self.wheel_body_indices, 7:10]
        base_quat = self.base_quat.unsqueeze(1).repeat(1, len(self.wheel_body_indices), 1)
        wheel_vel_body = quat_rotate_inverse(base_quat, wheel_vel_world)
        lateral_vel = wheel_vel_body[..., 1]
        wheel_contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0
        reward = torch.sum(torch.square(lateral_vel) * wheel_contact.float(), dim=1)
        return self._mask_invalid_terrain_reward(reward)
@@END
@@ _reward_tracking_lin_vel 1424-1430
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        if hasattr(self, 'env_has_goals'):
            reward[self.env_has_goals] *= 0.2
        return reward
@@END
@@ _reward_tracking_ang_vel 1432-1434
    def _reward_tracking_ang_vel(self):
        # Deprecated yaw-rate command tracking. commands[:, 2] is delta yaw.
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
@@END
@@ _reward_tracking_heading 1436-1437
    def _reward_tracking_heading(self):
        return self._reward_tracking_delta_yaw()
@@END
@@ _reward_tracking_delta_yaw 1439-1441
    def _reward_tracking_delta_yaw(self):
        delta_yaw = self._get_delta_yaw_to_goal()
        return 0.5 * (torch.cos(delta_yaw) + 1.0)
@@END
@@ _reward_yaw_rate_l2 1443-1444
    def _reward_yaw_rate_l2(self):
        return torch.square(self.base_ang_vel[:, 2])
@@END
@@ _reward_lin_vel_z 1446-1448
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
@@END
@@ _reward_ang_vel_xy 1450-1452
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
@@END
@@ _reward_orientation 1454-1456
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
@@END
@@ _reward_dof_acc 1458-1468
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum(torch.square(dof_acc), dim=1)
        leg_acc = dof_acc[:, self.leg_dof_indices]
        wheel_acc = dof_acc[:, self.wheel_dof_indices]
        reward = torch.sum(torch.square(leg_acc), dim=1)
        if wheel_acc.numel() > 0:
            reward = reward + self.cfg.rewards.wheel_acc_weight * torch.sum(torch.square(wheel_acc), dim=1)
        return reward
@@END
@@ _reward_joint_power 1470-1480
    def _reward_joint_power(self):
        #Penalize high power
        power = torch.abs(self.dof_vel) * torch.abs(self.torques)
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum(power, dim=1)
        leg_power = power[:, self.leg_dof_indices]
        wheel_power = power[:, self.wheel_dof_indices]
        reward = torch.sum(leg_power, dim=1)
        if wheel_power.numel() > 0:
            reward = reward + self.cfg.rewards.wheel_torque_weight * torch.sum(wheel_power, dim=1)
        return reward
@@END
@@ _reward_base_height 1482-1491
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
@@END
@@ _reward_foot_clearance 1505-1517
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
@@END
@@ _reward_action_rate 1519-1529
    def _reward_action_rate(self):
        # Penalize changes in actions
        action_rate = self.last_actions - self.actions
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum(torch.square(action_rate), dim=1)
        leg_action_rate = action_rate[:, self.leg_dof_indices]
        wheel_action_rate = action_rate[:, self.wheel_dof_indices]
        reward = torch.sum(torch.square(leg_action_rate), dim=1)
        if wheel_action_rate.numel() > 0:
            reward = reward + self.cfg.rewards.wheel_action_rate_weight * torch.sum(torch.square(wheel_action_rate), dim=1)
        return reward
@@END
@@ _reward_smoothness 1531-1541
    def _reward_smoothness(self):
        # second order smoothness
        action_delta = self.actions - self.last_actions - self.last_actions + self.last_last_actions
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum(torch.square(action_delta), dim=1)
        leg_delta = action_delta[:, self.leg_dof_indices]
        wheel_delta = action_delta[:, self.wheel_dof_indices]
        reward = torch.sum(torch.square(leg_delta), dim=1)
        if wheel_delta.numel() > 0:
            reward = reward + self.cfg.rewards.wheel_action_rate_weight * torch.sum(torch.square(wheel_delta), dim=1)
        return reward
@@END
@@ _reward_torques 1543-1552
    def _reward_torques(self):
        # Penalize torques
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum(torch.square(self.torques), dim=1)
        leg_torque = self.torques[:, self.leg_dof_indices]
        wheel_torque = self.torques[:, self.wheel_dof_indices]
        reward = torch.sum(torch.square(leg_torque), dim=1)
        if wheel_torque.numel() > 0:
            reward = reward + self.cfg.rewards.wheel_torque_weight * torch.sum(torch.square(wheel_torque), dim=1)
        return reward
@@END
@@ _reward_dof_vel 1554-1559
    def _reward_dof_vel(self):
        # Penalize dof velocities
        if not hasattr(self, 'leg_dof_indices'):
            return torch.sum(torch.square(self.dof_vel), dim=1)
        leg_dof_vel = self.dof_vel[:, self.leg_dof_indices]
        return torch.sum(torch.square(leg_dof_vel), dim=1)
@@END
@@ _reward_collision 1561-1563
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
@@END
@@ _reward_termination 1565-1570
    def _reward_termination(self):
        # Terminal reward / penalty
        failed_reset = self.reset_buf * ~self.time_out_buf
        if hasattr(self, 'final_goal_reset_buf'):
            failed_reset = failed_reset & ~self.final_goal_reset_buf
        return failed_reset
@@END
@@ _reward_dof_pos_limits 1572-1582
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
@@END
@@ _reward_dof_vel_limits 1584-1591
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if not hasattr(self, 'leg_dof_indices'):
            return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
        leg_dof_vel = self.dof_vel[:, self.leg_dof_indices]
        leg_dof_vel_limits = self.dof_vel_limits[self.leg_dof_indices]
        return torch.sum((torch.abs(leg_dof_vel) - leg_dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
@@END
@@ _reward_torque_limits 1593-1604
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
            return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        leg_torques = self.torques[:, self.leg_dof_indices]
        leg_limits = self.torque_limits[self.leg_dof_indices]
        reward = torch.sum((torch.abs(leg_torques) - leg_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        wheel_torques = self.torques[:, self.wheel_dof_indices]
        wheel_limits = self.torque_limits[self.wheel_dof_indices]
        if wheel_torques.numel() > 0:
            reward = reward + self.cfg.rewards.wheel_torque_weight * torch.sum((torch.abs(wheel_torques) - wheel_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        return reward
@@END
@@ _reward_feet_air_time 1606-1626
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
@@END
@@ _reward_stumble 1628-1632
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        feet_body_indices = self._get_reward_feet_body_indices()
        return torch.any(torch.norm(self.contact_forces[:, feet_body_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, feet_body_indices, 2]), dim=1)
@@END
@@ _reward_stand_still 1634-1640
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        if not hasattr(self, 'leg_dof_indices'):
            dof_error = torch.abs(self.dof_pos - self.default_dof_pos)
        else:
            dof_error = torch.abs(self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices])
        return torch.sum(dof_error, dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
@@END
@@ _reward_feet_contact_forces 1642-1645
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        feet_body_indices = self._get_reward_feet_body_indices()
        return torch.sum((torch.norm(self.contact_forces[:, feet_body_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
@@END
### legged_gym/legged_gym/envs/go2w/go2w_robot.py
@@ _reward_base_height 173-177
    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        obstacle_mask = self._get_obstacle_ahead_mask() if hasattr(self, '_get_obstacle_ahead_mask') else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        target = float(getattr(self.cfg.rewards, 'base_height_target', 0.34)) + float(getattr(self.cfg.rewards, 'obstacle_height_offset', 0.0)) * obstacle_mask.float()
        return torch.square(base_height - target)
@@END
@@ _reward_dof_vel 179-184
    def _reward_dof_vel(self):
        if not hasattr(self, 'leg_dof_indices'):
            dof_vel = self.dof_vel.clone()
            dof_vel[:, self.wheel_indices] = 0.
            return torch.sum(torch.square(dof_vel), dim=1)
        return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)
@@END
@@ _reward_feet_stumble 186-191
    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )
@@END
@@ _reward_stand_still 193-199
    def _reward_stand_still(self):
        if not hasattr(self, 'leg_dof_indices'):
            dof_err = self.dof_pos - self.default_dof_pos
            dof_err[:, self.wheel_indices] = 0.
        else:
            dof_err = self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
@@END
@@ _reward_hip_action_l2 201-202
    def _reward_hip_action_l2(self):
        return torch.sum(self.actions[:, [0, 4, 8, 12]] ** 2, dim=1)
@@END
@@ _reward_wheel_lateral_slip 243-244
    def _reward_wheel_lateral_slip(self):
        return self._reward_wheel_slip()
@@END
@@ _reward_wheel_slip 246-258
    def _reward_wheel_slip(self):
        if not hasattr(self, 'wheel_body_indices') or not hasattr(self, 'rigid_body_states'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        rigid_body_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_vel_world = rigid_body_state[:, self.wheel_body_indices, 7:10]
        base_quat = self.base_quat.unsqueeze(1).repeat(1, len(self.wheel_body_indices), 1)
        wheel_vel_body = quat_rotate_inverse(base_quat, wheel_vel_world)
        lateral_vel = wheel_vel_body[..., 1]
        wheel_contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0

        reward = torch.sum(torch.square(lateral_vel) * wheel_contact.float(), dim=1)
        return self._mask_invalid_terrain_reward(reward)
@@END
@@ _reward_wheel_clearance_near_obstacle 260-261
    def _reward_wheel_clearance_near_obstacle(self):
        return self._reward_wheel_clearance()
@@END
@@ _reward_wheel_clearance 263-277
    def _reward_wheel_clearance(self):
        if not hasattr(self, 'wheel_body_indices') or not hasattr(self, 'rigid_body_states'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        rigid_body_state = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_pos = rigid_body_state[:, self.wheel_body_indices, :3]
        terrain_h = self._get_heights_at_points(wheel_pos[..., :2])
        clearance = wheel_pos[..., 2] - terrain_h
        target = getattr(self.cfg.rewards, 'wheel_clearance_target', 0.10)
        err = torch.square(clearance - target)

        wheel_contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0
        reward = torch.sum(torch.exp(-err / 0.01) * (~wheel_contact).float(), dim=1)
        reward *= self._get_obstacle_ahead_mask().float()
        return self._mask_invalid_terrain_reward(reward)
@@END
@@ _reward_base_height_over_obstacle 279-280
    def _reward_base_height_over_obstacle(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
@@END
@@ _reward_wheel_climb_drive 282-300
    def _reward_wheel_climb_drive(self):
        if not hasattr(self, 'wheel_dof_indices') or not hasattr(self, 'wheel_body_indices'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        wheel_vel = self.dof_vel[:, self.wheel_dof_indices]
        wheel_forward_sign = getattr(self, 'wheel_forward_sign', torch.ones(len(self.wheel_dof_indices), dtype=torch.float, device=self.device))
        positive_wheel_spin = torch.clamp(wheel_vel * wheel_forward_sign.unsqueeze(0), min=0.0)
        env_ids = torch.arange(self.num_envs, device=self.device)
        cur_goal = self.env_goals[env_ids, self.cur_goal_idx] if hasattr(self, 'env_goals') and hasattr(self, 'cur_goal_idx') else self.root_states[:, :3]
        target_vec = cur_goal[:, :2] - self.root_states[:, :2]
        target_dir = target_vec / (torch.norm(target_vec, dim=1, keepdim=True) + 1e-6)
        progress_vel = torch.clamp(torch.sum(target_dir * self.root_states[:, 7:9], dim=1), min=0.0).unsqueeze(1)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        moving_cmd = (cmd_speed > float(getattr(self.cfg.rewards, 'stop_cmd_threshold', 0.05))).float().unsqueeze(1)
        wheel_contact = self.contact_forces[:, self.wheel_body_indices, 2] > 1.0

        reward = torch.sum(positive_wheel_spin * progress_vel * moving_cmd * wheel_contact.float(), dim=1)
        reward *= self._get_obstacle_ahead_mask().float()
        return self._mask_invalid_terrain_reward(reward)
@@END
@@ _reward_wheel_spin_without_progress 302-310
    def _reward_wheel_spin_without_progress(self):
        if not hasattr(self, 'wheel_dof_indices'):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        wheel_speed = torch.abs(self.dof_vel[:, self.wheel_dof_indices])
        low_progress = torch.abs(self.base_lin_vel[:, 0]) < 0.05
        reward = torch.sum(wheel_speed, dim=1) * low_progress.float()
        reward *= self._get_obstacle_ahead_mask().float()
        return self._mask_invalid_terrain_reward(reward)
@@END

`

## 6. 训练使用建议

- 新模型必须从头训练，旧 yaw-rate command checkpoint 不兼容。
- Stage0 先确认 stop/slow command、delta-yaw 朝向和基础 goal progress。
- Stage2 再打开越障相关奖励，重点观察 goal_progress、	racking_delta_yaw、wheel_clearance、wheel_climb_drive、wheel_slip、yaw_rate_l2。
- 如果机器人原地转圈，优先检查 delta_yaw_cmd、ase_yaw_rate、yaw_rate_l2 和 	racking_delta_yaw。
- 如果机器人冲障碍过快，优先调低 goal_progress、max_goal_speed 或提高 wheel_spin_without_progress / 稳定项惩罚。
