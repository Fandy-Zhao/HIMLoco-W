# Go2W Current Reward Functions With Code

本文档基于当前 worktree 导出，覆盖 Go2W reward 注册机制、Stage0/Stage2 scale、关键参数，以及 Stage0/Stage2 会实际注册的 reward 实现。

相关代码：

- `legged_gym/legged_gym/envs/go2w/go2w_config.py`
- `legged_gym/legged_gym/utils/task_registry.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/go2w/go2w_robot.py`

## 1. Reward 注册机制

`LeggedRobot._prepare_reward_function()` 会从 `cfg.rewards.scales` 读取 scale：

- scale 为 `0` 的项会被移除，不注册、不计算。
- 非零 scale 会乘以 policy `dt`。当前 Go2W 配置为 `sim.dt=0.005`、`control.decimation=4`，所以 policy `dt=0.02`。
- 普通 reward 先累加。如果 `cfg.rewards.only_positive_rewards=True`，普通 reward 总和会被 clip 到 `>=0`。
- `termination` reward 在 clip 之后单独加入。
- 函数名必须匹配 `_reward_<scale_name>`。

当前 command 语义：

- `commands[:, 0]`: x 方向速度命令。
- `commands[:, 1]`: y 方向速度命令。
- `commands[:, 2]`: `delta_yaw`，即当前 base yaw 到当前 goal 方向的 yaw error，不是 yaw-rate command。

Go2W stage profile 会在 `task_registry.py` 中通过 `--stage 0` / `--stage 2` 覆盖默认 reward scales。`apply_go2w_reward_profile()` 会先替换掉 inherited/legacy scale 容器，再只写入当前 stage 的核心 reward。

## 2. Stage0 / Stage2 Scale 对照

| Reward | Stage0 scale | Stage2 scale | 实现来源 | 语义 |
| --- | ---: | ---: | --- | --- |
| `goal_progress` | `1.0` | `0.0` | `LeggedRobot` | 沿当前 goal 方向的速度跟踪 reward；Stage2 关闭 |
| `goal_delta_progress` | `2.0` | `3.0` | `LeggedRobot` | 到当前 waypoint 距离减少速度 reward |
| `tracking_delta_yaw` | `1.0` | `1.0` | `LeggedRobot` | `delta_yaw` cosine 朝向 reward |
| `delta_yaw_progress` | `0.0` | `0.0` | `LeggedRobot` | `abs(delta_yaw)` 下降 reward；Stage0/2 关闭 |
| `goal_bonus` | `3.0` | `5.0` | `LeggedRobot` | waypoint/final goal 事件 reward |
| `wheel_clearance` | `0.0` | `-0.8` | `Go2w override` | 障碍前方、非接触轮 clearance 不足 cost |
| `wheel_climb_drive` | `0.0` | `0.0` | `Go2w override` | 目前保持关闭，不注册 |
| `wheel_spin_without_progress` | `-1e-4` | `-5e-4` | `Go2w override` | 轮子高速转但无目标方向进展 cost |
| `wheel_slip` | `-0.1` | `-0.1` | `Go2w override` | 接触轮横向滑移 cost |
| `base_height` | `-0.5` | `-0.2` | `Go2w override` | base 高度偏离目标 cost |
| `orientation` | `-1.0` | `-0.5` | `LeggedRobot` | base roll/pitch 倾斜 cost |
| `lin_vel_z` | `-2.0` | `0.0` | `LeggedRobot` | base 垂直速度 cost；Stage2 关闭 |
| `ang_vel_xy` | `-0.05` | `-0.03` | `LeggedRobot` | base roll/pitch 角速度 cost |
| `yaw_rate_l2` | `-0.01` | `-0.005` | `Go2w override` | 条件 yaw-rate cost，delta_yaw 大时放松 |
| `torques` | `-1e-5` | `-1e-5` | `LeggedRobot` | torque L2 cost，轮子 torque 有权重 |
| `dof_vel` | `-5e-5` | `0.0` | `Go2w override` | 腿部关节速度 cost；Stage2 关闭 |
| `dof_acc` | `0.0` | `0.0` | `LeggedRobot` | 关节加速度 cost；Stage0/2 关闭 |
| `action_rate` | `-0.01` | `-0.005` | `LeggedRobot` | action 差分 cost，轮子 action rate 有权重 |
| `dof_pos_limits` | `-0.9` | `-0.9` | `LeggedRobot` | 腿部关节越限 cost |
| `hip_action_l2` | `-0.05` | `0.0` | `Go2w override` | hip action L2 cost；Stage2 关闭 |
| `stand_still` | `0.0` | `0.0` | `Go2w override` | 低速命令下腿部偏离默认姿态 cost；Stage0/2 关闭 |
| `collision` | `-0.5` | `-0.5` | `LeggedRobot` | penalized body 接触 cost |
| `termination` | `-0.8` | `-1.5` | `LeggedRobot` | failed reset terminal cost |

运行时注册结果：

- Stage0 非零注册项不包含 `delta_yaw_progress`、`wheel_clearance`、`wheel_climb_drive`、`dof_acc`、`stand_still`。
- Stage2 非零注册项包含 `wheel_clearance`，不包含 `goal_progress`、`delta_yaw_progress`、`lin_vel_z`、`dof_vel`、`dof_acc`、`hip_action_l2`、`stand_still`、`wheel_climb_drive`。
- 负 scale 对应的函数均返回非负 cost；乘以负 scale 后变成惩罚。

## 3. Stage0 / Stage2 参数对照

| Param | Stage0 | Stage2 | 说明 |
| --- | ---: | ---: | --- |
| `reward_align_stage2` | `False` | `False` | 旧 `goal_progress_delta` 缓存路径关闭 |
| `use_delta_goal_progress` | `False` | `False` | 旧 `_reward_goal_progress_delta` 关闭 |
| `success_bonus_once` | `True` | `True` | goal bonus 一次性事件逻辑 |
| `goal_progress_delta_max` | `1.0` | `1.0` | `goal_delta_progress` 的速度奖励上界 |
| `min_goal_speed` | `0.0` | `0.15` | `goal_progress` 目标速度下限 |
| `max_goal_speed` | `2.0` | `4.0` | `goal_progress` 目标速度上限 |
| `min_progress_speed` | `0.05` | `0.05` | 空转判定的最小目标方向速度 |
| `stop_cmd_threshold` | `0.05` | `0.05` | 低于该命令速度视为 stop |
| `final_goal_bonus` | `0.0` | `5.0` | final goal 额外 bonus |
| `obstacle_height_offset` | `0.0` | `0.06` | 障碍前方提高 base height target |
| `obstacle_height_threshold` | `0.04` | `0.04` | 前方高度升高判定为 obstacle |
| `gap_height_threshold` | `0.06` | `0.06` | 前方高度下降判定为 gap |
| `wheel_clearance_target` | `0.08` | `0.10` | 兼容参数；当前 clearance 实现使用 radius + margin |
| `wheel_clearance_margin` | `0.04` | `0.04` | 轮心目标高度 = terrain height + wheel radius + margin |
| `contact_force_thresh` | `1.0` | `1.0` | wheel contact 判定阈值 |
| `only_positive_rewards` | `True` | `False` | Stage2 保留负 reward，约束真实生效 |

## 4. 实现细节摘要

| Reward | 返回值含义 | 关键实现 |
| --- | --- | --- |
| `goal_progress` | reward | `exp(-(vel_to_goal - target_speed)^2 / tracking_sigma)` |
| `goal_delta_progress` | reward | `(prev_goal_dist - curr_goal_dist) / dt`，clip 到 `[0, goal_progress_delta_max]` |
| `tracking_delta_yaw` | reward | `0.5 * (cos(commands[:,2]) + 1)` |
| `delta_yaw_progress` | reward/cost 混合 | `prev_abs_delta_yaw - curr_abs_delta_yaw`，clip 到 `[-0.2, 0.2]`；Stage0/2 scale 为 0 |
| `goal_bonus` | reward | 优先使用 `goal_reach_event` 和 `final_goal_event` |
| `wheel_clearance` | cost | 只在 obstacle gate 且非接触轮上惩罚轮心低于目标高度 |
| `wheel_spin_without_progress` | cost | moving command 下，`progress_vel < min_progress_speed` 时惩罚平均轮速平方 |
| `wheel_slip` | cost | 只惩罚接触轮 body frame 横向速度平方 |
| `yaw_rate_l2` | cost | `abs(delta_yaw)` 小时惩罚 yaw rate，转向需求大时放松 |

## 5. Active Reward Code

### Goal / Yaw

```python
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

def _reward_goal_delta_progress(self):
    if not hasattr(self, 'prev_goal_dist') or not hasattr(self, 'curr_goal_dist'):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    curr_dist = self._compute_goal_distance()
    delta = self.prev_goal_dist - curr_dist
    progress_vel = delta / max(self.dt, 1e-6)
    max_progress = float(getattr(self.cfg.rewards, 'goal_progress_delta_max', 1.0))
    reward = torch.clamp(progress_vel, 0.0, max_progress)
    self.curr_goal_dist[:] = curr_dist
    self.goal_progress_delta_raw[:] = reward
    self.prev_goal_dist[:] = curr_dist
    return reward

def _reward_tracking_delta_yaw(self):
    delta_yaw = self.commands[:, 2]
    return 0.5 * (torch.cos(delta_yaw) + 1.0)

def _reward_delta_yaw_progress(self):
    if not hasattr(self, 'prev_abs_delta_yaw') or self.commands.shape[1] < 3:
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    curr = torch.abs(self.commands[:, 2])
    delta = self.prev_abs_delta_yaw - curr
    self.prev_abs_delta_yaw[:] = curr
    return torch.clamp(delta, -0.2, 0.2)

def _reward_goal_bonus(self):
    if hasattr(self, 'goal_reach_event') and hasattr(self, 'final_goal_event'):
        reward = self.goal_reach_event.float()
        final_bonus = float(getattr(self.cfg.rewards, 'final_goal_bonus', 5.0))
        reward = reward + final_bonus * self.final_goal_event.float()
        if hasattr(self, 'env_has_goals'):
            reward[~self.env_has_goals] = 0.0
        return reward
    if not hasattr(self, 'reached_goal'):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    dist = self._compute_goal_distance() if hasattr(self, '_compute_goal_distance') else torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    yaw_err = torch.abs(self.commands[:, 2]) if self.commands.shape[1] >= 3 else torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    reached = (dist < float(getattr(self.cfg.rewards, 'goal_dist_thresh', 0.3))) & (yaw_err < float(getattr(self.cfg.rewards, 'goal_yaw_thresh', 0.4)))
    if bool(getattr(self.cfg.rewards, 'success_bonus_once', True)) and hasattr(self, 'goal_reached_buf'):
        reward = reached & (~self.goal_reached_buf)
        self.goal_reached_buf |= reached
        reward = reward.float()
    else:
        reward = reached.float()
    if hasattr(self, 'cur_goal_idx') and hasattr(self.cfg.terrain, 'num_goals'):
        final_goal_idx = int(self.cfg.terrain.num_goals) - 1
        final_reached = (self.cur_goal_idx >= final_goal_idx) & self.reached_goal
        final_bonus = float(getattr(self.cfg.rewards, 'final_goal_bonus', 5.0))
        reward = reward + final_bonus * final_reached.float()
    return reward
```

### Go2W Overrides

```python
def _reward_base_height(self):
    base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    return torch.square(base_height - self._get_base_height_target())

def _reward_dof_vel(self):
    if not hasattr(self, 'leg_dof_indices'):
        dof_vel = self.dof_vel.clone()
        dof_vel[:, self.wheel_indices] = 0.
        return torch.sum(torch.square(dof_vel), dim=1)
    return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)

def _reward_stand_still(self):
    if not hasattr(self, 'leg_dof_indices'):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0.
    else:
        dof_err = self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]
    return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

def _reward_hip_action_l2(self):
    return torch.sum(self.actions[:, [0, 4, 8, 12]] ** 2, dim=1)

def _reward_yaw_rate_l2(self):
    yaw_rate = self.base_ang_vel[:, 2]
    delta_yaw = torch.abs(self.commands[:, 2])
    gate = torch.clamp(1.0 - delta_yaw / 0.5, 0.0, 1.0)
    return gate * torch.square(yaw_rate)

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

def _reward_wheel_spin_without_progress(self):
    if not hasattr(self, 'wheel_dof_indices'):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    wheel_vel = torch.abs(self.dof_vel[:, self.wheel_dof_indices]).mean(dim=1)
    _, progress_vel, _, moving_cmd = self._get_goal_progress_state()
    min_progress_speed = float(getattr(self.cfg.rewards, 'min_progress_speed', 0.05))
    stuck = (progress_vel < min_progress_speed) & moving_cmd
    reward = stuck.float() * torch.square(wheel_vel)
    return self._mask_invalid_terrain_reward(reward)
```

### Base Stability / Regularization / Safety

```python
def _reward_lin_vel_z(self):
    return torch.square(self.base_lin_vel[:, 2])

def _reward_ang_vel_xy(self):
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

def _reward_orientation(self):
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

def _reward_dof_acc(self):
    dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
    if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
        return torch.sum(torch.square(dof_acc), dim=1)
    leg_acc = dof_acc[:, self.leg_dof_indices]
    wheel_acc = dof_acc[:, self.wheel_dof_indices]
    reward = torch.sum(torch.square(leg_acc), dim=1)
    if wheel_acc.numel() > 0:
        reward = reward + self.cfg.rewards.wheel_acc_weight * torch.sum(torch.square(wheel_acc), dim=1)
    return reward

def _reward_action_rate(self):
    action_rate = self.last_actions - self.actions
    if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
        return torch.sum(torch.square(action_rate), dim=1)
    leg_action_rate = action_rate[:, self.leg_dof_indices]
    wheel_action_rate = action_rate[:, self.wheel_dof_indices]
    reward = torch.sum(torch.square(leg_action_rate), dim=1)
    if wheel_action_rate.numel() > 0:
        reward = reward + self.cfg.rewards.wheel_action_rate_weight * torch.sum(torch.square(wheel_action_rate), dim=1)
    return reward

def _reward_torques(self):
    if not hasattr(self, 'leg_dof_indices') or not hasattr(self, 'wheel_dof_indices'):
        return torch.sum(torch.square(self.torques), dim=1)
    leg_torque = self.torques[:, self.leg_dof_indices]
    wheel_torque = self.torques[:, self.wheel_dof_indices]
    reward = torch.sum(torch.square(leg_torque), dim=1)
    if wheel_torque.numel() > 0:
        wheel_torque_weight = getattr(self.cfg.rewards, 'wheel_drive_torque_weight', self.cfg.rewards.wheel_torque_weight)
        reward = reward + wheel_torque_weight * torch.sum(torch.square(wheel_torque), dim=1)
    return reward

def _reward_dof_pos_limits(self):
    if not hasattr(self, 'leg_dof_indices'):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    leg_pos = self.dof_pos[:, self.leg_dof_indices]
    leg_limits = self.dof_pos_limits[self.leg_dof_indices]
    out_of_limits = -(leg_pos - leg_limits[:, 0]).clip(max=0.)
    out_of_limits += (leg_pos - leg_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

def _reward_collision(self):
    return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

def _reward_termination(self):
    failed_reset = self.reset_buf * ~self.time_out_buf
    if hasattr(self, 'final_goal_reset_buf'):
        failed_reset = failed_reset & ~self.final_goal_reset_buf
    return failed_reset
```

## 6. Inactive / Legacy Reward Functions

这些函数当前仍存在，但 Stage0/Stage2 非零 scale 不使用它们：

| Function | 当前状态 |
| --- | --- |
| `_reward_goal_progress_delta` | 旧距离差分路径，`use_delta_goal_progress=False`，Stage0/2 不注册 |
| `_reward_delta_yaw_progress` | yaw error 下降 shaping，Stage0/2 scale 为 0 |
| `_reward_tracking_goal_vel` | legacy alias，转发到 `goal_progress` |
| `_reward_tracking_goal_vel_cmd_scaled` | legacy alias，转发到 `goal_progress` |
| `_reward_tracking_goal_yaw` | legacy alias，转发到 `tracking_delta_yaw` |
| `_reward_tracking_lin_vel` | 普通速度跟踪，Go2W Stage0/2 不使用 |
| `_reward_tracking_ang_vel` | deprecated yaw-rate tracking，返回 0 |
| `_reward_tracking_heading` | legacy alias，转发到 `tracking_delta_yaw` |
| `_reward_reach_goal` | 稀疏 current waypoint reward，Stage0/2 使用 `goal_bonus` 替代 |
| `_reward_finish_course` | 稀疏 final waypoint reward，Stage0/2 使用 `goal_bonus + final_goal_bonus` 替代 |
| `_reward_success_bonus` | 旧 success event reward，Stage0/2 不注册 |
| `_reward_early_success` | 旧 time shaping，Stage0/2 不注册 |
| `_reward_time_penalty` | 旧 time penalty，Stage0/2 不注册 |
| `_reward_wheel_climb_drive` | Go2W 实现仍存在，但 Stage0/2 scale 为 0 |
| `_reward_goal_progress` | Stage0 使用；Stage2 scale 为 0，改由 `goal_delta_progress` 负责推进 |
| `_reward_lin_vel_z` | Stage0 使用；Stage2 scale 为 0，避免抑制抬身/越障 |
| `_reward_dof_vel` | Stage0 弱使用；Stage2 scale 为 0 |
| `_reward_dof_acc` | Stage0/2 scale 为 0，保留 `action_rate` 作为主要平滑项 |
| `_reward_hip_action_l2` | Stage0 弱使用；Stage2 scale 为 0 |
| `_reward_stand_still` | Stage0/2 scale 为 0 |
| `_reward_wheel_torque` | wheel torque cost，Stage0/2 不注册 |
| `_reward_wheel_vel_smooth` | wheel acceleration smoothness，Stage0/2 不注册 |
| `_reward_joint_power` | power cost，Stage0/2 不注册 |
| `_reward_foot_clearance` | 四足脚 clearance，Stage0/2 不注册 |
| `_reward_smoothness` | 二阶 action smoothness，Stage0/2 不注册 |
| `_reward_dof_vel_limits` | velocity limit cost，Stage0/2 不注册 |
| `_reward_torque_limits` | torque limit cost，Stage0/2 不注册 |
| `_reward_feet_air_time` | 四足步态 reward，Stage0/2 不注册 |
| `_reward_stumble` | 四足 stumble cost，Stage0/2 不注册 |
| `_reward_feet_contact_forces` | foot contact force cost，Stage0/2 不注册 |

## 7. Static Validation

当前静态检查结果：

```text
stage0 nonzero rewards:
goal_progress, goal_delta_progress, tracking_delta_yaw, goal_bonus,
wheel_spin_without_progress, wheel_slip, base_height, orientation, lin_vel_z,
ang_vel_xy, yaw_rate_l2, torques, dof_vel, action_rate, dof_pos_limits,
hip_action_l2, collision, termination

stage2 nonzero rewards:
goal_delta_progress, tracking_delta_yaw, goal_bonus, wheel_clearance,
wheel_spin_without_progress, wheel_slip, base_height, orientation, ang_vel_xy,
yaw_rate_l2, torques, action_rate, dof_pos_limits, collision, termination

missing reward methods:
[]
```

语法检查：

```text
python -m py_compile \
  legged_gym/legged_gym/envs/base/legged_robot.py \
  legged_gym/legged_gym/envs/go2w/go2w_robot.py \
  legged_gym/legged_gym/envs/go2w/go2w_config.py \
  legged_gym/legged_gym/utils/task_registry.py
```

当前环境缺少 `numpy`，因此未运行真实 Isaac Gym env reset/step smoke test。
