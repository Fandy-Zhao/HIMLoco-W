# go2W Heading Command 与 Stage2 Reward 清理报告

## 1. 修改动机

go2W/HIM 原始 command 第三维为 `yaw_rate_cmd`，即：

```python
commands[:, 0] = vx_cmd        # m/s
commands[:, 1] = vy_cmd        # m/s
commands[:, 2] = yaw_rate_cmd  # rad/s
```

当前 Stage2 parkour/goal 训练需要参考 Extreme-Parkour 的 waypoint heading 思路，让策略直接学习“目标朝向角 + 平面速度 + waypoint 推进”。因此本次做破坏性修改：

```python
commands[:, 0] = vx_cmd          # m/s
commands[:, 1] = vy_cmd          # m/s
commands[:, 2] = target_yaw_cmd  # rad, world-frame absolute yaw
```

旧 checkpoint 不能复用，需要从头训练。

## 2. 涉及文件

```text
legged_gym/legged_gym/envs/base/legged_robot.py
legged_gym/legged_gym/envs/base/legged_robot_config.py
legged_gym/legged_gym/envs/go2w/go2w_robot.py
legged_gym/legged_gym/envs/go2w/go2w_config.py
legged_gym/legged_gym/utils/task_registry.py
legged_gym/legged_gym/scripts/play.py
documents/go2w_heading_command_reward_cleanup_report.md
```

## 3. Command 语义变化

`commands[:, 2]` 不再表示 yaw 角速度，而是世界坐标系下的目标 yaw 角：

```python
self.commands[env_ids, 2] = torch_rand_float(
    self.command_ranges["heading"][0],
    self.command_ranges["heading"][1],
    (len(env_ids), 1),
    device=self.device,
).squeeze(1)
```

goal terrain 中直接写 waypoint 方向角：

```python
self.target_pos_rel[:] = current_goals[:, :2] - self.root_states[:, :2]
self.target_yaw[:] = torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0])
self.commands[active, 2] = wrap_to_pi(self.target_yaw[active])
self.goal_yaw_error[:] = wrap_to_pi(self.commands[:, 2] - self._get_base_yaw())
```

当前 base yaw 不直接解析 quaternion 顺序，而是复用项目已有 `quat_apply`：

```python
def _get_base_yaw(self):
    forward = quat_apply(self.base_quat, self.forward_vec)
    return torch.atan2(forward[:, 1], forward[:, 0])
```

## 4. Observation 修改

obs 维度保持不变。go2W 仍为：

```text
num_one_step_observations = 73
num_observations = 365
num_privileged_obs = 263
```

但 command obs 第三维从 yaw-rate command 替换为 heading error：

```python
def _get_command_obs(self):
    heading_scale = float(getattr(self.obs_scales, 'heading_error', 1.0))
    return torch.cat((
        self.commands[:, :2] * self.commands_scale[:2],
        self._get_heading_error().unsqueeze(1) * heading_scale,
    ), dim=1)
```

go2W actor obs 中：

```python
self._get_command_obs()
```

因此 actor 看到的是：

```text
vx_cmd
vy_cmd
heading_error = wrap_to_pi(target_yaw_cmd - current_yaw)
```

## 5. Reward 函数新增清单

### tracking_heading

```python
def _reward_tracking_heading(self):
    heading_error = self._get_heading_error()
    return torch.exp(
        -torch.square(heading_error) / float(getattr(self.cfg.rewards, 'heading_sigma', 0.25))
    )
```

### yaw_rate_l2

```python
def _reward_yaw_rate_l2(self):
    return torch.square(self.base_ang_vel[:, 2])
```

### tracking_goal_vel_cmd_scaled

```python
def _reward_tracking_goal_vel_cmd_scaled(self):
    ...
    cmd_speed = torch.norm(self.commands[:, :2], dim=1)
    target_speed = torch.clamp(
        cmd_speed,
        min=0.0,
        max=float(getattr(self.cfg.rewards, 'max_goal_speed', 0.8)),
    )
    reward = torch.exp(
        -torch.square(vel_to_goal - target_speed) / self.cfg.rewards.tracking_sigma
    )
```

### wheel_lateral_slip

已保留上一版新增逻辑，只惩罚轮体坐标系横向速度，不惩罚正常前向滚动。

## 6. Reward 函数替换清单

| 旧项 | 状态 | 新项 |
|---|---|---|
| `tracking_ang_vel` | `legacy`，Stage2 scale=0 | `tracking_heading` |
| `tracking_goal_vel` | `replaced-by-new-reward`，Stage2 scale=0 | `tracking_goal_vel_cmd_scaled` |
| `wheel_slip` | `replaced-by-new-reward`，Stage2 scale=0 | `wheel_lateral_slip` |

`tracking_ang_vel` 函数名没有删除，避免 Stage1 或其他配置引用时报错，但实现已不再做：

```python
self.commands[:, 2] - self.base_ang_vel[:, 2]
```

而是转到 heading reward：

```python
def _reward_tracking_ang_vel(self):
    return self._reward_tracking_heading()
```

## 7. Stage2 禁用的冗余 Reward 清单

Stage2 中显式置 0：

```text
tracking_ang_vel
tracking_goal_vel
wheel_slip
feet_air_time
foot_clearance
feet_clearance
feet_stumble
stumble
wheel_torque
```

这些项分别属于：

| reward | 标注 |
|---|---|
| `tracking_ang_vel` | `legacy`, `disabled-in-stage2` |
| `tracking_goal_vel` | `replaced-by-new-reward`, `disabled-in-stage2` |
| `wheel_slip` | `replaced-by-new-reward`, `disabled-in-stage2` |
| `feet_air_time` | `stage1-only`, `disabled-in-stage2` |
| `foot_clearance` | `stage1-only`, `disabled-in-stage2` |
| `feet_clearance` | `stage1-only`, `disabled-in-stage2` |
| `feet_stumble` | `stage1-only`, `disabled-in-stage2` |
| `stumble` | `stage1-only`, `disabled-in-stage2` |
| `wheel_torque` | `disabled-in-stage2` |

未删除这些函数，因为 Stage1 或其他任务仍可能引用；删除前需要进一步逐项 grep 和任务级验证。

## 8. Stage2 Reward Scales 覆盖表

Stage2 覆盖集中在 `task_registry.py`：

| reward | Stage2 scale |
|---|---:|
| `tracking_ang_vel` | `0.0` |
| `tracking_heading` | `0.8` |
| `tracking_goal_yaw` | `0.8` |
| `yaw_rate_l2` | `-0.02` |
| `tracking_goal_vel` | `0.0` |
| `tracking_goal_vel_cmd_scaled` | `2.0` |
| `reach_goal` | `1.5` |
| `finish_course` | `8.0` |
| `tracking_lin_vel` | `5.0` |
| `wheel_clearance_near_obstacle` | `1.0` |
| `base_height_over_obstacle` | `0.8` |
| `wheel_climb_drive` | `0.1` |
| `wheel_spin_without_progress` | `-0.02` |
| `wheel_lateral_slip` | `-0.3` |
| `wheel_slip` | `0.0` |
| `lin_vel_z` | `-0.8` |
| `ang_vel_xy` | `-0.03` |
| `orientation` | `-0.2` |
| `base_height` | `-0.25` |
| `torques` | `-1e-5` |
| `dof_vel` | `-5e-5` |
| `dof_acc` | `-1e-7` |
| `action_rate` | `-0.005` |
| `collision` | `-0.3` |
| `dof_pos_limits` | `-0.5` |
| `hip_action_l2` | `-0.05` |

## 9. 接触与 Termination 修改

Stage2 保持轮子可接触障碍，不把 wheel body 加入 termination：

```python
env_cfg.asset.penalize_contacts_on = ["base", "trunk", "thigh", "calf"]
env_cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
```

到达最后一个 goal 后 reset 仍保留：

```python
return self.env_has_goals & (self.cur_goal_idx >= final_goal_idx) & self.reached_goal
```

## 10. 风险点

1. 这是破坏性 command 语义修改，旧 checkpoint 不能直接 play 或 resume。
2. `tracking_ang_vel` 函数名仍保留为 legacy compatibility，但语义已变为 heading tracking。
3. Stage1 默认 scale 没有迁移到 Stage2 版本；如果继续训练 Stage1，也是在新 target-yaw command 语义下训练。
4. `commands[:, 2]` 是 absolute yaw，policy obs 使用 heading error，避免 `pi/-pi` 跳变。
5. `tracking_goal_vel_cmd_scaled` 会限制目标推进速度，避免旧 goal velocity reward 诱导无限冲障碍。

## 11. 验证步骤

已完成：

```bash
python3 -m compileall -q \
  legged_gym/legged_gym/envs/base/legged_robot.py \
  legged_gym/legged_gym/envs/base/legged_robot_config.py \
  legged_gym/legged_gym/envs/go2w \
  legged_gym/legged_gym/utils/task_registry.py \
  legged_gym/legged_gym/scripts/play.py

git diff --check -- \
  legged_gym/legged_gym/envs/base/legged_robot.py \
  legged_gym/legged_gym/envs/base/legged_robot_config.py \
  legged_gym/legged_gym/envs/go2w/go2w_robot.py \
  legged_gym/legged_gym/envs/go2w/go2w_config.py \
  legged_gym/legged_gym/utils/task_registry.py \
  legged_gym/legged_gym/scripts/play.py
```

Stage2 配置验证：

```text
tracking_ang_vel = 0.0
tracking_heading = 0.8
tracking_goal_yaw = 0.8
yaw_rate_l2 = -0.02
tracking_goal_vel = 0.0
tracking_goal_vel_cmd_scaled = 2.0
missing reward methods = []
```

2-env smoke：

```text
obs_shape = (2, 365)
priv_shape = (2, 263)
num_commands = torch.Size([2, 3])
target_yaw_cmd in [-pi, pi]
heading_error in [-pi, pi]
step return len = 7
forced_final_reset_buf = tensor([True, True])
```

## 12. 训练建议

1. 从头训练，不加载旧 yaw-rate command checkpoint。
2. Stage2 初期使用较低难度 terrain，先确认机器人能朝 waypoint heading 稳定推进。
3. 观察训练日志中的：

```text
command_vx
command_vy
target_yaw_cmd
current_yaw
heading_error
rew_tracking_heading
rew_tracking_goal_yaw
rew_tracking_goal_vel_cmd_scaled
base_yaw_rate
yaw_rate_l2
```

4. 如果仍出现原地转圈，优先调低 `tracking_goal_yaw` 或增加 `yaw_rate_l2` 负权重。
5. 如果冲障碍过猛，优先降低 `max_goal_speed` 或 `tracking_goal_vel_cmd_scaled`。
6. 如果不愿抬身/抬轮，逐步增大 `wheel_clearance_near_obstacle` 与 `base_height_over_obstacle`，同时确认 `orientation`、`lin_vel_z`、`action_rate` 没有过强。
