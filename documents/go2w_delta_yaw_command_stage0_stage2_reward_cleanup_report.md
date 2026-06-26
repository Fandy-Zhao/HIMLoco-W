# go2W Delta-Yaw Command 与 Stage0/Stage2 Reward 清理报告

## 1. 修改动机

本次继续清理 go2W/HIM 的 command、goal 与 reward 设计。上一版将 `commands[:, 2]` 从 yaw rate 改成 absolute target yaw；本次最终改为：

```python
commands[:, 2] = delta_yaw_to_next_goal
```

也就是机器人当前朝向到下一 waypoint/goal 方向的相对角度误差。这样 actor 直接看到“目标在当前机体朝向左/右多少角度”，避免 absolute yaw 的世界坐标语义和 observation 周期跳变问题。

## 2. Command 语义变化

最终 command 定义：

```python
commands[:, 0] = vx_cmd                  # m/s
commands[:, 1] = vy_cmd                  # m/s
commands[:, 2] = delta_yaw_to_next_goal  # rad
```

`delta_yaw_to_next_goal` 的计算：

```python
target_vec = cur_goal[:, :2] - self.root_states[:, :2]
target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
delta_yaw = wrap_to_pi(target_yaw - self._get_base_yaw())
self.commands[:, 2] = delta_yaw
```

当前 base yaw 通过项目已有 `quat_apply` 和 `forward_vec` 计算，不手写 quaternion 顺序：

```python
def _get_base_yaw(self):
    forward = quat_apply(self.base_quat, self.forward_vec)
    return torch.atan2(forward[:, 1], forward[:, 0])
```

## 3. Goal 更新逻辑

新增统一入口：

```python
def _update_goal_commands(self):
    ...
    self.target_pos_rel[:] = target_vec
    self.target_yaw[:] = target_yaw
    self.goal_yaw_error[:] = delta_yaw
    self.commands[:, 2] = delta_yaw
    self.env_has_goals[:] = True
```

`_resample_commands()` 现在只采样平面速度，不再随机采样第三维：

```python
self.commands[env_ids, 0] = torch_rand_float(...)
self.commands[env_ids, 1] = torch_rand_float(...)
self._update_goal_commands()
```

`env_has_goals` 保留为兼容字段，但现在默认所有 env 都视为有 goal。

## 4. Observation 修改

actor obs 维度不变，go2W 仍为：

```text
num_one_step_observations = 73
num_observations = 365
num_privileged_obs = 263
```

command obs 改成：

```python
def _get_command_obs(self):
    delta_yaw_scale = float(getattr(self.obs_scales, 'delta_yaw', 1.0))
    return torch.cat((
        self.commands[:, :2] * self.commands_scale[:2],
        self.commands[:, 2:3] * delta_yaw_scale,
    ), dim=1)
```

actor 看到：

```text
vx_cmd
vy_cmd
delta_yaw_to_next_goal
```

## 5. Reward 精简清单

保留核心语义清晰的 reward：

```text
tracking_delta_yaw
goal_progress
wheel_slip
base_height
wheel_clearance
wheel_climb_drive
wheel_spin_without_progress
tracking_lin_vel
yaw_rate_l2
```

旧重复项保留为 deprecated/alias，避免其他任务或旧配置直接崩：

| 旧 reward | 处理方式 |
|---|---|
| `tracking_ang_vel` | deprecated，返回 0，stage scale=0 |
| `tracking_heading` | alias 到 `tracking_delta_yaw`，stage scale=0 |
| `tracking_goal_yaw` | alias 到 `tracking_delta_yaw`，stage scale=0 |
| `tracking_goal_vel` | alias 到 `goal_progress`，stage scale=0 |
| `tracking_goal_vel_cmd_scaled` | alias 到 `goal_progress`，stage scale=0 |
| `wheel_lateral_slip` | alias 到修正后的 `wheel_slip`，stage scale=0 |
| `base_height_over_obstacle` | deprecated，stage scale=0 |
| `wheel_clearance_near_obstacle` | alias 到 `wheel_clearance` |
| `feet_air_time` / `foot_clearance` / `feet_clearance` | Stage0/2 禁用 |
| `feet_stumble` / `stumble` | Stage0/2 禁用 |
| `wheel_torque` | Stage0/2 暂时禁用 |

## 6. Stage0 Reward Scales

Stage0 在 `task_registry.py` 中通过 `apply_go2w_stage0_scales()` 覆盖，不改默认 config scales。

核心目标：平地/简单地形基础行走、速度跟随、delta yaw 对齐。

关键 scale：

```text
tracking_lin_vel = 5.0
tracking_delta_yaw = 0.5
goal_progress = 0.5
reach_goal = 0.5
wheel_slip = -0.1
yaw_rate_l2 = -0.01
wheel_clearance = 0.0
wheel_climb_drive = 0.0
wheel_spin_without_progress = 0.0
```

旧重复项全部置 0：

```text
tracking_ang_vel
tracking_heading
tracking_goal_yaw
tracking_goal_vel
tracking_goal_vel_cmd_scaled
wheel_lateral_slip
base_height_over_obstacle
```

## 7. Stage2 Reward Scales

Stage2 在 `task_registry.py` 中通过 `apply_go2w_stage2_scales()` 覆盖。

核心目标：复杂地形/parkour 越障，沿 goal 推进、抬轮/抬身、轮子有效爬障。

关键 scale：

```text
tracking_lin_vel = 3.0
tracking_delta_yaw = 0.8
goal_progress = 2.5
reach_goal = 1.5
finish_course = 8.0
wheel_clearance = 1.0
wheel_climb_drive = 0.1
wheel_spin_without_progress = -0.02
wheel_slip = -0.3
yaw_rate_l2 = -0.02
base_height = -0.25
orientation = -0.2
lin_vel_z = -0.8
action_rate = -0.005
```

Stage2 禁用：

```text
tracking_ang_vel = 0
tracking_heading = 0
tracking_goal_yaw = 0
tracking_goal_vel = 0
tracking_goal_vel_cmd_scaled = 0
wheel_lateral_slip = 0
base_height_over_obstacle = 0
feet_air_time = 0
foot_clearance = 0
feet_clearance = 0
feet_stumble = 0
stumble = 0
wheel_torque = 0
```

## 8. 接触与 Termination 修改

Stage2 继续允许轮子接触障碍，不把 wheel body 加入 termination：

```python
env_cfg.asset.penalize_contacts_on = ["base", "trunk", "thigh", "calf"]
env_cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
```

final goal reset 不再触发 termination penalty：

```python
failed_reset = self.reset_buf * ~self.time_out_buf
if hasattr(self, 'final_goal_reset_buf'):
    failed_reset = failed_reset & ~self.final_goal_reset_buf
return failed_reset
```

## 9. Play/Eval 修改

`play.py` 不再把第三维当 yaw rate 或 absolute yaw 手动写入。正常逻辑调用：

```python
env._update_goal_commands()
```

如果手动测试第三维，含义也是 delta yaw：

```text
0.0     目标在正前方
1.5708 目标在左侧 90 度
-1.5708 目标在右侧 90 度
```

## 10. 验证结果

已执行：

```bash
python3 -m compileall -q ...
git diff --check -- ...
```

Stage0/Stage2 配置验证：

```text
num_commands = 3
tracking_ang_vel = 0
tracking_heading = 0
tracking_goal_yaw = 0
tracking_goal_vel = 0
tracking_goal_vel_cmd_scaled = 0
wheel_lateral_slip = 0
base_height_over_obstacle = 0
missing reward methods = []
```

2-env smoke：

```text
obs_shape = (2, 365)
priv_shape = (2, 263)
commands_shape = (2, 3)
commands[:, 2] == goal_yaw_error == expected_delta_yaw
commands[:, 2] in [-pi, pi]
step return len = 7
goal_progress valid
wheel_slip valid
final_goal_reset_buf = True
termination_reward_for_final_goal = 0
```

## 11. 训练建议

训练流程：

```text
stage0: 从头训练基础行走
stage2: 从 stage0 checkpoint 继续训练越障
```

Stage0 建议：

```text
flat / parkour_flat / mild rough
低速开始
wheel_clearance / wheel_climb / wheel_spin 关闭
```

Stage2 建议：

```text
先低台阶、低坡、简单 bridge
再逐步加入 T_step_stl / BridgeA / BridgeB
lin_vel_y 保持 0，减少 lateral command 和 goal progress 冲突
```

## 12. 已知风险

1. 这是破坏性 command 语义修改，旧 checkpoint 不能复用。
2. `env_has_goals` 仍保留兼容，但逻辑上所有 env 都应有 goal。
3. legacy reward 函数未删除，只通过 stage scale 禁用或 alias，避免影响其他任务。
4. 如果训练中原地旋转，优先增大 `yaw_rate_l2` 负权重或降低 `tracking_delta_yaw`。
5. 如果冲障碍过猛，降低 `goal_progress` 或 `max_goal_speed`。
6. 如果不抬轮，增大 `wheel_clearance` 或适度增大 `obstacle_height_offset`。
