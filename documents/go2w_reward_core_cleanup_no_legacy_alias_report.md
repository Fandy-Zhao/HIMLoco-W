# go2W Reward Core Cleanup No Legacy Alias Report

## 1. 删除的旧 Reward Scale 清单

go2W default / Stage0 / Stage2 不再保留以下 legacy scale。Stage0/Stage2 会先将 `env_cfg.rewards.scales` 替换为新的 `Go2WCoreRewardScales` 空容器，再只写入核心 reward，避免 inherited base scale 通过 `class_to_dict()` 残留到 dump 或 TensorBoard。

已清理的 legacy scale：

```text
tracking_lin_vel
tracking_ang_vel
tracking_heading
tracking_goal_yaw
tracking_goal_vel
tracking_goal_vel_cmd_scaled
reach_goal
finish_course
wheel_lateral_slip
wheel_clearance_near_obstacle
base_height_over_obstacle
wheel_torque
joint_power
smoothness
wheel_vel_smooth
feet_air_time
foot_clearance
feet_clearance
feet_stumble
stumble
feet_contact_forces
```

## 2. 删除或不再启用的旧 Alias 函数

go2W 专属 alias 函数已删除：

```text
Go2w._reward_wheel_lateral_slip
Go2w._reward_wheel_clearance_near_obstacle
Go2w._reward_base_height_over_obstacle
Go2w._reward_feet_stumble
```

base class 中的通用旧函数暂时保留，避免影响其他任务；但 go2W default / Stage0 / Stage2 不再配置这些 scale，因此不会进入 `_prepare_reward_function()`。

## 3. 最终保留的核心 Reward 清单

go2W Stage0/Stage2 只允许以下核心 reward 名称：

```text
goal_progress
tracking_delta_yaw
goal_bonus
wheel_clearance
wheel_climb_drive
wheel_spin_without_progress
wheel_slip
base_height
orientation
lin_vel_z
ang_vel_xy
yaw_rate_l2
torques
dof_vel
dof_acc
action_rate
dof_pos_limits
hip_action_l2
stand_still
collision
termination
```

## 4. Stage0 最终 Scale Dump

远端验证输出：

```text
STAGE 0
action_rate=-0.01
ang_vel_xy=-0.05
base_height=-0.5
collision=-0.5
dof_acc=-2.5e-07
dof_pos_limits=-0.9
dof_vel=-0.0001
goal_bonus=0.0
goal_progress=2.0
hip_action_l2=-0.1
lin_vel_z=-2.0
orientation=-1.0
stand_still=-0.01
termination=-0.8
torques=-1e-05
tracking_delta_yaw=0.5
wheel_clearance=0.0
wheel_climb_drive=0.0
wheel_slip=-0.1
wheel_spin_without_progress=0.0
yaw_rate_l2=-0.01
LEGACY_PRESENT []
NON_CORE_PRESENT []
MISSING_NONZERO_FN []
```

运行时 `_prepare_reward_function()` 会移除 0 scale，因此 Stage0 runtime reward keys 不含 `goal_bonus`、`wheel_clearance`、`wheel_climb_drive`、`wheel_spin_without_progress`。

## 5. Stage2 最终 Scale Dump

远端验证输出：

```text
STAGE 2
action_rate=-0.005
ang_vel_xy=-0.03
base_height=-0.25
collision=-0.3
dof_acc=-1e-07
dof_pos_limits=-0.5
dof_vel=-5e-05
goal_bonus=1.5
goal_progress=3.0
hip_action_l2=-0.05
lin_vel_z=-0.8
orientation=-0.2
stand_still=-0.01
termination=-0.8
torques=-1e-05
tracking_delta_yaw=0.8
wheel_clearance=1.0
wheel_climb_drive=0.1
wheel_slip=-0.3
wheel_spin_without_progress=-0.02
yaw_rate_l2=-0.02
LEGACY_PRESENT []
NON_CORE_PRESENT []
MISSING_NONZERO_FN []
```

## 6. `_validate_reward_scales` 检查结果

已在 `LeggedRobot._prepare_reward_function()` 开头增加：

```python
def _validate_reward_scales(self):
    missing = []
    for name, scale in self.reward_scales.items():
        if scale == 0:
            continue
        fn_name = '_reward_' + name
        if not hasattr(self, fn_name):
            missing.append((name, fn_name, scale))
    if missing:
        msg = "\n".join([f"{name}: {fn_name}, scale={scale}" for name, fn_name, scale in missing])
        raise RuntimeError(f"Missing reward functions for non-zero scales:\n{msg}")
```

Stage0/Stage2 验证结果：

```text
MISSING_NONZERO_FN []
```

## 7. Grep 验证结果

已在远端执行源码 grep，并排除 `__pycache__`：

```text
PATTERN:tracking_goal_vel|tracking_goal_vel_cmd_scaled|tracking_goal_yaw|tracking_heading|tracking_ang_vel
STATUS:1
PATTERN:wheel_lateral_slip|wheel_clearance_near_obstacle|base_height_over_obstacle
STATUS:1
PATTERN:reach_goal|finish_course
STATUS:1
PATTERN:wheel_torque|joint_power|smoothness|wheel_vel_smooth
STATUS:1
```

`STATUS:1` 表示目标源码范围内无匹配项。

## 8. Compileall / Git Diff Check 结果

远端已执行：

```bash
python3 -m compileall -q \
  legged_gym/legged_gym/envs/base/legged_robot.py \
  legged_gym/legged_gym/envs/go2w \
  legged_gym/legged_gym/utils/task_registry.py \
  legged_gym/legged_gym/scripts/play.py

git diff --check
```

结果：均通过。

## 9. 2-env Smoke Test 结果

Stage0：

```text
stage 0 obs (2, 365) priv (2, 263) step_len 7
rew_keys ['action_rate', 'ang_vel_xy', 'base_height', 'collision', 'dof_acc', 'dof_pos_limits', 'dof_vel', 'goal_progress', 'hip_action_l2', 'lin_vel_z', 'orientation', 'stand_still', 'termination', 'torques', 'tracking_delta_yaw', 'wheel_slip', 'yaw_rate_l2']
stage 0 goal_bonus_waypoint [1.0, 1.0] final [1.0, 1.0]
stage 0 reward cleanup smoke ok
```

Stage2：

```text
stage 2 obs (2, 365) priv (2, 263) step_len 7
rew_keys ['action_rate', 'ang_vel_xy', 'base_height', 'collision', 'dof_acc', 'dof_pos_limits', 'dof_vel', 'goal_bonus', 'goal_progress', 'hip_action_l2', 'lin_vel_z', 'orientation', 'stand_still', 'termination', 'torques', 'tracking_delta_yaw', 'wheel_clearance', 'wheel_climb_drive', 'wheel_slip', 'wheel_spin_without_progress', 'yaw_rate_l2']
stage 2 goal_bonus_waypoint [1.0, 1.0] final [6.0, 6.0]
stage 2 reward cleanup smoke ok
```

验证点：

- `reset()` 返回 `(obs, privileged_obs)`。
- `step()` 返回 7 项。
- obs shape 为 `[2, 365]`，privileged obs shape 为 `[2, 263]`。
- Stage0/Stage2 runtime reward keys 不含 legacy reward。
- Stage2 `goal_bonus` 合并 waypoint bonus 和 final-goal bonus，final goal reward 为 `1 + final_goal_bonus = 6`。
- final-goal reset 不触发 termination penalty。

## 10. 涉及文件

```text
legged_gym/legged_gym/envs/base/legged_robot.py
legged_gym/legged_gym/envs/go2w/go2w_robot.py
legged_gym/legged_gym/envs/go2w/go2w_config.py
legged_gym/legged_gym/utils/task_registry.py
documents/go2w_reward_core_cleanup_no_legacy_alias_report.md
```
