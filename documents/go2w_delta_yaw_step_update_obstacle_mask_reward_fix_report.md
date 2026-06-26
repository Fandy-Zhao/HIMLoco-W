# go2W Delta-Yaw Command, Obstacle Mask, and Reward Cleanup Report

## 1. 修改动机

本次修改把 go2W/HIM 的第三维 command 从旧的 yaw-rate command 彻底改为 waypoint 相对朝向误差：

```python
commands[:, 0] = vx_cmd              # m/s
commands[:, 1] = vy_cmd              # m/s
commands[:, 2] = delta_yaw_to_goal   # rad, wrap_to_pi(target_yaw - base_yaw)
```

旧逻辑中 `commands[:, 2]` 曾被用作 `yaw_rate_cmd`，并与 `base_ang_vel[:, 2]` 做 tracking。该语义会把角度和角速度混用，导致 play/训练时机器人容易原地乱转、无法稳定朝 waypoint 推进。本次改为每步根据当前 base yaw 和当前 goal 重新计算 delta yaw，训练从头开始，不兼容旧 checkpoint。

## 2. 涉及文件

- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
- `legged_gym/legged_gym/envs/go2w/go2w_robot.py`
- `legged_gym/legged_gym/envs/go2w/go2w_config.py`
- `legged_gym/legged_gym/utils/task_registry.py`
- `legged_gym/legged_gym/scripts/play.py`

## 3. Command 语义变化

`commands[:, 2]` 现在只表示 `delta_yaw_to_goal`。采样 command 时只采样 `vx/vy`，随后调用 `_update_goal_commands()` 计算第三维。

核心实现：

```python
def _update_goal_commands(self):
    if not hasattr(self, 'env_goals') or not hasattr(self, 'cur_goal_idx') or self.terrain_goals is None:
        return
    env_ids = torch.arange(self.num_envs, device=self.device)
    cur_goal = self.env_goals[env_ids, self.cur_goal_idx]
    target_vec = cur_goal[:, :2] - self.root_states[:, :2]
    target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
    delta_yaw = wrap_to_pi(target_yaw - self._get_base_yaw())

    self.target_pos_rel[:] = target_vec
    self.target_yaw[:] = target_yaw
    self.goal_yaw_error[:] = delta_yaw
    self.commands[:, 2] = delta_yaw
    if hasattr(self, 'env_has_goals'):
        self.env_has_goals[:] = True
```

`_post_physics_step_callback()` 在每个 physics step 都会调用 `_update_goals()` 和 `_update_goal_commands()`，因此 delta yaw 不再只在 command resampling 时更新。

## 4. Observation 修改

actor command observation 仍保持 3 维，不改变 go2W 的总 observation 维度：

```python
def _get_command_obs(self):
    delta_yaw_scale = float(getattr(self.obs_scales, 'delta_yaw', getattr(self.obs_scales, 'heading_error', 1.0)))
    return torch.cat((
        self.commands[:, :2] * self.commands_scale[:2],
        self._get_delta_yaw_to_goal().unsqueeze(1) * delta_yaw_scale,
    ), dim=1)
```

维度保持：

- `num_one_step_observations = 73`
- `num_observations = 365`
- `num_privileged_obs = 263`

## 5. Reward 函数新增/替换

### `tracking_delta_yaw`

使用 cosine reward，避免角度误差和角速度混用：

```python
def _reward_tracking_delta_yaw(self):
    delta_yaw = self._get_delta_yaw_to_goal()
    return 0.5 * (torch.cos(delta_yaw) + 1.0)
```

### `goal_progress`

替代简单的 `tracking_goal_vel` / `tracking_goal_vel_cmd_scaled`，目标速度由 `vx/vy` command 大小决定，并支持 stop/slow command：

```python
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
```

### `yaw_rate_l2`

保留 yaw-rate 稳定正则，但不再作为 command tracking：

```python
def _reward_yaw_rate_l2(self):
    return torch.square(self.base_ang_vel[:, 2])
```

## 6. Reward 替换清单

- `tracking_ang_vel`: legacy，函数返回 0；Stage0/Stage2 scale 置 0。
- `tracking_heading`: replaced-by-new-reward，转发/禁用，Stage0/Stage2 不使用。
- `tracking_goal_yaw`: replaced-by-new-reward，语义由 `tracking_delta_yaw` 覆盖，Stage0/Stage2 不重复使用。
- `tracking_goal_vel`: replaced-by-new-reward，转发到 `goal_progress`，Stage0/Stage2 scale 置 0。
- `tracking_goal_vel_cmd_scaled`: replaced-by-new-reward，转发到 `goal_progress`，Stage0/Stage2 scale 置 0。
- `wheel_slip`: 保留函数名但改为 lateral-only wheel slip penalty，不再惩罚正常前向滚动。

## 7. Stage2 禁用的冗余 Reward

Stage2 中显式置 0：

- `tracking_ang_vel`
- `tracking_heading`
- `tracking_goal_yaw`
- `tracking_goal_vel`
- `tracking_goal_vel_cmd_scaled`
- `wheel_lateral_slip`
- `base_height_over_obstacle`
- `feet_air_time`
- `foot_clearance`
- `feet_clearance`
- `feet_stumble`
- `stumble`
- `wheel_torque`

没有删除这些函数，因为 Stage1、旧任务或配置仍可能引用它们。

## 8. Stage2 Reward Scales 覆盖表

| Reward | Stage2 scale |
| --- | ---: |
| `tracking_lin_vel` | `0.0` |
| `tracking_delta_yaw` | `0.8` |
| `goal_progress` | `3.0` |
| `reach_goal` | `1.5` |
| `finish_course` | `8.0` |
| `tracking_ang_vel` | `0.0` |
| `tracking_heading` | `0.0` |
| `tracking_goal_yaw` | `0.0` |
| `tracking_goal_vel` | `0.0` |
| `tracking_goal_vel_cmd_scaled` | `0.0` |
| `wheel_clearance` | `1.0` |
| `wheel_climb_drive` | `0.1` |
| `wheel_spin_without_progress` | `-0.02` |
| `wheel_slip` | `-0.3` |
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
| `termination` | `-0.8` |

关键参数：

```python
env_cfg.rewards.min_goal_speed = 0.15
env_cfg.rewards.max_goal_speed = 0.8
env_cfg.rewards.stop_cmd_threshold = 0.05
env_cfg.rewards.obstacle_height_offset = 0.06
env_cfg.rewards.obstacle_height_threshold = 0.04
env_cfg.rewards.gap_height_threshold = 0.06
env_cfg.rewards.wheel_clearance_target = 0.10
env_cfg.rewards.only_positive_rewards = False
env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
```

## 9. 接触、Obstacle Mask 与 Termination 修改

### Obstacle mask

不再用 `env_has_goals` 或 terrain type 作为障碍 mask。当前 mask 基于机器人目标方向前方若干探针点的高程差：

```python
def _get_obstacle_ahead_mask(self):
    distances = getattr(self.cfg.rewards, 'obstacle_probe_distances', [0.25, 0.40, 0.55, 0.70])
    distances = torch.tensor(distances, dtype=torch.float, device=self.device)
    if distances.numel() == 0:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    heading = self._get_base_yaw() + self.commands[:, 2]
    forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    points_xy = self.root_states[:, None, :2] + forward[:, None, :] * distances[None, :, None]
    ahead_heights = self._get_heights_at_points(points_xy)
    base_height = self._get_heights_at_points(self.root_states[:, None, :2])[:, 0]

    height_diff = ahead_heights - base_height.unsqueeze(1)
    obstacle_threshold = float(getattr(self.cfg.rewards, 'obstacle_height_threshold', 0.04))
    gap_threshold = float(getattr(self.cfg.rewards, 'gap_height_threshold', 0.06))
    step_or_wall = torch.any(height_diff > obstacle_threshold, dim=1)
    gap = torch.any(height_diff < -gap_threshold, dim=1)
    mask = step_or_wall | gap
    if hasattr(self, 'env_terrain_idx'):
        mask &= self.env_terrain_idx >= 0
    return mask
```

### `wheel_climb_drive`

使用 `wheel_forward_sign` 判断轮子正向驱动，且只在有移动 command、接触、前方有障碍、沿 goal 方向有进展时给奖励。

### final-goal reset

到达最后一个 goal 会置 `final_goal_reset_buf`，但 `_reward_termination()` 排除该 buffer，因此不会把完成课程当失败惩罚。`reset_idx()` 中会清空 `final_goal_reset_buf`。

## 10. 风险点

- 这是破坏性 command 语义修改，旧 yaw-rate checkpoint 不能继续加载使用。
- `commands[:, 2]` 是 body 当前朝向到 waypoint 方向的 delta yaw，不是 world-frame target yaw，也不是 yaw rate。
- 如果测试时手动缩小 `terrain.num_rows`，必须同步保证 `max_init_terrain_level <= num_rows - 1`，否则初始化 terrain level 会越界。
- Stage0/Stage2 已关闭 `tracking_lin_vel`，旧 command curriculum 现在在缺少该 reward 或 scale 为 0 时直接返回。
- `wheel_forward_sign` 当前默认全 1；如 URDF 轮轴方向后续确认有反向轮，需要按实际 DOF 顺序改 sign。

## 11. 验证步骤

已在 40 服务器 `/home/zhaozhuofan/test/HIMLoco-W` 的 `hw` 环境完成：

```bash
PYTHONPATH=legged_gym python -m compileall -q legged_gym/legged_gym
git diff --check
```

配置断言：

- Stage0: `tracking_lin_vel=0.0`, `tracking_delta_yaw=0.5`, `goal_progress=2.0`, `min_goal_speed=0.0`, `max_goal_speed=0.6`, `stop_cmd_threshold=0.05`
- Stage2: `tracking_lin_vel=0.0`, `tracking_delta_yaw=0.8`, `goal_progress=3.0`, `reach_goal=1.5`, `finish_course=8.0`, `only_positive_rewards=False`

2-env Stage2 smoke 已验证：

- `reset()` 返回 `(obs, privileged_obs)`，shape 为 `(2, 365)` 和 `(2, 263)`
- `step()` 返回 7 项
- `commands[:, 2] == goal_yaw_error == wrap_to_pi(target_yaw - base_yaw)`
- base 位置变化后 delta yaw 每步更新
- `tracking_delta_yaw(0)=1.0`，`tracking_delta_yaw(pi/2)=0.5`
- stop/slow command 下 `goal_progress` 目标速度为 0
- obstacle mask 对前方 step/gap 为 True，对 flat 为 False
- `wheel_climb_drive` 受 `wheel_forward_sign` 控制
- final-goal reset 不触发 termination penalty，reset 后 `final_goal_reset_buf` 清空

Stage0 smoke 已验证：

- `reset()` / `step()` 通过
- shape 为 `(2, 365)` / `(2, 263)`
- `tracking_lin_vel` 不参与，`tracking_delta_yaw` 和 `goal_progress` 生效
- stop command reward 接近 1，移动 command 但无进展时 reward 下降

Stage2 train smoke 已完成 1 次迭代：

- `HIMActorCritic` actor 输入 92
- critic 输入 263
- estimator 输入 365
- 1 次 rollout/learn 完成，无 reward 名称错误，无 command curriculum KeyError

## 12. 训练建议

从头训练新模型，命令示例：

```bash
cd /home/zhaozhuofan/test/HIMLoco-W
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hw
PYTHONPATH=legged_gym CUDA_VISIBLE_DEVICES=0 python legged_gym/legged_gym/scripts/train.py --task go2w --stage 0 --headless
PYTHONPATH=legged_gym CUDA_VISIBLE_DEVICES=0 python legged_gym/legged_gym/scripts/train.py --task go2w --stage 2 --headless
```

建议先 Stage0 确认 stop/slow command、delta-yaw facing 和基础推进稳定，再进入 Stage2 越障。play/eval 时不要再手动把 `commands[:, 2]` 当 yaw rate；如果需要手动测试方向，应理解为当前朝向到目标方向的 delta yaw，goal terrain 中会由 `_update_goal_commands()` 自动覆盖。
