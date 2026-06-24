# HIMLoco-W 奖励设计

本文档说明 HIMLoco-W 的奖励设计，包括通用奖励计算流程，以及 `go2w` 任务的具体奖励项和权重。

## 1. 奖励计算流程

### 1.1. 奖励准备
在 `legged_gym/legged_gym/envs/base/legged_robot.py` 中：

- `self.reward_scales = class_to_dict(self.cfg.rewards.scales)`
- `self._prepare_reward_function()` 在环境初始化时调用。
- 该函数会移除值为 0 的奖励尺度，并将所有非零尺度乘以 `dt`。
- 它根据可用的奖励名称构建 `self.reward_functions` 和 `self.reward_names`。

### 1.2. 奖励计算
在 `legged_gym/legged_gym/envs/base/legged_robot.py` 中：

- `compute_reward()` 的执行流程为：
  1. `self.rew_buf[:] = 0.`
  2. 对每个 `self.reward_functions` 中的奖励函数：
     - `rew = self.reward_functions[i]() * self.reward_scales[name]`
     - `self.rew_buf += rew`
     - `self.episode_sums[name] += rew`
  3. 如果 `self.cfg.rewards.only_positive_rewards` 启用，则将负奖励裁剪为 0。
  4. 如果 reward scales 中包含 `termination`，则在裁剪后添加终止奖励。

### 1.3. 重要行为

- 奖励项乘以 `dt`，因此尺度已被时间步长归一化。
- 每个奖励项返回一个大小为 `num_envs` 的向量。
- 最终奖励是所有加权奖励项的累加。
- 每个奖励项的 episode 累计值会单独记录在 `episode_sums` 中。

## 2. 奖励项列表

基础环境在 `legged_robot.py` 中定义了大量奖励函数。

### 2.1. 目标跟踪与任务奖励

- `tracking_goal_vel`
  - 奖励朝当前目标方向前进的速度分量。
  - 计算机器人基座到目标方向上的速度投影。
  - 负值被裁剪为 0。
  - 如果存在 `env_has_goals`，则对于无目标的环境，该奖励为 0。

- `tracking_goal_yaw`
  - 奖励与目标航向的方向一致性。
  - `exp(-abs(goal_yaw_error))`。
  - 无目标时该奖励为 0。

- `reach_goal`
  - 到达目标时的二值奖励。
  - 目标达成时为 `1.0`，否则为 `0`。

- `finish_course`
  - 完成最后一个目标点时的二值奖励。
  - 仅在 `cur_goal_idx >= num_goals - 1` 且 `reached_goal` 时生效。

### 2.2. 轮子相关奖励

- `wheel_torque`
  - 轮子力矩平方和。
  - 仅在存在轮子索引时有效。
  - 会屏蔽无效地形。

- `wheel_vel_smooth`
  - 轮子加速度平方和。
  - 鼓励轮速变化平滑。
  - 仅在存在轮子索引时有效。

- `wheel_slip`
  - 轮子打滑速度之和。
  - 通过接触力阈值识别接触轮子。
  - 对无效地形上的打滑给予惩罚。

### 2.3. 指令跟踪奖励

- `tracking_lin_vel`
  - 水平线速度跟踪奖励。
  - `exp(-||command_xy - base_lin_vel_xy||^2 / tracking_sigma)`。
  - 对带目标环境，该奖励会减少 80%。

- `tracking_ang_vel`
  - 偏航角速度跟踪奖励。
  - `exp(-(command_yaw_rate - base_ang_vel_z)^2 / tracking_sigma)`。

### 2.4. 稳定性与正则化惩罚

- `lin_vel_z`
  - 惩罚垂直方向基座速度。
  - `square(base_lin_vel_z)`。

- `ang_vel_xy`
  - 惩罚横滚和俯仰角速度。
  - `sum(square(base_ang_vel_xy))`。

- `orientation`
  - 惩罚基座非直立姿态。
  - `sum(square(projected_gravity_xy))`。

- `dof_acc`
  - 惩罚关节加速度。
  - 对腿部加速度求和，若包含轮子则附加轮子加速度并乘以 `wheel_acc_weight`。

- `joint_power`
  - 惩罚驱动功率：`abs(dof_vel) * abs(torque)`。
  - 腿部功率加轮子功率，轮子项乘以 `wheel_torque_weight`。

- `base_height`
  - 惩罚基座高度偏离目标高度。
  - `square(base_height - base_height_target)`。
  - 某些地形或目标状态下，该惩罚会减半。

- `foot_clearance`
  - 惩罚足端高度与期望清晰高度偏差乘以横向足速。
  - 鼓励足端在运动时保持安全高度。

- `action_rate`
  - 惩罚连续动作变化。
  - 腿部动作加轮子动作项，轮子项乘以 `wheel_action_rate_weight`。

- `smoothness`
  - 惩罚二阶动作变化。
  - 与 `action_rate` 类似，但使用两步差分。

- `torques`
  - 惩罚作用力矩平方。
  - 腿部力矩加轮子力矩，轮子项乘以 `wheel_torque_weight`。

- `dof_vel`
  - 惩罚关节速度。
  - 若存在轮子关节，则仅对腿部关节计算。

- `collision`
  - 惩罚受限身体的接触碰撞。
  - 当接触力超过 0.1 时计为碰撞。

### 2.5. 限制与安全惩罚

- `termination`
  - 仅在 episode 终止时的终止惩罚/奖励。
  - 返回 `reset_buf * ~time_out_buf`。

- `dof_pos_limits`
  - 惩罚关节位置越过范围限制。
  - 计算超过上下限的违例量之和。

- `dof_vel_limits`
  - 惩罚超过软速度限制的关节速度。
  - 使用 `soft_dof_vel_limit` 的软阈值。

- `torque_limits`
  - 惩罚超过软力矩限制的力矩。
  - 轮子项乘以 `wheel_torque_weight`。

- `feet_air_time`
  - 奖励更长的空中时间步态。
  - 只有在接触恢复且空中时间大于 0.5s 时累计。
  - 仅在指令水平速度非零时给予奖励。

- `stumble`
  - 惩罚足端撞击垂直表面。
  - 当水平接触力明显大于竖直接触力时认为发生绊倒。

- `stand_still`
  - 惩罚指令接近零时仍有动作。
  - 使用关节姿态偏离默认姿态的程度。

- `feet_contact_forces`
  - 惩罚超过 `max_contact_force` 的足端接触力。

## 3. 基础配置默认值

基础奖励配置定义在 `legged_gym/legged_gym/envs/base/legged_robot_config.py`。

重要默认字段：

- `only_positive_rewards = True`
  - 在添加终止奖励前将总奖励裁剪为非负。
- `tracking_sigma = 0.25`
- `soft_dof_pos_limit = 0.9`
- `soft_dof_vel_limit = 0.9`
- `soft_torque_limit = 1.0`
- `max_contact_force = 100.0`
- `wheel_torque_weight = 0.5`
- `wheel_acc_weight = 0.2`
- `wheel_action_rate_weight = 0.5`

## 4. `go2w` 任务奖励设置

`go2w` 环境配置位于 `legged_gym/legged_gym/envs/go2w/go2w_config.py`，其设置为：

- `only_positive_rewards = False`
- `tracking_sigma = 0.4`
- `soft_dof_pos_limit = 0.9`
- `soft_dof_vel_limit = 0.9`
- `soft_torque_limit = 1.0`
- `base_height_target = 0.34`
- `max_contact_force = 100.0`
- `wheel_torque_weight = 0.5`
- `wheel_acc_weight = 0.2`
- `wheel_action_rate_weight = 0.5`

### `go2w` 的奖励尺度权重

- `termination = -0.8`
- `tracking_lin_vel = 1.0`
- `tracking_ang_vel = 0.5`
- `lin_vel_z = -2.0`
- `ang_vel_xy = -0.05`
- `orientation = -1.0`
- `torques = -1e-5`
- `dof_vel = -1e-4`
- `dof_acc = -2.5e-7`
- `base_height = -0.5`
- `feet_air_time = 0.0`
- `foot_clearance = 0.0`
- `feet_clearance = 0.0`
- `collision = -0.5`
- `feet_stumble = 0.0`
- `stumble = 0.0`
- `action_rate = -0.01`
- `stand_still = -0.01`
- `dof_pos_limits = -0.9`
- `dof_vel_limits = -0.0`
- `torque_limits = -0.0`
- `arm_pos = -0.0`
- `hip_action_l2 = -0.1`
- `tracking_goal_vel = 1.0`
- `tracking_goal_yaw = 0.3`
- `reach_goal = 0.5`
- `finish_course = 0.0`
- `wheel_torque = 0.0`
- `wheel_vel_smooth = -1e-4`
- `wheel_slip = -0.03`

## 5. 奖励项如何组合

每个环境步：

- 计算每个激活的奖励项 `r_i`。
- 乘以对应权重 `w_i`。
- 总奖励：`reward = sum_i w_i * r_i`。
- 若启用了 `only_positive_rewards`，先对总奖励进行非负裁剪，再添加终止奖励。
- 终止奖励最后添加。

## 6. 分析要点

- `go2w` 中 `only_positive_rewards = False`，因此允许负奖励。
- `tracking_lin_vel` 和 `tracking_ang_vel` 是主要的正向任务奖励。
- `termination` 是对环境重置的小负惩罚。
- 多数正则化项是小的负值，用于鼓励稳定、平滑和安全行为。
- `tracking_goal_*` 项支持存在目标时的目标驱动行为。
- 轮子相关项大多数当前被弱化或禁用（如 `wheel_torque`、`finish_course`）。

## 7. 推荐补充

如果你想获得更详细的每回合奖励报告，环境已经记录了每个奖励项的 `episode_sums[name]`。可以进一步通过 episode info 或 TensorBoard 将其输出。
