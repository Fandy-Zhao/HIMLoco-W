# GO2W Stage0/Stage2 Dynamic Reward Tests Report

1. Test script path: `legged_gym/legged_gym/tests/debug_go2w_stage_tests.py`
2. Stage0 test result: `PASS`
3. Stage2 test result: `PASS`
13. Recommend Stage0 short training: `YES`
14. Recommend Stage2 short training: `YES`

## Stage 0

Overall status: **PASS**

| Check | Result |
| --- | --- |
| `delta_yaw_step_update` | PASS |
| `stage0_stop_slow` | PASS |
| `tracking_delta_yaw_cos` | PASS |
| `obstacle_mask_height_diff` | PASS |
| `base_height_offset_gating` | PASS |
| `wheel_slip_lateral_only` | PASS |
| `wheel_forward_sign` | PASS |
| `final_goal_reset_clear` | PASS |
| `commands2_update_path` | PASS |

4. delta_yaw every-step update verification: PASS
   - max_delta_yaw_error = `0.0`
   - commands[:,2] range = `[-0.5509533882141113, -0.1628732681274414]`
5. Stage0 stop/slow command verification: PASS
   - stop target_speed_mean = `0.0`
   - slow target_speed_mean = `0.0`
   - moving target_speed_mean = `0.20000000298023224`
6. tracking_delta_yaw cosine reward verification: PASS
   - rewards for [0, pi/2, -pi/2, pi] = `[1.0, 0.4999999701976776, 0.4999999701976776, 0.0]`
7. obstacle mask front height-difference verification: PASS
   - flat/runtime mask ratio = `0.0`
   - synthetic step/gap mask = `[False, True, True]`
8. base_height offset gating verification: PASS
   - base_target = `0.34`, obstacle_offset = `0.0`, mask_ratio = `0.0`
9. wheel_slip lateral-only verification: PASS
   - slip_forward = `0.0`, slip_lateral = `4.0`, runtime_mean = `0.04996639862656593`
10. wheel_climb_drive wheel forward-direction verification: PASS
   - wheel_names = `['FL_foot_joint', 'FR_foot_joint', 'RL_foot_joint', 'RR_foot_joint']`
   - wheel_forward_sign = `[1.0, 1.0, 1.0, 1.0]`
11. final_goal_reset_buf clear verification: PASS
   - failed_reset_reward_mean after normal failed reset = `1.0`
12. train/play commands[:,2] update-path grep verification: PASS
   - grep summary:
     - `legged_gym/legged_gym/envs/base/legged_robot_config.py:125: heading_command = True # legacy flag; commands[:, 2] now stores target yaw directly`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:514: return torch.clamp(self.commands[:, 2], -torch.pi, torch.pi)`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:620: self.commands[:, 2] = delta_yaw`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:1459: # Deprecated yaw-rate command tracking. commands[:, 2] is delta yaw.`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:1466: delta_yaw = self.commands[:, 2]`
     - `legged_gym/legged_gym/envs/go2w/go2w_robot.py:207: heading = self._get_base_yaw() + self.commands[:, 2]`
     - `legged_gym/legged_gym/scripts/play.py:50: env.commands[:, 2] = delta_yaw`
     - `legged_gym/legged_gym/scripts/play.py:146: 'delta_yaw_cmd': env.commands[robot_index, 2].item(),`

## Stage 2

Overall status: **PASS**

| Check | Result |
| --- | --- |
| `delta_yaw_step_update` | PASS |
| `stage0_stop_slow` | PASS |
| `tracking_delta_yaw_cos` | PASS |
| `obstacle_mask_height_diff` | PASS |
| `base_height_offset_gating` | PASS |
| `wheel_slip_lateral_only` | PASS |
| `wheel_forward_sign` | PASS |
| `final_goal_reset_clear` | PASS |
| `commands2_update_path` | PASS |

4. delta_yaw every-step update verification: PASS
   - max_delta_yaw_error = `0.0`
   - commands[:,2] range = `[-1.3901185989379883, -0.1645526885986328]`
5. Stage0 stop/slow command verification: PASS
   - stop target_speed_mean = `0.0`
   - slow target_speed_mean = `0.0`
   - moving target_speed_mean = `0.20000000298023224`
6. tracking_delta_yaw cosine reward verification: PASS
   - rewards for [0, pi/2, -pi/2, pi] = `[1.0, 0.4999999701976776, 0.4999999701976776, 0.0]`
7. obstacle mask front height-difference verification: PASS
   - flat/runtime mask ratio = `0.5`
   - synthetic step/gap mask = `[False, True, True]`
8. base_height offset gating verification: PASS
   - base_target = `0.34`, obstacle_offset = `0.06`, mask_ratio = `0.5`
9. wheel_slip lateral-only verification: PASS
   - slip_forward = `0.0`, slip_lateral = `4.0`, runtime_mean = `0.006178292911499739`
10. wheel_climb_drive wheel forward-direction verification: PASS
   - wheel_names = `['FL_foot_joint', 'FR_foot_joint', 'RL_foot_joint', 'RR_foot_joint']`
   - wheel_forward_sign = `[1.0, 1.0, 1.0, 1.0]`
11. final_goal_reset_buf clear verification: PASS
   - failed_reset_reward_mean after normal failed reset = `1.0`
12. train/play commands[:,2] update-path grep verification: PASS
   - grep summary:
     - `legged_gym/legged_gym/envs/base/legged_robot_config.py:125: heading_command = True # legacy flag; commands[:, 2] now stores target yaw directly`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:514: return torch.clamp(self.commands[:, 2], -torch.pi, torch.pi)`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:620: self.commands[:, 2] = delta_yaw`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:1459: # Deprecated yaw-rate command tracking. commands[:, 2] is delta yaw.`
     - `legged_gym/legged_gym/envs/base/legged_robot.py:1466: delta_yaw = self.commands[:, 2]`
     - `legged_gym/legged_gym/envs/go2w/go2w_robot.py:207: heading = self._get_base_yaw() + self.commands[:, 2]`
     - `legged_gym/legged_gym/scripts/play.py:50: env.commands[:, 2] = delta_yaw`
     - `legged_gym/legged_gym/scripts/play.py:146: 'delta_yaw_cmd': env.commands[robot_index, 2].item(),`

## Readiness

All Stage0 and Stage2 dynamic reward tests passed.

- Stage0 short training: 5k-20k iterations.
- Stage2 short training: from Stage0 checkpoint, low obstacle curriculum.
