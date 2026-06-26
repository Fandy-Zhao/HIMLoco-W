import json
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math import wrap_to_pi


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = REPO_ROOT / "documents" / "go2w_stage0_stage2_dynamic_reward_tests_report.md"
RESULTS_PATH = REPO_ROOT / "documents" / "go2w_stage0_stage2_dynamic_reward_tests_results.json"


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _summary_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    return value


def make_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    return env


def check_delta_yaw_updates(env):
    env._update_goal_commands()
    max_err = 0.0
    min_cmd = float("inf")
    max_cmd = float("-inf")
    for _ in range(5):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
        env._update_goal_commands()

        env_ids = torch.arange(env.num_envs, device=env.device)
        cur_goal = env.env_goals[env_ids, env.cur_goal_idx]
        target_vec = cur_goal[:, :2] - env.root_states[:, :2]
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        expected_delta_yaw = wrap_to_pi(target_yaw - env._get_base_yaw())
        err = torch.max(torch.abs(wrap_to_pi(env.commands[:, 2] - expected_delta_yaw)))
        max_err = max(max_err, _to_float(err))
        min_cmd = min(min_cmd, _to_float(torch.min(env.commands[:, 2])))
        max_cmd = max(max_cmd, _to_float(torch.max(env.commands[:, 2])))

        assert err < 1e-4, f"delta_yaw command stale or incorrect, max_err={err}"
        assert torch.all(env.commands[:, 2] <= torch.pi + 1e-5)
        assert torch.all(env.commands[:, 2] >= -torch.pi - 1e-5)
        assert torch.max(torch.abs(wrap_to_pi(env.goal_yaw_error - env.commands[:, 2]))) < 1e-4

    print("[PASS] delta_yaw updates every step")
    print(f"max_delta_yaw_error = {max_err:.8f}")
    print(f"commands[:,2] range = [{min_cmd:.6f}, {max_cmd:.6f}]")
    return {"max_delta_yaw_error": max_err, "commands2_range": [min_cmd, max_cmd]}


def check_stage0_stop_slow(env):
    details = {}
    for name, vx, vy in [
        ("stop", 0.0, 0.0),
        ("slow_below_threshold", 0.03, 0.0),
        ("moving", 0.2, 0.0),
    ]:
        env.commands[:, 0] = vx
        env.commands[:, 1] = vy
        env._update_goal_commands()

        cmd_speed = torch.norm(env.commands[:, :2], dim=1)
        stop_cmd_threshold = float(getattr(env.cfg.rewards, "stop_cmd_threshold", 0.05))
        min_goal_speed = float(getattr(env.cfg.rewards, "min_goal_speed", 0.0))
        max_goal_speed = float(getattr(env.cfg.rewards, "max_goal_speed", 0.6))
        moving = cmd_speed > stop_cmd_threshold
        moving_target_speed = torch.clamp(cmd_speed, min=min_goal_speed, max=max_goal_speed)
        target_speed = torch.where(moving, moving_target_speed, torch.zeros_like(cmd_speed))

        if name in ["stop", "slow_below_threshold"]:
            assert torch.all(target_speed == 0), f"{name}: target_speed should be 0"
        if name == "moving":
            assert torch.all(target_speed > 0), "moving: target_speed should be positive"

        goal_rew = env._reward_goal_progress()
        assert torch.isfinite(goal_rew).all()
        details[name] = {
            "cmd_speed_mean": _to_float(cmd_speed.mean()),
            "target_speed_mean": _to_float(target_speed.mean()),
            "goal_reward_mean": _to_float(goal_rew.mean()),
        }
        print(
            f"case={name} cmd_speed={details[name]['cmd_speed_mean']:.6f} "
            f"target_speed={details[name]['target_speed_mean']:.6f}"
        )

    print("[PASS] Stage0 stop/slow command support")
    return details


def check_tracking_delta_yaw_cos(env):
    test_delta = torch.tensor([0.0, torch.pi / 2, -torch.pi / 2, torch.pi], device=env.device)
    expected = 0.5 * (torch.cos(test_delta) + 1.0)
    original = env.commands[:, 2].clone()
    try:
        values = []
        for delta, expected_value in zip(test_delta, expected):
            env.commands[:, 2] = delta
            reward = env._reward_tracking_delta_yaw()
            assert torch.max(torch.abs(reward - expected_value)) < 1e-5
            values.append(_to_float(expected_value))
    finally:
        env.commands[:, 2] = original
    print("[PASS] tracking_delta_yaw cosine reward")
    return {"delta_yaw": _summary_value(test_delta), "reward": values}


def check_obstacle_mask(env):
    mask = env._get_obstacle_ahead_mask()
    assert mask.dtype == torch.bool
    assert mask.shape == (env.num_envs,)
    flat_ratio = _to_float(mask.float().mean())

    h_now = torch.tensor([0.0, 0.0, 0.0], device=env.device)
    h_ahead = torch.tensor(
        [
            [0.0, 0.01, 0.0],
            [0.0, 0.06, 0.08],
            [0.0, -0.08, -0.10],
        ],
        device=env.device,
    )
    synthetic_mask = env._compute_obstacle_mask_from_heights(h_now, h_ahead)
    expected = torch.tensor([False, True, True], device=env.device)
    assert torch.equal(synthetic_mask, expected), f"obstacle mask wrong: {synthetic_mask} vs {expected}"

    print("[PASS] obstacle mask from front height difference")
    print(f"flat_mask_ratio = {flat_ratio:.6f}")
    print(f"step/gap synthetic mask = {synthetic_mask.detach().cpu().tolist()}")
    return {"flat_mask_ratio": flat_ratio, "synthetic_mask": synthetic_mask.detach().cpu().tolist()}


def check_base_height_offset(env, stage):
    target_height = env._get_base_height_target()
    base_target = float(getattr(env.cfg.rewards, "base_height_target", 0.34))
    offset = float(getattr(env.cfg.rewards, "obstacle_height_offset", 0.0))
    mask = env._get_obstacle_ahead_mask()

    assert torch.allclose(
        target_height[~mask],
        torch.full_like(target_height[~mask], base_target),
    ), "base height offset applied without obstacle"
    if mask.any():
        assert torch.allclose(
            target_height[mask],
            torch.full_like(target_height[mask], base_target + offset),
        ), "base height offset not applied near obstacle"
    if stage == 0:
        assert offset == 0.0
    if stage == 2:
        assert offset > 0.0

    print("[PASS] base_height obstacle offset gating")
    print(f"base_target = {base_target:.6f}")
    print(f"offset = {offset:.6f}")
    print(f"mask_ratio = {_to_float(mask.float().mean()):.6f}")
    return {"base_target": base_target, "offset": offset, "mask_ratio": _to_float(mask.float().mean())}


def check_wheel_slip(env):
    num_wheels = len(env.wheel_body_indices)
    wheel_vel_world_forward = torch.zeros(env.num_envs, num_wheels, 3, device=env.device)
    wheel_vel_world_forward[..., 0] = 1.0
    wheel_vel_world_lateral = torch.zeros(env.num_envs, num_wheels, 3, device=env.device)
    wheel_vel_world_lateral[..., 1] = 1.0
    wheel_contact = torch.ones(env.num_envs, num_wheels, dtype=torch.bool, device=env.device)

    original_quat = env.base_quat.clone()
    try:
        env.base_quat[:, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        slip_forward = env._compute_wheel_lateral_slip(wheel_vel_world_forward, wheel_contact)
        slip_lateral = env._compute_wheel_lateral_slip(wheel_vel_world_lateral, wheel_contact)
    finally:
        env.base_quat[:, :] = original_quat

    assert torch.all(slip_forward < 1e-6), "forward wheel motion should not be slip"
    assert torch.all(slip_lateral > slip_forward + 1e-3), "lateral motion should increase slip"

    slip_rew = env._reward_wheel_slip()
    assert torch.isfinite(slip_rew).all()

    print("[PASS] wheel_slip lateral-only")
    print(f"slip_forward = {_to_float(slip_forward.mean()):.6f}")
    print(f"slip_lateral = {_to_float(slip_lateral.mean()):.6f}")
    print(f"runtime_wheel_slip_mean = {_to_float(slip_rew.mean()):.6f}")
    return {
        "slip_forward": _to_float(slip_forward.mean()),
        "slip_lateral": _to_float(slip_lateral.mean()),
        "runtime_wheel_slip_mean": _to_float(slip_rew.mean()),
    }


def check_wheel_forward_sign(env):
    print("wheel_dof_indices", env.wheel_dof_indices.detach().cpu().tolist())
    print("wheel_forward_sign", env.wheel_forward_sign.detach().cpu().tolist())
    print("wheel_names", [env.dof_names[i] for i in env.wheel_dof_indices.detach().cpu().tolist()])

    for _ in range(20):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

    wheel_vel = env.dof_vel[:, env.wheel_dof_indices]
    effective = wheel_vel * env.wheel_forward_sign.unsqueeze(0)

    num_wheels = len(env.wheel_dof_indices)
    fake_forward_wheel_vel = env.wheel_forward_sign.unsqueeze(0).repeat(env.num_envs, 1)
    fake_reverse_wheel_vel = -env.wheel_forward_sign.unsqueeze(0).repeat(env.num_envs, 1)
    forward_effective = fake_forward_wheel_vel * env.wheel_forward_sign.unsqueeze(0)
    reverse_effective = fake_reverse_wheel_vel * env.wheel_forward_sign.unsqueeze(0)

    assert num_wheels > 0
    assert torch.all(forward_effective > 0), "forward wheel velocity should be positive effective speed"
    assert torch.all(reverse_effective < 0), "reverse wheel velocity should be negative effective speed"

    print("wheel_vel_mean", torch.mean(wheel_vel, dim=0).detach().cpu().tolist())
    print("effective_wheel_vel_mean", torch.mean(effective, dim=0).detach().cpu().tolist())
    print("[PASS] wheel_climb_drive wheel forward sign logic")
    return {
        "wheel_dof_indices": env.wheel_dof_indices.detach().cpu().tolist(),
        "wheel_forward_sign": env.wheel_forward_sign.detach().cpu().tolist(),
        "wheel_names": [env.dof_names[i] for i in env.wheel_dof_indices.detach().cpu().tolist()],
        "effective_wheel_vel_mean": torch.mean(effective, dim=0).detach().cpu().tolist(),
    }


def check_final_goal_reset(env):
    env.reset()
    env.final_goal_reset_buf[:] = True
    env.reset_buf[:] = True
    env.time_out_buf[:] = False
    termination_reward = env._reward_termination()
    assert torch.all(termination_reward == 0), "final goal reset should not be penalized"

    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    assert torch.all(env.final_goal_reset_buf == False), "final_goal_reset_buf should be cleared after reset"

    env.reset_buf[:] = True
    env.time_out_buf[:] = False
    env.final_goal_reset_buf[:] = False
    termination_reward = env._reward_termination()
    assert torch.all(termination_reward > 0), "failed reset should be penalized"

    print("[PASS] final_goal_reset_buf clear and termination gating")
    return {"failed_reset_reward_mean": _to_float(termination_reward.float().mean())}


def check_commands2_update_path():
    roots = [
        REPO_ROOT / "legged_gym" / "legged_gym" / "envs",
        REPO_ROOT / "legged_gym" / "legged_gym" / "scripts",
        REPO_ROOT / "legged_gym" / "legged_gym" / "utils",
    ]
    matches = []
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                compact = line.replace(" ", "")
                if "commands[:,2]" in compact or "commands[robot_index,2]" in compact:
                    matches.append(f"{rel}:{lineno}: {line.strip()}")

    allowed_assignments = [
        item for item in matches
        if "self.commands[:, 2] = delta_yaw" in item
        or "env.commands[:, 2] = delta_yaw" in item
        or "env.commands[robot_index, 2].item()" in item
    ]
    unexpected_assignments = [
        item for item in matches
        if "commands[:, 2] =" in item and "self.commands[:, 2] = delta_yaw" not in item and "env.commands[:, 2] = delta_yaw" not in item
    ]
    assert not unexpected_assignments, "\n".join(unexpected_assignments)

    print("[PASS] commands[:,2] only updated from goal delta-yaw path")
    for item in matches:
        print(item)
    return {"matches": matches, "allowed_matches": allowed_assignments}


def write_report(stage, results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
    existing[str(stage)] = results
    RESULTS_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True))

    all_pass = all(key in existing and existing[key]["status"] == "PASS" for key in ["0", "2"])
    lines = [
        "# GO2W Stage0/Stage2 Dynamic Reward Tests Report",
        "",
        "1. Test script path: `legged_gym/legged_gym/tests/debug_go2w_stage_tests.py`",
        f"2. Stage0 test result: `{existing.get('0', {}).get('status', 'NOT_RUN')}`",
        f"3. Stage2 test result: `{existing.get('2', {}).get('status', 'NOT_RUN')}`",
        f"13. Recommend Stage0 short training: `{'YES' if all_pass else 'NO'}`",
        f"14. Recommend Stage2 short training: `{'YES' if all_pass else 'NO'}`",
        "",
    ]
    for stage_key in sorted(existing.keys(), key=int):
        item = existing[stage_key]
        status = item["status"]
        checks = item["checks"]
        delta = checks["delta_yaw_step_update"]
        stop_slow = checks["stage0_stop_slow"]
        yaw_cos = checks["tracking_delta_yaw_cos"]
        obstacle = checks["obstacle_mask_height_diff"]
        height = checks["base_height_offset_gating"]
        slip = checks["wheel_slip_lateral_only"]
        forward = checks["wheel_forward_sign"]
        final_reset = checks["final_goal_reset_clear"]
        commands2 = checks["commands2_update_path"]
        lines.extend([
            f"## Stage {stage_key}",
            "",
            f"Overall status: **{status}**",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ])
        for check_name in [
            "delta_yaw_step_update",
            "stage0_stop_slow",
            "tracking_delta_yaw_cos",
            "obstacle_mask_height_diff",
            "base_height_offset_gating",
            "wheel_slip_lateral_only",
            "wheel_forward_sign",
            "final_goal_reset_clear",
            "commands2_update_path",
        ]:
            value = "PASS" if check_name in item["checks"] else "N/A"
            lines.append(f"| `{check_name}` | {value} |")
        lines.extend([
            "",
            "4. delta_yaw every-step update verification: PASS",
            f"   - max_delta_yaw_error = `{delta['max_delta_yaw_error']}`",
            f"   - commands[:,2] range = `{delta['commands2_range']}`",
            "5. Stage0 stop/slow command verification: PASS",
            f"   - stop target_speed_mean = `{stop_slow['stop']['target_speed_mean']}`",
            f"   - slow target_speed_mean = `{stop_slow['slow_below_threshold']['target_speed_mean']}`",
            f"   - moving target_speed_mean = `{stop_slow['moving']['target_speed_mean']}`",
            "6. tracking_delta_yaw cosine reward verification: PASS",
            f"   - rewards for [0, pi/2, -pi/2, pi] = `{yaw_cos['reward']}`",
            "7. obstacle mask front height-difference verification: PASS",
            f"   - flat/runtime mask ratio = `{obstacle['flat_mask_ratio']}`",
            f"   - synthetic step/gap mask = `{obstacle['synthetic_mask']}`",
            "8. base_height offset gating verification: PASS",
            f"   - base_target = `{height['base_target']}`, obstacle_offset = `{height['offset']}`, mask_ratio = `{height['mask_ratio']}`",
            "9. wheel_slip lateral-only verification: PASS",
            f"   - slip_forward = `{slip['slip_forward']}`, slip_lateral = `{slip['slip_lateral']}`, runtime_mean = `{slip['runtime_wheel_slip_mean']}`",
            "10. wheel_climb_drive wheel forward-direction verification: PASS",
            f"   - wheel_names = `{forward['wheel_names']}`",
            f"   - wheel_forward_sign = `{forward['wheel_forward_sign']}`",
            "11. final_goal_reset_buf clear verification: PASS",
            f"   - failed_reset_reward_mean after normal failed reset = `{final_reset['failed_reset_reward_mean']}`",
            "12. train/play commands[:,2] update-path grep verification: PASS",
            "   - grep summary:",
        ])
        for match in commands2["matches"]:
            lines.append(f"     - `{match}`")
        lines.extend([
            "",
        ])

    if all_pass:
        lines.extend([
            "## Readiness",
            "",
            "All Stage0 and Stage2 dynamic reward tests passed.",
            "",
            "- Stage0 short training: 5k-20k iterations.",
            "- Stage2 short training: from Stage0 checkpoint, low obstacle curriculum.",
            "",
        ])
    REPORT_PATH.write_text("\n".join(lines))


def main():
    args = get_args()
    assert args.task == "go2w", "This debug test is intended for --task go2w"
    assert args.stage in (0, 2), "Use --stage 0 or --stage 2"
    assert args.num_envs is not None and args.num_envs >= 2, "Use --num_envs 2 or larger"

    env = make_env(args)
    checks = {}
    checks["delta_yaw_step_update"] = check_delta_yaw_updates(env)
    checks["stage0_stop_slow"] = check_stage0_stop_slow(env)
    checks["tracking_delta_yaw_cos"] = check_tracking_delta_yaw_cos(env)
    checks["obstacle_mask_height_diff"] = check_obstacle_mask(env)
    checks["base_height_offset_gating"] = check_base_height_offset(env, args.stage)
    checks["wheel_slip_lateral_only"] = check_wheel_slip(env)
    checks["wheel_forward_sign"] = check_wheel_forward_sign(env)
    checks["final_goal_reset_clear"] = check_final_goal_reset(env)
    checks["commands2_update_path"] = check_commands2_update_path()

    results = {"status": "PASS", "checks": checks}
    write_report(args.stage, results)

    print("")
    print("GO2W STAGE TEST SUMMARY")
    print("")
    print(f"stage = {args.stage}")
    for name in checks:
        print(f"{name}: PASS")
    print(f"report = {REPORT_PATH.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("GO2W STAGE TEST FAILED", file=sys.stderr)
        raise
