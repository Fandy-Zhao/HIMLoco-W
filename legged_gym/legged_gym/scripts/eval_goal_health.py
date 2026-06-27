import csv
import os
from pathlib import Path
from collections import Counter

import torch
import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def parse_args():
    return get_args()


def tensor_list(x):
    if x is None:
        return []
    if hasattr(x, 'detach'):
        x = x.detach().cpu()
    return x.numpy().tolist() if hasattr(x, 'numpy') else list(x)


def unhealthy_reason(row):
    if not row['raw_success']:
        return 'not_raw_success'
    reasons = []
    if row['yaw_error_final'] >= 0.6:
        reasons.append('yaw')
    if not (0.18 <= row['base_height_final'] <= 0.65):
        reasons.append('base_height')
    if row['roll_pitch_proxy_final'] >= 0.65:
        reasons.append('roll_pitch')
    if row['wheel_slip_final'] >= 0.8:
        reasons.append('wheel_slip')
    if row['collision_or_stumble']:
        reasons.append('collision')
    if row['timeout']:
        reasons.append('timeout')
    return '+'.join(reasons) if reasons else 'healthy'


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 512)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    obs = env.get_observations()

    rows = []
    max_steps = int(env.max_episode_length) * max(2, int(args.eval_episodes // env.num_envs) + 2)
    for _ in range(max_steps):
        if args.fixed_vx is not None:
            env.commands[:, 0] = args.fixed_vx
            env.commands[:, 1] = args.fixed_vy
            if hasattr(env, '_update_goal_commands'):
                env._update_goal_commands()
        actions = policy(obs.detach())
        obs, _, _, dones, infos, _, _ = env.step(actions.detach())
        details = infos.get('episode_details') if isinstance(infos, dict) else None
        if details:
            n = len(tensor_list(details.get('env_ids')))
            fields = {k: tensor_list(v) for k, v in details.items()}
            for i in range(n):
                row = {
                    'episode_index': len(rows),
                    'env_id': int(fields['env_ids'][i]),
                    'raw_success': bool(fields['raw_success'][i]),
                    'healthy_success': bool(fields['healthy_success'][i]),
                    'goal_dist_final': float(fields['goal_dist_final'][i]),
                    'yaw_error_final': float(fields['yaw_error_final'][i]),
                    'base_height_final': float(fields['base_height_final'][i]),
                    'roll_pitch_proxy_final': float(fields['roll_pitch_proxy_final'][i]),
                    'wheel_slip_final': float(fields['wheel_slip_final'][i]),
                    'collision_or_stumble': bool(fields['collision_or_stumble'][i]),
                    'timeout': bool(fields['timeout'][i]),
                    'episode_length': float(fields['episode_length'][i]),
                }
                row['time_to_goal'] = row['episode_length'] * env.dt
                row['unhealthy_reason'] = unhealthy_reason(row)
                rows.append(row)
        if len(rows) >= args.eval_episodes:
            break

    rows = rows[:args.eval_episodes]
    csv_path = out_dir / 'goal_health_episodes.csv'
    fieldnames = ['episode_index','env_id','raw_success','healthy_success','unhealthy_reason','goal_dist_final','yaw_error_final','base_height_final','roll_pitch_proxy_final','wheel_slip_final','collision_or_stumble','timeout','episode_length','time_to_goal']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    raw = [r for r in rows if r['raw_success']]
    healthy = [r for r in rows if r['healthy_success']]
    unhealthy = [r for r in rows if r['raw_success'] and not r['healthy_success']]
    reasons = Counter(r['unhealthy_reason'] for r in unhealthy)
    def avg(seq, key):
        return sum(r[key] for r in seq) / len(seq) if seq else 0.0
    report = [
        '# Goal Health Evaluation', '',
        f'- task: `{args.task}`',
        f'- stage: `{args.stage}`',
        f'- load_run: `{args.load_run}`',
        f'- checkpoint: `{args.checkpoint}`',
        f'- episodes: {len(rows)}',
        f'- raw_success_rate: {len(raw) / max(len(rows), 1):.4f}',
        f'- healthy_success_rate: {len(healthy) / max(len(rows), 1):.4f}',
        f'- unhealthy_success_rate: {len(unhealthy) / max(len(rows), 1):.4f}',
        f'- mean_time_to_goal_raw_success: {avg(raw, "time_to_goal"):.4f}',
        f'- mean_wheel_slip_raw_success: {avg(raw, "wheel_slip_final"):.4f}',
        f'- mean_yaw_error_raw_success: {avg(raw, "yaw_error_final"):.4f}',
        f'- timeout_ratio: {sum(r["timeout"] for r in rows) / max(len(rows), 1):.4f}',
        f'- collision_or_stumble_ratio: {sum(r["collision_or_stumble"] for r in rows) / max(len(rows), 1):.4f}',
        '', '## Unhealthy Reason Breakdown', '',
    ]
    for reason, count in reasons.most_common():
        report.append(f'- {reason}: {count}')
    report += ['', f'Episodes CSV: `{csv_path}`', '']
    report_path = out_dir / 'goal_health_report.md'
    report_path.write_text('\n'.join(report), encoding='utf-8')
    print(report_path)
    print(csv_path)
    print('\n'.join(report[:16]))


if __name__ == '__main__':
    main()
