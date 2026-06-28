import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import torch


def to_jsonable(obj, _visited=None):
    if _visited is None:
        _visited = set()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value, _visited) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(item, _visited) for item in obj]
    if callable(obj):
        return str(obj)
    if hasattr(obj, "__dict__") or isinstance(obj, type):
        return class_to_dict(obj, _visited)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def class_to_dict(obj, _visited=None):
    if _visited is None:
        _visited = set()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    obj_id = id(obj)
    if obj_id in _visited:
        return str(obj)
    _visited.add(obj_id)
    try:
        if isinstance(obj, dict):
            return {str(key): class_to_dict(value, _visited) for key, value in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [class_to_dict(item, _visited) for item in obj]
        if isinstance(obj, (Path, Enum, torch.device, torch.dtype, np.ndarray, np.generic)) or torch.is_tensor(obj):
            return to_jsonable(obj, _visited)

        result = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(obj, key)
            except Exception:
                continue
            if callable(value):
                continue
            if isinstance(value, type(sys)):
                continue
            result[key] = class_to_dict(value, _visited)
        if result:
            return result
        return to_jsonable(obj, _visited)
    finally:
        _visited.discard(obj_id)


def _safe_getattr(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _run_git(repo_root, args):
    try:
        completed = subprocess.run(
            ["git"] + list(args),
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return {"error": str(exc)}


def get_git_info(repo_root):
    info = {
        "commit": None,
        "branch": None,
        "is_dirty": None,
        "status_short": None,
        "diff_stat": None,
    }
    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_short = _run_git(repo_root, ["status", "--short"])
    diff_stat = _run_git(repo_root, ["diff", "--stat"])
    diff_quiet = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    info["commit"] = commit if isinstance(commit, str) else None
    info["branch"] = branch if isinstance(branch, str) else None
    info["status_short"] = status_short if isinstance(status_short, str) else status_short
    info["diff_stat"] = diff_stat if isinstance(diff_stat, str) else diff_stat
    if diff_quiet.returncode in (0, 1):
        info["is_dirty"] = diff_quiet.returncode == 1
    else:
        info["is_dirty"] = {"error": diff_quiet.stderr.strip() or f"returncode={diff_quiet.returncode}"}
    return to_jsonable(info)


def get_system_info(device):
    gpu_names = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                gpu_names.append(torch.cuda.get_device_name(index))
            except Exception as exc:
                gpu_names.append(f"<error: {exc}>")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "device": str(device),
    }


def count_parameters(model):
    if model is None:
        return {"total": 0, "trainable": 0}
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _linear_dims(module):
    dims = []
    if module is None:
        return dims
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            dims.append({"in_features": int(child.in_features), "out_features": int(child.out_features)})
    return dims


def _module_activation_name(module):
    if module is None:
        return None
    for child in module.modules():
        if child is module:
            continue
        if isinstance(
            child,
            (
                torch.nn.ELU,
                torch.nn.SELU,
                torch.nn.ReLU,
                torch.nn.LeakyReLU,
                torch.nn.Tanh,
                torch.nn.Sigmoid,
                torch.nn.SiLU,
            ),
        ):
            return child.__class__.__name__
    return None


def get_model_metadata(actor_critic):
    if actor_critic is None:
        return {}

    parameter_counts = count_parameters(actor_critic)
    modules = {}
    for name in [
        "actor", "actor_body", "leg_head", "wheel_head", "critic", "estimator",
        "adaptation_module", "history_encoder", "priv_encoder", "target",
    ]:
        module = getattr(actor_critic, name, None)
        if module is not None:
            modules[name] = str(module)

    network_config = {
        "actor_linear_layers": _linear_dims(getattr(actor_critic, "actor", None)),
        "critic_linear_layers": _linear_dims(getattr(actor_critic, "critic", None)),
        "activation": _module_activation_name(getattr(actor_critic, "actor", None)),
        "init_noise_std": float(actor_critic.std.detach().mean().cpu().item()) if hasattr(actor_critic, "std") else None,
        "history_size": int(getattr(actor_critic, "history_size", 0)) if hasattr(actor_critic, "history_size") else None,
        "num_actor_obs": int(getattr(actor_critic, "num_actor_obs", 0)) if hasattr(actor_critic, "num_actor_obs") else None,
        "num_one_step_obs": int(getattr(actor_critic, "num_one_step_obs", 0)) if hasattr(actor_critic, "num_one_step_obs") else None,
        "split_action_head": bool(getattr(actor_critic, "split_action_head", False)),
        "actor_body_linear_layers": _linear_dims(getattr(actor_critic, "actor_body", None)),
        "leg_head_linear_layers": _linear_dims(getattr(actor_critic, "leg_head", None)),
        "wheel_head_linear_layers": _linear_dims(getattr(actor_critic, "wheel_head", None)),
    }
    estimator = getattr(actor_critic, "estimator", None)
    if estimator is not None:
        network_config["estimator_encoder_linear_layers"] = _linear_dims(getattr(estimator, "encoder", None))
        network_config["estimator_target_linear_layers"] = _linear_dims(getattr(estimator, "target", None))
        network_config["estimator_num_latent"] = int(getattr(estimator, "num_latent", 0)) if hasattr(estimator, "num_latent") else None
        network_config["estimator_num_prototype"] = int(getattr(getattr(estimator, "proto", None), "num_embeddings", 0)) if hasattr(getattr(estimator, "proto", None), "num_embeddings") else None

    return {
        "class": actor_critic.__class__.__name__,
        "repr": str(actor_critic),
        "num_parameters": parameter_counts["total"],
        "num_trainable_parameters": parameter_counts["trainable"],
        "modules": modules,
        "state_dict_keys": list(actor_critic.state_dict().keys()),
        "network_config": to_jsonable(network_config),
    }


def get_reward_metadata(env, env_cfg):
    reward_cfg = getattr(env_cfg, "rewards", None)
    reward_scales_cfg = class_to_dict(_safe_getattr(reward_cfg, "scales", {}))

    reward_params = {}
    if reward_cfg is not None:
        for key in dir(reward_cfg):
            if key.startswith("_") or key == "scales":
                continue
            value = getattr(reward_cfg, key)
            if callable(value):
                continue
            reward_params[key] = to_jsonable(value)

    runtime_reward_scales = {}
    runtime_reward_names = []
    missing_nonzero = []
    zero_scale_rewards_removed = []

    for name, scale in reward_scales_cfg.items():
        try:
            numeric_scale = float(scale)
        except Exception:
            continue
        if numeric_scale == 0.0:
            zero_scale_rewards_removed.append(name)

    if env is not None:
        runtime_reward_scales = to_jsonable(getattr(env, "reward_scales", {}))
        runtime_reward_names = list(getattr(env, "reward_names", []))
        if not runtime_reward_names and hasattr(env, "reward_functions"):
            runtime_reward_names = [
                function.__name__.replace("_reward_", "")
                for function in getattr(env, "reward_functions", [])
                if hasattr(function, "__name__")
            ]
        for name, scale in getattr(env, "reward_scales", {}).items():
            if float(scale) == 0.0:
                continue
            if not hasattr(env, "_reward_" + name) and name != "termination":
                missing_nonzero.append(name)

    return {
        "scales": reward_scales_cfg,
        "runtime_reward_scales": runtime_reward_scales,
        "runtime_reward_names": runtime_reward_names,
        "params": reward_params,
        "validation": {
            "missing_nonzero_reward_functions": missing_nonzero,
            "zero_scale_rewards_removed": zero_scale_rewards_removed,
        },
    }


def _command_metadata(env_cfg, env):
    asset_name = _safe_getattr(_safe_getattr(env_cfg, "asset", None), "name", None)
    semantics = {
        "commands[:,0]": "vx_cmd",
        "commands[:,1]": "vy_cmd",
        "commands[:,2]": "yaw_rate_cmd",
    }
    delta_yaw_update = None
    if asset_name == "go2w":
        semantics["commands[:,2]"] = "delta_yaw_to_goal"
        delta_yaw_update = "updated every step from current base yaw and next goal"

    return {
        "num_commands": _safe_getattr(_safe_getattr(env_cfg, "commands", None), "num_commands", None),
        "command_semantics": semantics,
        "ranges": class_to_dict(_safe_getattr(_safe_getattr(env_cfg, "commands", None), "ranges", {})),
        "resampling_time": _safe_getattr(_safe_getattr(env_cfg, "commands", None), "resampling_time", None),
        "delta_yaw_update": delta_yaw_update,
        "runtime_commands": to_jsonable(getattr(env, "commands", None)[:1] if env is not None and hasattr(env, "commands") else None),
    }


def _terrain_metadata(env_cfg, env):
    terrain_cfg = _safe_getattr(env_cfg, "terrain", None)
    return {
        "terrain_cfg_full": class_to_dict(terrain_cfg),
        "mesh_type": _safe_getattr(terrain_cfg, "mesh_type", None),
        "terrain_proportions": to_jsonable(_safe_getattr(terrain_cfg, "terrain_proportions", None)),
        "terrain_extra_proportions": to_jsonable(_safe_getattr(terrain_cfg, "terrain_extra_proportions", None)),
        "curriculum": _safe_getattr(terrain_cfg, "curriculum", None),
        "num_goals": _safe_getattr(terrain_cfg, "num_goals", None),
        "goal_threshold": _safe_getattr(terrain_cfg, "next_goal_threshold", None),
        "env_terrain_idx": to_jsonable(getattr(env, "env_terrain_idx", None)[:16] if env is not None and hasattr(env, "env_terrain_idx") else None),
    }


def _asset_metadata(env_cfg, env):
    asset_cfg = _safe_getattr(env_cfg, "asset", None)
    return {
        "file": _safe_getattr(asset_cfg, "file", None),
        "name": _safe_getattr(asset_cfg, "name", None),
        "foot_name": to_jsonable(_safe_getattr(asset_cfg, "foot_name", None)),
        "wheel_name": to_jsonable(_safe_getattr(asset_cfg, "wheel_name", None)),
        "wheel_body_name": to_jsonable(_safe_getattr(asset_cfg, "wheel_body_name", None)),
        "penalize_contacts_on": to_jsonable(_safe_getattr(asset_cfg, "penalize_contacts_on", None)),
        "terminate_after_contacts_on": to_jsonable(_safe_getattr(asset_cfg, "terminate_after_contacts_on", None)),
        "self_collisions": _safe_getattr(asset_cfg, "self_collisions", None),
        "fix_base_link": _safe_getattr(asset_cfg, "fix_base_link", None),
        "default_dof_drive_mode": _safe_getattr(asset_cfg, "default_dof_drive_mode", None),
        "runtime_wheel_body_names": to_jsonable(getattr(env, "wheel_body_names", None)),
    }


def _observations_metadata(env_cfg, env):
    env_section = _safe_getattr(env_cfg, "env", None)
    normalization = _safe_getattr(env_cfg, "normalization", None)
    terrain_cfg = _safe_getattr(env_cfg, "terrain", None)
    return {
        "num_observations": _safe_getattr(env_section, "num_observations", None),
        "num_privileged_obs": _safe_getattr(env_section, "num_privileged_obs", None),
        "num_one_step_observations": _safe_getattr(env_section, "num_one_step_observations", None),
        "history_len": int(getattr(env, "history_length", 0)) if env is not None and hasattr(env, "history_length") else None,
        "obs_scales": class_to_dict(_safe_getattr(normalization, "obs_scales", {})),
        "command_obs_semantics": ["vx_cmd", "vy_cmd", "delta_yaw_to_goal" if _safe_getattr(_safe_getattr(env_cfg, "asset", None), "name", None) == "go2w" else "yaw_rate_cmd"],
        "uses_height_measurements": bool(_safe_getattr(terrain_cfg, "measure_heights", False)),
    }


def _action_metadata(env_cfg, env):
    control_cfg = _safe_getattr(env_cfg, "control", None)
    env_section = _safe_getattr(env_cfg, "env", None)
    return {
        "num_actions": _safe_getattr(env_section, "num_actions", None),
        "action_scale": _safe_getattr(control_cfg, "action_scale", None),
        "control_type": _safe_getattr(control_cfg, "control_type", None),
        "stiffness": to_jsonable(_safe_getattr(control_cfg, "stiffness", None)),
        "damping": to_jsonable(_safe_getattr(control_cfg, "damping", None)),
        "decimation": _safe_getattr(control_cfg, "decimation", None),
        "leg_dof_indices": to_jsonable(getattr(env, "leg_dof_indices", None)),
        "wheel_dof_indices": to_jsonable(getattr(env, "wheel_dof_indices", None)),
        "wheel_forward_sign": to_jsonable(getattr(env, "wheel_forward_sign", None)),
    }


def _env_metadata(env_cfg):
    env_section = _safe_getattr(env_cfg, "env", None)
    control_cfg = _safe_getattr(env_cfg, "control", None)
    sim_cfg = _safe_getattr(env_cfg, "sim", None)
    dt = _safe_getattr(sim_cfg, "dt", None)
    decimation = _safe_getattr(control_cfg, "decimation", None)
    control_frequency = None
    if dt and decimation:
        try:
            control_frequency = 1.0 / (float(dt) * float(decimation))
        except Exception:
            control_frequency = None
    return {
        "num_envs": _safe_getattr(env_section, "num_envs", None),
        "num_observations": _safe_getattr(env_section, "num_observations", None),
        "num_privileged_obs": _safe_getattr(env_section, "num_privileged_obs", None),
        "num_actions": _safe_getattr(env_section, "num_actions", None),
        "episode_length_s": _safe_getattr(env_section, "episode_length_s", None),
        "control_frequency": control_frequency,
        "sim_dt": dt,
        "decimation": decimation,
        "env_cfg_full": class_to_dict(env_cfg),
    }


def _algorithm_metadata(train_cfg, runner):
    algorithm_cfg = _safe_getattr(train_cfg, "algorithm", None)
    alg = getattr(runner, "alg", None) if runner is not None else None
    return {
        "algorithm_class": alg.__class__.__name__ if alg is not None else None,
        "num_learning_epochs": _safe_getattr(algorithm_cfg, "num_learning_epochs", getattr(alg, "num_learning_epochs", None)),
        "num_mini_batches": _safe_getattr(algorithm_cfg, "num_mini_batches", getattr(alg, "num_mini_batches", None)),
        "clip_param": _safe_getattr(algorithm_cfg, "clip_param", getattr(alg, "clip_param", None)),
        "gamma": _safe_getattr(algorithm_cfg, "gamma", getattr(alg, "gamma", None)),
        "lam": _safe_getattr(algorithm_cfg, "lam", getattr(alg, "lam", None)),
        "entropy_coef": _safe_getattr(algorithm_cfg, "entropy_coef", getattr(alg, "entropy_coef", None)),
        "learning_rate": _safe_getattr(algorithm_cfg, "learning_rate", getattr(alg, "learning_rate", None)),
        "max_grad_norm": _safe_getattr(algorithm_cfg, "max_grad_norm", getattr(alg, "max_grad_norm", None)),
        "desired_kl": _safe_getattr(algorithm_cfg, "desired_kl", getattr(alg, "desired_kl", None)),
        "schedule": _safe_getattr(algorithm_cfg, "schedule", getattr(alg, "schedule", None)),
    }


def _task_stage_metadata(args, env_cfg, train_cfg):
    env_stage = _safe_getattr(_safe_getattr(env_cfg, "env", None), "learning_stage", None)
    runner_stage = _safe_getattr(_safe_getattr(train_cfg, "runner", None), "learning_stage", None)
    return {
        "task_name": getattr(args, "task", None),
        "experiment_name": _safe_getattr(_safe_getattr(train_cfg, "runner", None), "experiment_name", None),
        "run_name": _safe_getattr(_safe_getattr(train_cfg, "runner", None), "run_name", None),
        "args_stage": getattr(args, "stage", None) if hasattr(args, "stage") else getattr(args, "learning_stage", None),
        "env_learning_stage": env_stage,
        "runner_learning_stage": runner_stage,
    }


def _paths_metadata(log_dir, metadata_path, runner):
    return {
        "cwd": os.getcwd(),
        "log_dir": log_dir,
        "metadata_path": metadata_path,
        "checkpoint_dir": log_dir,
        "resume_path": getattr(runner, "resume_path", None) if runner is not None else None,
    }


def _command_line_metadata(args):
    return {
        "argv": sys.argv,
        "args": to_jsonable(vars(args)),
    }


def _notes_metadata(env_cfg, env, runner):
    asset_name = _safe_getattr(_safe_getattr(env_cfg, "asset", None), "name", None)
    notes = {}
    if asset_name == "go2w":
        notes["commands_2_semantics"] = "commands[:, 2] = delta_yaw_to_goal, not yaw_rate"
    if env is not None and hasattr(env, "reward_names"):
        notes["runtime_reward_names"] = list(env.reward_names)
    if runner is not None:
        notes["runner_class"] = runner.__class__.__name__
    return notes


def _repo_root_from_log_dir(log_dir):
    if log_dir is None:
        return Path.cwd()
    current = Path(log_dir).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def save_train_metadata(args, env_cfg, train_cfg, env=None, runner=None, actor_critic=None, log_dir=None):
    try:
        resolved_log_dir = log_dir or getattr(runner, "log_dir", None)
        if resolved_log_dir is None:
            experiment_name = _safe_getattr(_safe_getattr(train_cfg, "runner", None), "experiment_name", "default")
            run_name = _safe_getattr(_safe_getattr(train_cfg, "runner", None), "run_name", "default")
            resolved_log_dir = os.path.join("logs", experiment_name, run_name)

        metadata_dir = os.path.join(resolved_log_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "train_metadata.json")
        repo_root = _repo_root_from_log_dir(resolved_log_dir)
        device = getattr(runner, "device", getattr(args, "rl_device", "cpu"))

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "command_line": _command_line_metadata(args),
            "paths": _paths_metadata(resolved_log_dir, metadata_path, runner),
            "git": get_git_info(repo_root),
            "system": get_system_info(device),
            "task": _task_stage_metadata(args, env_cfg, train_cfg),
            "stage": _task_stage_metadata(args, env_cfg, train_cfg),
            "env": _env_metadata(env_cfg),
            "train": {
                "train_cfg_full": class_to_dict(train_cfg),
            },
            "model": get_model_metadata(actor_critic),
            "algorithm": _algorithm_metadata(train_cfg, runner),
            "reward": get_reward_metadata(env, env_cfg),
            "command": _command_metadata(env_cfg, env),
            "terrain": _terrain_metadata(env_cfg, env),
            "asset": _asset_metadata(env_cfg, env),
            "normalization": class_to_dict(_safe_getattr(env_cfg, "normalization", {})),
            "observations": _observations_metadata(env_cfg, env),
            "action": _action_metadata(env_cfg, env),
            "notes": _notes_metadata(env_cfg, env, runner),
        }

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(metadata), handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[metadata] saved train metadata to {metadata_path}")
        return metadata_path
    except Exception as exc:
        print(f"[metadata] warning: failed to save train metadata: {exc}")
        return None


def save_latest_checkpoint_metadata(checkpoint_path, runner):
    try:
        log_dir = getattr(runner, "log_dir", None)
        if log_dir is None:
            return None
        metadata_dir = os.path.join(log_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        latest_checkpoint_path = os.path.join(metadata_dir, "latest_checkpoint.json")

        env = getattr(runner, "env", None)
        env_cfg = getattr(env, "cfg", None)
        payload = {
            "latest_checkpoint": checkpoint_path,
            "iteration": getattr(runner, "last_saved_iteration", getattr(runner, "current_learning_iteration", None)),
            "timestamp": datetime.now().isoformat(),
            "total_timesteps": getattr(runner, "tot_timesteps", None),
            "reward_scales": to_jsonable(getattr(env, "reward_scales", None)),
            "stage": {
                "env_learning_stage": _safe_getattr(_safe_getattr(env_cfg, "env", None), "learning_stage", None),
                "asset_name": _safe_getattr(_safe_getattr(env_cfg, "asset", None), "name", None),
            },
        }
        with open(latest_checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        return latest_checkpoint_path
    except Exception as exc:
        print(f"[metadata] warning: failed to save latest checkpoint metadata: {exc}")
        return None
