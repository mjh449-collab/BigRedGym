"""Evaluate the Assignment 6 Go2 payload rating or a hardware stress case."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from go2_deploy.deploy_config import DeployConfig
from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry
from scripts.evaluate_go2trot import record_traces, summarize_case
from scripts.payload_traces import attach_payload


COMMAND = (1.0, 0.0, 0.0)
HEIGHT = 0.30
GAIT_HZ = 2.0


def summarize_payload(trace, dt, warmup_steps):
    command = torch.tensor(COMMAND)
    gait = summarize_case(trace, dt, warmup_steps, GAIT_HZ, command)
    valid = trace["alive"].clone()
    valid[:warmup_steps] = False
    result = {"survival": gait["survival"], "samples": gait["samples"]}
    if not valid.any() or "residual_jitter_rms_rad" not in gait:
        return result

    speed = trace["velocity"][..., 0][valid]
    torque = trace["torque_fraction"][valid]
    mean_height = trace["height"][valid].mean().item()
    return result | {
        "mean_forward_speed_m_s": speed.mean().item(),
        "forward_rmse_m_s": (speed - COMMAND[0]).square().mean().sqrt().item(),
        "mean_height_m": mean_height,
        "height_error_m": abs(mean_height - HEIGHT),
        "swing_load_fraction": gait["swing_load_fraction"],
        "stride_load_max_deviation_p90": gait["stride_load_max_deviation_p90"],
        "complete_strides": gait["complete_strides"],
        "torque_p95_fraction_by_joint": torque.quantile(0.95, dim=0).tolist(),
        "any_joint_at_limit_fraction": (torque >= 1 - 1e-6)
        .any(dim=-1)
        .float()
        .mean()
        .item(),
        "jitter_rms_rad": gait["residual_jitter_rms_rad"],
    }


def rate_case(result, unloaded):
    if "forward_rmse_m_s" not in result:
        return result | {"passed": False, "failures": ["insufficient samples"]}
    if "mean_forward_speed_m_s" not in unloaded:
        return result | {"passed": False, "failures": ["missing unloaded reference"]}

    speed_delta = abs(
        result["mean_forward_speed_m_s"] - unloaded["mean_forward_speed_m_s"]
    )
    sag = unloaded["mean_height_m"] - result["mean_height_m"]
    checks = {
        "forward_rmse": result["forward_rmse_m_s"] <= 0.20 + 1e-6,
        "speed_relative_to_unloaded": speed_delta <= 0.10 + 1e-6,
        "height_error": result["height_error_m"] <= 0.03 + 1e-6,
        "sag_relative_to_unloaded": sag <= 0.015 + 1e-6,
        "survival": result["survival"] >= 0.95 - 1e-6,
        "swing_load": result["swing_load_fraction"] <= 0.20 + 1e-6,
        "stride_load_balance": result["complete_strides"] > 0
        and result["stride_load_max_deviation_p90"] <= 0.12 + 1e-6,
        "torque_p95": max(result["torque_p95_fraction_by_joint"]) <= 0.80 + 1e-6,
        "torque_saturation": result["any_joint_at_limit_fraction"] <= 0.01 + 1e-6,
    }
    return result | {
        "speed_delta_from_unloaded_m_s": speed_delta,
        "additional_sag_m": sag,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "jitter_passed": result["jitter_rms_rad"] <= 0.02 + 1e-6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--original_cfg", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--suite", choices=("nominal", "randomized", "stress"), default="nominal"
    )
    parser.add_argument("--masses", type=float, nargs="+")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--com-offset-m", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--policy-hz", type=int, help="Override both policy and physics rate"
    )
    parser.add_argument("--gains", type=float, nargs=2, metavar=("KP", "KD"))
    parser.add_argument("--ema-alpha", type=float, help="Set to 1 for no smoothing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nominal = args.suite == "nominal"
    masses = (
        args.masses
        if args.masses is not None
        else ([0, 2.5, 5, 7.5, 10] if nominal else None)
    )
    replicas = args.replicas if args.replicas is not None else (20 if nominal else 100)
    if not masses or any(not 0 <= mass < float("inf") for mass in masses):
        parser.error("provide finite nonnegative payload masses")
    if (
        len(set(masses)) != len(masses)
        or replicas < 1
        or not 5 <= args.seconds < float("inf")
    ):
        parser.error("use unique masses, positive replicas, and at least 5 seconds")
    if nominal and 0 not in masses:
        parser.error("the nominal assessment needs the 0 kg reference")
    if args.com_offset_m is not None:
        if args.suite != "stress" or not all(map(math.isfinite, args.com_offset_m)):
            parser.error("use a finite COM offset only with --suite stress")
    if args.policy_hz is not None and args.policy_hz <= 20:
        parser.error("--policy-hz must exceed 20 Hz to measure jitter above 10 Hz")
    if args.gains is not None and not all(0 < gain < math.inf for gain in args.gains):
        parser.error("--gains must be finite and positive")
    if args.ema_alpha is not None and not 0 < args.ema_alpha <= 1:
        parser.error("--ema-alpha must be in (0, 1]")

    import gym.envs  # noqa: F401

    env_cfg, train_cfg = copy.deepcopy(
        task_registry.get_cfgs(
            "go2trot",
            original_cfg=args.original_cfg,
            experiment_name=args.experiment_name,
            load_run=args.load_run,
        )
    )
    env_cfg.env.num_envs = len(masses) * replicas
    if args.policy_hz is not None:
        env_cfg.control.ctrl_frequency = args.policy_hz
        env_cfg.control.desired_sim_frequency = args.policy_hz
    if args.gains is not None:
        kp, kd = args.gains
        env_cfg.control.stiffness = dict.fromkeys(env_cfg.control.stiffness, kp)
        env_cfg.control.damping = dict.fromkeys(env_cfg.control.damping, kd)
    if args.ema_alpha is not None:
        env_cfg.control.action_ema_alpha = args.ema_alpha
    env_cfg.env.episode_length_s = args.seconds + 1
    env_cfg.commands.resampling_time = args.seconds + 1
    env_cfg.commands.ranges.base_height = [HEIGHT, HEIGHT]
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if getattr(env_cfg.domain_randomization, "payload", None) is None:
        attach_payload(env_cfg)
    payload = env_cfg.domain_randomization.payload
    payload.mass_range = [min(masses), max(masses)]
    if nominal:
        apply_domain_randomization_override(env_cfg, "payload-only")
        payload.com_offset_range = [[0.0, 0.0]] * 3
    if args.com_offset_m is not None:
        payload.com_offset_range = [[offset, offset] for offset in args.com_offset_m]

    env_cfg.seed = args.seed
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    train_cfg.log_dir = None
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    set_seed(args.seed)
    path = Path(get_load_path(args.experiment_name, args.load_run, args.checkpoint))
    env = task_registry.make_env("go2trot", env_cfg, args.device, True, "mujoco")
    try:
        ids = torch.arange(env.num_envs, device=env.device)
        fixed_mass = torch.tensor(masses, dtype=torch.float, device=env.device)
        env._backend.set_payload(
            ids,
            fixed_mass.repeat_interleave(replicas),
            env._backend.payload_com.clone(),
        )
        env._update_payload_observation()
        runner = task_registry.make_alg_runner(env, train_cfg)
        state = torch.load(path, weights_only=True, map_location=env.device)
        runner.alg.actor.load_state_dict(state["actor_state_dict"])
        runner.switch_to_eval()
        commands = torch.tensor(COMMAND, device=env.device).repeat(env.num_envs, 1)
        frequencies = torch.full((env.num_envs,), GAIT_HZ, device=env.device)
        trace = record_traces(env, runner, commands, frequencies, args.seconds)
    finally:
        env._backend.close()

    results = {}
    for i, mass in enumerate(masses):
        selection = slice(i * replicas, (i + 1) * replicas)
        results[f"{mass:g}"] = summarize_payload(
            {key: value[:, selection] for key, value in trace.items()},
            env.dt,
            round(2 / env.dt),
        )
    if nominal:
        unloaded = results["0"]
        results = {mass: rate_case(case, unloaded) for mass, case in results.items()}
    else:
        threshold = 0.95 if args.suite == "randomized" else 0.80
        for case in results.values():
            case.update(
                survival_threshold=threshold,
                passed=case["survival"] >= threshold - 1e-6,
            )

    control = env_cfg.control
    deploy = DeployConfig()
    report = {
        "protocol": "assignment6-payload-v1",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arguments": vars(args)
        | {"masses": masses, "replicas": replicas, "output": str(args.output)},
        "joint_order": list(env.actuated_dof_names),
        "controller": {
            "policy_hz": control.ctrl_frequency,
            "simulation_hz": 1 / env_cfg.sim_dt,
            "stiffness": control.stiffness,
            "damping": control.damping,
            "action_ema_alpha": getattr(control, "action_ema_alpha", 1.0),
        },
        "payload_mount_m": payload.mount_pos,
        "payload_com_offset_ranges_m": payload.com_offset_range,
        "configured_deploy_matches": {
            "observations": list(train_cfg.actor.obs) == deploy.obs_vector,
            "policy_rate": control.ctrl_frequency == deploy.ctrl_freq,
            "nominal_gains": set(control.stiffness.values()) == {deploy.kp}
            and set(control.damping.values()) == {deploy.kd},
            "action_ema": getattr(control, "action_ema_alpha", 1.0)
            == deploy.ema_smoothing_factor,
        },
        "cases_by_payload_kg": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
