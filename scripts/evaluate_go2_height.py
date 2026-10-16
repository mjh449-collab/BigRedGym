"""Check height transitions and movement separately from training rewards."""

import argparse
import json

import torch

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.helpers import get_load_path
from gym.utils.task_registry import task_registry


def summarize(height, velocity, height_target, command, terminated):
    height_error = (height - height_target).square().mean().sqrt().item()
    error = velocity - command
    linear_error = error[:, :2].square().sum(1).mean().sqrt().item()
    yaw_error = error[:, 2].square().mean().sqrt().item()
    finite = bool(torch.isfinite(height).all() and torch.isfinite(velocity).all())
    return {
        "height_target_m": float(height_target),
        "height_actual_m": height.mean().item(),
        "height_rmse_m": height_error,
        "velocity_command": command.tolist(),
        "velocity_actual": velocity.mean(0).tolist(),
        "linear_rmse_m_s": linear_error,
        "yaw_rmse_rad_s": yaw_error,
        "terminated": bool(terminated),
        "passed": finite
        and not bool(terminated)
        and height_error <= 0.03
        and linear_error <= 0.2
        and yaw_error <= 0.25,
    }


@torch.no_grad()
def evaluate(env, runner, segment_seconds=4.0):
    lo, hi = env.cfg.commands.ranges.base_height
    middle = env.cfg.reward_settings.base_height_target
    # Two stationary replicas alternate low/high in opposite orders. The other
    # robots must walk or turn at middle height; standing still cannot pass.
    commands = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.5]],
        device=env.device,
        dtype=torch.float,
    )
    labels = ["height_cycle_a", "height_cycle_b", "forward", "strafe", "turn"]
    failed = torch.zeros(5, device=env.device, dtype=torch.bool)
    env.phase_frequency[:] = 2.0
    results = []
    for segment in range(4):
        endpoints = [lo, hi] if segment % 2 == 0 else [hi, lo]
        targets = torch.tensor(endpoints + [middle] * 3, device=env.device)
        env.commands[:] = commands
        env.base_height_command[:, 0] = targets
        heights, velocities = [], []
        for _ in range(round(segment_seconds / env.dt)):
            runner.set_actions(
                runner.actor_cfg["actions"], runner.get_inference_actions()
            )
            env.step()
            failed |= env.terminated
            heights.append(env.base_height[:, 0].clone())
            velocities.append(
                torch.cat(
                    (env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]), 1
                ).clone()
            )
        # Judge the last second, giving each height transition time to settle.
        count = round(1.0 / env.dt)
        height = torch.stack(heights)[-count:].cpu()
        velocity = torch.stack(velocities)[-count:].cpu()
        for i, label in enumerate(labels):
            results.append(
                {"case": label, "segment": segment + 1}
                | summarize(
                    height[:, i],
                    velocity[:, i],
                    targets[i].cpu(),
                    commands[i].cpu(),
                    failed[i],
                )
            )
    return {"passed": all(case["passed"] for case in results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default="go2trot")
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("mujoco", "vsim"), default="mujoco")
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    args = parser.parse_args()
    if args.segment_seconds < 2.0:
        parser.error("each segment must last at least two seconds")
    import gym.envs  # noqa: F401

    cfg, train = task_registry.get_cfgs(
        "go2trot",
        original_cfg=True,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )
    cfg.env.num_envs = 5
    cfg.env.episode_length_s = 4 * args.segment_seconds + 1
    cfg.commands.resampling_time = cfg.env.episode_length_s + 1
    cfg.push_robots.toggle = False
    cfg.seed = 123
    apply_domain_randomization_override(cfg, "off")
    train.runner.device = args.device
    train.runner.resume = False
    train.logging.enable_local_saving = False
    train.log_dir = None
    task_registry.convert_frequencies_to_params(cfg, train)
    env = task_registry.make_env("go2trot", cfg, args.device, True, args.backend)
    try:
        runner = task_registry.make_alg_runner(env, train)
        checkpoint = get_load_path(args.experiment_name, args.load_run, args.checkpoint)
        runner.load(checkpoint, load_optimizer=False)
        runner.switch_to_eval()
        report = {"checkpoint": checkpoint} | evaluate(
            env, runner, args.segment_seconds
        )
        print(json.dumps(report, indent=2))
    finally:
        env._backend.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
