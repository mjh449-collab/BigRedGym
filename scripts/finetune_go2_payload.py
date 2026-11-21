"""Warm-start Go2Trot at the fixed deployment height from a saved run."""

import copy
import hashlib
import json
from pathlib import Path

import torch

from gym.utils.helpers import get_load_path
from gym.utils.task_registry import task_registry
from scripts.train import make_train_parser, setup, train


HEIGHT_INPUT = "base_height_target"


def fold_height_input(state, column, value):
    """Absorb a constant normalized-by-scale observation into the first bias."""
    weight = state["NN.0.weight"]
    bias = state["NN.0.bias"]
    return state | {
        "NN.0.weight": torch.cat((weight[:, :column], weight[:, column + 1 :]), dim=1),
        "NN.0.bias": bias + weight[:, column] * value,
    }


def model_weights(state, source, target, env, source_scaling):
    source_obs = list(source.obs)
    target_obs = list(target["obs"])
    if source.normalize_obs or target["normalize_obs"]:
        raise ValueError(
            "This warm start requires models without running normalization"
        )
    if source.activation != target["activation"]:
        raise ValueError("Saved and target models must use the same activation")
    for field in target_obs:
        before = torch.as_tensor(getattr(source_scaling, field, 1.0))
        after = torch.as_tensor(env.scales.get(field, 1.0)).cpu()
        if not torch.equal(before, after):
            raise ValueError(f"Observation scale changed for {field}")
    if source_obs == target_obs:
        return state
    if [field for field in source_obs if field != HEIGHT_INPUT] != target_obs:
        raise ValueError("Saved observations do not match the fixed-height model")
    column = sum(
        env.get_state(field).shape[1]
        for field in source_obs[: source_obs.index(HEIGHT_INPUT)]
    )
    low, high = env.command_ranges["base_height"]
    if low != high:
        raise ValueError("Cannot fold a height input when the target varies")
    return fold_height_input(
        state, column, low / getattr(source_scaling, HEIGHT_INPUT, 1.0)
    )


def main():
    parser = make_train_parser()
    parser.description = __doc__
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-checkpoint", type=int, default=-1)
    parser.add_argument("--rollout-size", type=int)
    parser.add_argument(
        "--rating-focus",
        action="store_true",
        help="Fine-tune for straight 2 Hz trotting near the 1 m/s payload rating.",
    )
    args = parser.parse_args()
    if (
        args.task != "go2trot"
        or args.resume
        or args.original_cfg
        or args.load_run
        or args.checkpoint is not None
    ):
        parser.error("use --task go2trot and select the old run with --source-run")
    if args.rollout_size is not None and args.rollout_size < 1:
        parser.error("--rollout-size must be positive")

    import gym.envs  # noqa: F401

    source_env, source_train = task_registry.get_cfgs(
        args.task,
        original_cfg=True,
        experiment_name="go2trot",
        load_run=args.source_run,
    )
    path = Path(get_load_path("go2trot", args.source_run, args.source_checkpoint))
    weights = torch.load(path, weights_only=True, map_location="cpu")
    env_cfg, train_cfg = copy.deepcopy(task_registry.get_cfgs(args.task))
    if args.rollout_size is not None:
        train_cfg.algorithm.rollout_size = args.rollout_size
    if args.rating_focus:
        env_cfg.commands.ranges.lin_vel_x = [0.8, 1.0, 1.2]
        env_cfg.commands.ranges.lin_vel_y = 0.0
        env_cfg.commands.ranges.yaw_vel = 0.0
        env_cfg.control.gait_freq = [2.0, 2.0]
    train_cfg, runner = setup(args, env_cfg, train_cfg)
    try:
        for name in ("actor", "critic"):
            state = model_weights(
                weights[f"{name}_state_dict"],
                getattr(source_train, name),
                getattr(runner, f"{name}_cfg"),
                runner.env,
                source_env.scaling,
            )
            getattr(runner.alg, name).load_state_dict(state)
        provenance = vars(args) | {
            "source_checkpoint_path": str(path.resolve()),
            "source_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (Path(train_cfg.log_dir) / "warm_start.json").write_text(
            json.dumps(provenance, indent=2)
        )
        print(f"Fine-tuning run: {train_cfg.log_dir}", flush=True)
        train(train_cfg, runner)
    finally:
        runner.env._backend.close()


if __name__ == "__main__":
    main()
