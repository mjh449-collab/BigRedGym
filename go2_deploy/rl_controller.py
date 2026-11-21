from gym.utils.task_registry import task_registry
from gym.utils.helpers import set_seed
import gym.envs  # noqa: F401
from go2_deploy.deploy_config import DeployConfig
from gym.envs.go2.go2trot_config import Go2TrotRunnerCfg
from learning.modules.actor import Actor

import hashlib
import io
from pathlib import Path
import random
import torch


# Returns Actor constructed from logs/go2/run_name (default: most recent)
def setup_actor(run_name=None):
    deploy_cfg = DeployConfig()
    env_cfg, train_cfg = task_registry.get_cfgs(name=deploy_cfg.task_name)

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 99999
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 99999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "init_state") and hasattr(env_cfg.init_state, "reset_mode"):
        env_cfg.init_state.reset_mode = "reset_to_range"

    env_cfg.seed = random.randint(0, 10000)
    train_cfg.seed = random.randint(0, 10000)

    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = True
    if run_name is not None:
        train_cfg.runner.load_run = run_name
    train_cfg.runner.checkpoint = -1
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        name=deploy_cfg.task_name, env_cfg=env_cfg, device="cpu", headless=True
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return runner.alg.actor


class RLController:
    def __init__(self, checkpoint_path=None):
        self.cfg = DeployConfig()
        self.checkpoint_path = None
        self.checkpoint_sha256 = None
        if checkpoint_path is None:
            self.actor = setup_actor()
        else:
            self._load_selected_actor(Path(checkpoint_path).resolve())

    def _load_selected_actor(self, path):
        config = Go2TrotRunnerCfg.actor
        if config.obs != self.cfg.obs_vector or config.actions != ["dof_pos_target"]:
            raise ValueError(
                "Training and deployment observation or action layouts differ"
            )
        num_obs = sum(self.cfg.obs_sizes[name] for name in self.cfg.obs_vector)
        num_actions = self.cfg.default_dof_pos.numel()
        data = path.read_bytes()
        checkpoint = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        actor = Actor(
            num_obs,
            num_actions,
            config.hidden_dims,
            activation=config.activation,
            normalize_obs=config.normalize_obs,
        )
        weights = checkpoint["actor_state_dict"]
        trained_obs = weights["NN.0.weight"].shape[1]
        if trained_obs != num_obs:
            raise ValueError(
                f"Checkpoint expects {trained_obs} inputs; this deploy config sends "
                f"{num_obs}. Use a checkpoint trained or migrated for this layout."
            )
        actor.load_state_dict(weights)
        self.actor = actor.eval()
        if not torch.isfinite(self.act(torch.zeros(num_obs))).all():
            raise ValueError(
                "Checkpoint produced nonfinite actions in offline preflight"
            )
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = hashlib.sha256(data).hexdigest()

    # Returns torch.tensor(12): offsets from the static default posture in radians.
    @torch.inference_mode()
    def act(self, obs_vector):
        scale = torch.tensor(getattr(self.cfg.DeployScaling, "dof_pos_target", 1.0))
        return self.actor.act_inference(obs_vector) * scale
