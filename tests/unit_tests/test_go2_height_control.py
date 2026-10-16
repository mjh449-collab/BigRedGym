"""Height rewards and training commands must distinguish both endpoint poses."""

import math

import pytest
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg
from gym.utils.helpers import class_to_dict


def test_height_reward_has_meter_tolerance_independent_of_observation_scale():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.base_height_command = torch.full((5, 1), 0.30)
    task.base_height = torch.tensor([[0.30], [0.25], [0.35], [0.20], [0.40]])
    task.scales = {"base_height": 0.3}
    expected = torch.tensor([1.0, math.exp(-1), math.exp(-1), math.exp(-4), math.exp(-4)])
    torch.testing.assert_close(task._reward_tracking_base_height(), expected)
    task.scales["base_height"] = 100.0
    torch.testing.assert_close(task._reward_tracking_base_height(), expected)


def test_command_sampler_trains_endpoints_and_standing_without_touching_other_envs():
    torch.manual_seed(19)
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.device = "cpu"
    task.num_envs = 8192
    task.command_ranges = class_to_dict(task.cfg.commands.ranges)
    task.commands = torch.ones(task.num_envs, 3)
    task.base_height_command = torch.full((task.num_envs, 1), 0.30)
    mask = torch.arange(task.num_envs) % 2 == 0
    task._resample_commands(mask)
    assert torch.equal(task.commands[~mask], torch.ones_like(task.commands[~mask]))
    assert torch.equal(
        task.base_height_command[~mask],
        torch.full_like(task.base_height_command[~mask], 0.30),
    )
    height = task.base_height_command[mask]
    lo, hi = task.cfg.commands.ranges.base_height
    assert torch.all((height >= lo) & (height <= hi))
    assert (height == lo).float().mean().item() == pytest.approx(0.25, abs=0.04)
    assert (height == hi).float().mean().item() == pytest.approx(0.25, abs=0.04)
    stopped = (task.commands[mask] == 0).all(dim=1).float().mean().item()
    assert stopped == pytest.approx(0.7, abs=0.04)
