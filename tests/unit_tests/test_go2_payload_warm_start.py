import torch

from learning.modules.actor import Actor
from scripts.finetune_go2_payload import fold_height_input


def test_fixed_height_actor_preserves_actions_for_same_physical_state():
    torch.manual_seed(43)
    source = Actor(5, 2, [16, 8], normalize_obs=False)
    target = Actor(4, 2, [16, 8], normalize_obs=False)
    scaled_height = 0.30 / 0.40
    target.load_state_dict(fold_height_input(source.state_dict(), 2, scaled_height))

    obs = torch.randn(32, 4)
    source_obs = torch.cat(
        (obs[:, :2], obs.new_full((32, 1), scaled_height), obs[:, 2:]), 1
    )
    torch.testing.assert_close(target(obs), source(source_obs))
