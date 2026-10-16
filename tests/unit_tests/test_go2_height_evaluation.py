"""A height-only policy must fail the movement acceptance check."""

import torch

from scripts.evaluate_go2_height import summarize


def test_standing_still_cannot_pass_a_forward_command():
    report = summarize(
        torch.full((50,), 0.3),
        torch.zeros(50, 3),
        0.3,
        torch.tensor([0.5, 0.0, 0.0]),
        False,
    )
    assert not report["passed"]
    assert report["linear_rmse_m_s"] == 0.5


def test_tracking_and_survival_are_both_required():
    command = torch.tensor([0.5, 0.0, 0.0])
    height = torch.full((50,), 0.3)
    velocity = command.repeat(50, 1)
    assert summarize(height, velocity, 0.3, command, False)["passed"]
    assert not summarize(height, velocity, 0.3, command, True)["passed"]
    assert not summarize(height, velocity, 0.4, command, False)["passed"]
