import torch

from scripts.evaluate_payload_readiness import rate_case, summarize_payload
from tests.unit_tests.test_go2trot_performance import ideal_trace


def payload_trace(speed):
    trace = ideal_trace()
    trace["velocity"][..., 0] = speed
    trace["height"] = torch.full_like(trace["alive"], 0.30, dtype=torch.float)
    trace["torque_fraction"] = torch.full_like(trace["residual"], 0.30)
    return trace


def test_equal_undertracking_at_zero_and_rated_load_fails_rating():
    case = summarize_payload(payload_trace(0.66), 0.01, 200)
    rated = rate_case(case, case)
    assert rated["checks"]["speed_relative_to_unloaded"]
    assert rated["failures"] == ["forward_rmse"]


def test_sag_and_brief_saturation_fail_even_with_passing_height_and_p95():
    unloaded = summarize_payload(payload_trace(1.0), 0.01, 200)
    trace = payload_trace(1.0)
    trace["height"].fill_(0.28)
    trace["torque_fraction"][200:212, :, 0] = 1
    rated = rate_case(summarize_payload(trace, 0.01, 200), unloaded)
    assert rated["checks"]["height_error"]
    assert rated["checks"]["torque_p95"]
    assert rated["failures"] == ["sag_relative_to_unloaded", "torque_saturation"]
