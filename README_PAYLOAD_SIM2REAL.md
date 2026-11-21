# Go2 payload transfer candidate

Compare every candidate at **1.0 m/s, 0.30 m target body height, and a 2 Hz straight trot**. Reproduce the Assignment 6 baseline from its saved run first. The report measured about 0.66 m/s even at 0 kg, so check unloaded tracking before expanding training beyond the existing 0–5 kg range. A candidate must bring unloaded forward-speed RMSE to 0.20 m/s or less while preserving the trot and meeting the height and torque bounds. Then train for the wider load range and rerun the same assessment. The minimum hardware goal is 5 kg and the rated goal is 7.5 kg.

The candidate keeps height as a fixed training target and evaluation metric. The actor has no height-command input. Forward errors are no longer divided by `1 + |forward command|` in the reward. At 1.0 m/s the old normalization divided squared forward error by four. This change makes undertracking more costly during training. Trained policy weights are required to measure its effect.

Training and simulation use the custom deployment controller's configured nominal defaults of 100 Hz, Kp 30, Kd 2, and an action EMA with 0.1 weight on the newest action. The previous-action observation uses the filtered offset. Training still randomizes gains around the nominal values. Before claiming hardware parity, compare these settings with the hardware control CSV. Use its timestamps to measure the observed loop rate and its `kp_mult` and `kd_mult` columns to confirm the effective gains. Firmware dynamics and torque limits still need hardware validation.

## Evaluate a saved run

Run from the repo root. Put the baseline run folder under `logs/go2trot/`, including its `model_*.pt` and `files/` snapshot. Prefer an explicit `--checkpoint <iteration>` if the folder has multiple models. Use `--original_cfg` for the baseline and again for a saved candidate whose settings differ from the current checkout.

```bash
uv run python -m scripts.evaluate_payload_readiness --load_run BASELINE_RUN \
  --original_cfg --masses 0 --output logs/baseline-unloaded.json

uv run python -m scripts.evaluate_payload_readiness --load_run CANDIDATE_RUN \
  --original_cfg --masses 0 --output logs/candidate-unloaded.json
```

When the candidate passes unloaded, omit `--masses 0` for the full nominal sweep of 0, 2.5, 5, 7.5, and 10 kg. The evaluator runs 20 trials of 20 seconds per mass and excludes the first 2 seconds from tracking measurements. Falls count throughout. The report records each criterion, the checkpoint hash, the nominal controller settings, and whether the saved config matches the current deployment defaults. Compare branches using the same seed and trial settings. Use the 90th percentile of the per-stride maximum foot-share deviation for the trot criterion, as in the existing Go2Trot evaluator. Torque fractions are sampled at the policy rate, which equals the physics rate for this candidate.

For readiness, use `--suite randomized --masses 7.5` for 100 trials at a proposed 7.5 kg test load. Use `--suite stress --masses 10` for the extra 2.5 kg test. For an offset stress test at 7.5 kg, use `--suite stress --masses 7.5 --com-offset-m 0.07 0 0` for a 2 cm margin beyond the currently configured +5 cm fore/aft COM bound. Choose the axis and sign based on the measured mount and repeat as needed. These suites report their separate survival thresholds. The nominal result's `jitter_passed` field checks the 10 Hz and 0.02 rad RMS readiness bound.
