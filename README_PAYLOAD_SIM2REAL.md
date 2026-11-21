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

## Warm-start from Assignment 6

The saved November 10 run expects a height input. The warm-start script folds the fixed 0.30 m input into the actor and critic's first-layer biases. For the same physical state, this preserves the original network outputs at 0.30 m. Training uses fresh optimizers and the current controller settings. Checkpoint 0 is the migrated policy before any fine-tuning and lets you measure the effect of changing the controller settings separately.

```bash
uv run python -m scripts.finetune_go2_payload --task go2trot \
  --source-run Nov10_01-32-37_ --source-checkpoint 550 --device cuda:0 \
  --run_name sim2real-finetune --seed 123 --headless --disable_wandb
```

To reproduce a short CPU pilot, use `--device cpu --num_envs 32 --rollout-size 8192 --batch_size 2048 --max_iterations 12 --save_interval 3`. Keep the run's `warm_start.json` alongside the checkpoints. Evaluate a selected checkpoint on a second seed before using it to decide on hardware testing.

`--rating-focus` restricts fine-tuning to straight commands around 1.0 m/s and a 2 Hz gait. This is a separate experiment for the payload rating. Check slow starts and ordinary joystick commands separately before considering that model for deployment.

## Isolate controller changes in simulation

For the November 10 checkpoint, the short unloaded diagnostic uses `--masses 0 --replicas 8 --seconds 8 --original_cfg`. Start with the saved settings, then use these overrides to isolate the changes. Every row uses the same checkpoint with a 0.30 m height target.

| Overrides | Mean forward speed (m/s) | Forward RMSE (m/s) |
| --- | ---: | ---: |
| Saved settings | 0.657 | 0.354 |
| `--policy-hz 100` | 0.663 | 0.345 |
| `--policy-hz 100 --gains 30 2` | −0.158 | 1.165 |
| `--policy-hz 100 --ema-alpha 0.1` | 0.168 | 0.835 |
| `--policy-hz 100 --gains 30 2 --ema-alpha 0.1` | −0.074 | 1.081 |

The gain change had the largest effect on these saved weights. The EMA also reduced forward speed. Raising the rate alone had little effect on speed. Use this diagnostic to compare settings from a real control log before committing to a new hardware configuration.

## November 10 checkpoint results

The original checkpoint 550 reproduced the reported unloaded speed at 0.662 m/s (0.350 m/s RMSE). In the full payload sweep, 5 kg exceeded the 1.5 cm sag limit (2.24 cm) and 7.5 kg averaged 0.601 m/s. The 12-iteration broad fine-tune and its 24-iteration rating-focused continuation under the configured deployment settings failed the speed gate. The best broad survival result at 5 kg was 20/20, but that checkpoint averaged about zero forward speed.

Two simulation-only profiles at 100 Hz with the saved gains (Kp 20, Kd 0.5) passed the unloaded speed gate on seed 456 with 20 trials of 20 seconds. With `--ema-alpha 0.35`, unloaded RMSE was 0.151 m/s and 5 kg survival was 7/20. With `--ema-alpha 0.5`, unloaded RMSE was 0.170 m/s and 5 kg survival was 20/20, but 5 kg RMSE was 0.291 m/s and sag was 3.20 cm. Both profiles missed the unloaded jitter limit. These results support further simulation tuning at 0 and 5 kg before a hardware test.
