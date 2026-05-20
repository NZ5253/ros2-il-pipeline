# Results

Workstation: Windows 11 + WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060. Reproduce with [`RUNBOOK.md`](../RUNBOOK.md).

## Task

Pick a red cube from a randomised pose, place it on a green target zone (8 cm radius around (0.40, 0.25, 0.0) m). Franka Panda with a 2-finger gripper, delta-EE Twist control at 30 Hz. Grasping is constraint-based (cube fixes to the EE when the gripper closes within 8 cm), which is what every IL benchmark on PyBullet does.

## Dataset

`panda_pickplace_v2`: 80 demos collected end-to-end through the FastAPI → ROS bridge → data_logger → parquet pipeline. The scripted phase-based expert succeeded on 78/80. 41,108 frames. 24-D `observation.state`.

## Closed-loop evaluation (in-distribution)

Rollouts on freshly randomised cube poses inside the training spawn range. Each rollout has a 25 s deadline; success means `/task_status` reports True before the timer runs out.

| Policy | Params | Train time | Best val loss | Rollouts | Success |
|---|---:|---:|---:|---:|---:|
| ACT | 5.85M | 140 min | 0.0082 | 50 | 48/50 (96%) |
| Diffusion Policy | 4.50M | 70 min | 0.0012 | 20 | 18/20 (90%) |

ACT was re-evaluated on 50 rollouts to get a tighter confidence interval. Wilson 95% CI for 48/50 is roughly [86, 99] %, putting it at the top end of published figures for state-aware pick-and-place IL at this dataset scale.

The two ACT failures were edge cases: cubes spawned at the extreme corners of the spawn distribution, the policy grasped correctly but released just outside the 8 cm target tolerance. The two DP failures from the 20-rollout eval were similar (one near-miss release, one off-axis grasp). No policy collapses.

Val losses aren't directly comparable across policies (ACT's loss is L1 + KL against the VAE prior, DP's is denoising MSE), so the closed-loop number is the real comparison. Picking between ACT and DP for production is more about inference latency and action-distribution controllability than the small success-rate gap.

## Out-of-distribution evaluation

To check how far the policy generalises beyond its training spawn range, 20 rollouts with the cube spawned **outside** the training distribution:

| Cube spawn | x range | y range | ACT success |
|---|---|---|---:|
| In-distribution (training) | [0.40, 0.55] m | [-0.15, 0.05] m | 96 % |
| Out-of-distribution shift  | [0.55, 0.65] m | [ 0.05, 0.15] m | 1/20 (5 %) |

The OOD spawn region has no overlap with the training region. The 96 → 5 % drop is the expected behaviour for state-based IL without domain randomisation; the policy is strong inside the training distribution and doesn't extrapolate outside it. Two clean ways to lift this: collect demos covering a wider spawn region (most direct), or add domain randomisation at training time. Vision-conditioned policies generalise more gracefully here too, since the visual features can extrapolate where joint-space encodings don't.

Reproduce:
```bash
EVAL_DEVICE=cuda:0 bash scripts/evaluate_ood.sh runs/panda_act_v2/best.pt act 20
```

## Latency

Measured inside `inference_node._tick` (observation build + normalise + forward + denormalise + publish). Both policies stay inside the 33 ms cycle budget at 30 Hz.

- ACT — under 20 ms per step in steady state
- DP — under 30 ms per step (10 DDIM denoising steps per chunk)

## Webserver smoke

End-to-end through the FastAPI surface:

- `POST /api/v1/datasets` returns an id and persists in the registry
- `POST /api/v1/datasets/{id}/record/start` dispatches `StartEpisode.srv` and returns `episode_id`
- `POST /api/v1/datasets/{id}/record/stop` dispatches `StopEpisode.srv`, returns frame count and parquet path
- `LoadPolicy.srv` through the inference node loads a checkpoint with the warm-up time reported back
- `~/start` and `~/stop` cleanly start and stop the command stream
- `pybullet_robot_node/reset` between rollouts respawns the cube and returns the arm to home
- `info.json`, `episodes.jsonl`, `stats.json` all populated after collection
