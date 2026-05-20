# Results

Workstation: Windows 11 + WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060. Reproduce with [`RUNBOOK.md`](../RUNBOOK.md).

## Task

Pick a red cube from a randomised pose, place it on a green target zone (8 cm radius around (0.40, 0.25, 0.0) m). Franka Panda with a 2-finger gripper, delta-EE Twist control at 30 Hz. Grasping is constraint-based (cube fixes to the EE when the gripper closes within 8 cm), which is what every IL benchmark on PyBullet does.

## Dataset

`panda_pickplace_v2`: 80 demos collected end-to-end through the FastAPI → ROS bridge → data_logger → parquet pipeline. The scripted phase-based expert succeeded on 78/80. 41,108 frames. 24-D `observation.state`.

## Closed-loop evaluation

20 rollouts per policy on freshly randomised cube poses. Each rollout has a 25 s deadline; success means `/task_status` reports True before the timer runs out.

| Policy | Params | Train time | Best val loss | Success |
|---|---:|---:|---:|---:|
| ACT | 5.85M | 140 min | 0.0082 | 19/20 (95%) |
| Diffusion Policy | 4.50M | 70 min | 0.0012 | 18/20 (90%) |

Both are at the top end of published figures for state-aware pick-and-place IL at this dataset scale.

The single ACT failure was on a cube pose at the edge of the spawn distribution; the arm approached and grasped correctly but released ~1 cm outside the 8 cm tolerance. The two DP failures were similar (one near-miss release, one off-axis grasp). Neither were policy collapses, just edge cases.

Val losses aren't directly comparable across policies (ACT's loss is L1 + KL against the VAE prior, DP's is denoising MSE), so the closed-loop number is the real comparison. The 5-point ACT lead is consistent with ACT's better sample efficiency at 80 demos. Picking between ACT and DP for production is more a question of inference latency and action-distribution controllability than success rate.

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
