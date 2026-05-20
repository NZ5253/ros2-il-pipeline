# Evaluation Results

All numbers from runs on the workstation (Windows 11 + WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060). Reproduction commands in [`WORKSTATION_RUNBOOK.md`](../WORKSTATION_RUNBOOK.md).

## Setup

- **Task**: pick a red cube from a randomised pose and deliver it to a green target zone (8 cm radius around (0.40, 0.25, 0.0) m).
- **Robot**: Franka Panda (7-DOF arm + 2-finger gripper) in PyBullet, controlled with delta-EE Twist actions interpreted via PyBullet IK.
- **Grasping**: constraint-based pickup when the gripper is commanded closed and the cube is within 8 cm — the standard IL-sim convention (Robomimic, ALOHA, Diffusion Policy paper, RoboHive).
- **Dataset**: `panda_pickplace_v2` — 80 demonstrations collected through the full HTTP → ROS bridge → data logger → parquet pipeline. 78/80 expert successes (97.5 %). 41,108 frames. 24-D `observation.state`.
- **Evaluation**: 20 closed-loop rollouts per policy on freshly randomised cube poses. Success = `/task_status` reports True during the rollout. 25 s per-rollout timeout.

## Headline numbers

| Policy | Parameters | Train time (RTX 4060) | Best val loss | **Success rate** |
|---|---:|---:|---:|---:|
| **ACT** | 5.85 M | 140 min | 0.0082 | **19 / 20 = 95 %** |
| **Diffusion Policy** | 4.50 M | 70 min | 0.0012 | **18 / 20 = 90 %** |

Both policies sit at the top of published figures for state-aware pick-and-place IL on this dataset scale.

## ACT (primary)

LeRobot ACT, 500 epochs, batch 32, AdamW lr 1e-4, KL weight 10. Inputs split into STATE (14-D, joints) and ENV (10-D, EE pose + cube xyz). Chunk size 50 frames (1.67 s planning horizon at 30 Hz). Deployment uses canonical temporal ensembling (`temporal_ensemble_coeff = 0.01`, `n_action_steps = 1`, ensembler reset on each rollout) — predictions for the current timestep are exponentially-weighted-averaged across all overlapping chunks.

The single rollout failure occurred on a cube pose at the edge of the spawn distribution; the gripper approached correctly but released the cube ~1 cm outside the 8 cm target tolerance.

## Diffusion Policy (comparison)

LeRobot Diffusion Policy, 500 epochs, batch 64, AdamW lr 1e-4. Same STATE/ENV feature split as ACT. UNet sized to 4.5 M params (`down_dims = (64, 128, 256)`) so the comparison is fair — the LeRobot default ~250 M-param UNet would overfit at 80 demos. Inference runs 10 DDIM denoising steps per chunk to keep per-step latency inside the 30 Hz control budget.

The 5-point gap behind ACT is consistent with ACT's better sample efficiency in the 80-demo regime; the val losses are not directly comparable (DP's denoising MSE vs ACT's L1 + KL), so closed-loop success is the meaningful comparison. The choice between ACT and DP for production deployment is reasonable on engineering criteria (inference latency, action-distribution controllability) rather than success-rate alone.

## Inference latency

| Policy | Per-step latency (steady state) | Sustained rate |
|---|---:|---:|
| ACT | < 20 ms | 30 Hz |
| Diffusion Policy | < 30 ms | 30 Hz |

Latency is measured inside `inference_node._tick` (observation build + normalise + policy forward + denormalise + publish). Both stay inside the 33 ms cycle budget.

## Webserver integration

End-to-end smoke test on the REST + WebSocket surface:

| Endpoint | Verified |
|---|---|
| `POST /api/v1/datasets` | ✓ — returns dataset id, persists in registry |
| `POST /api/v1/datasets/{id}/record/start` | ✓ — dispatches `StartEpisode.srv` via ROS bridge |
| `POST /api/v1/datasets/{id}/record/stop`  | ✓ — dispatches `StopEpisode.srv`, returns frame count and parquet path |
| `LoadPolicy.srv` through inference node | ✓ — checkpoint loads with warm-up time reported |
| `/inference_node/start` and `/inference_node/stop` | ✓ — command stream starts and stops cleanly |
| `pybullet_robot_node/reset` between rollouts | ✓ — cube respawns, arm returns to home |
| Parquet shards, `info.json`, `episodes.jsonl`, `stats.json` | ✓ — all populated after collection |
