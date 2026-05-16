# Evaluation Results

This document records the experiments run on the implemented pipeline. It is structured so each experiment has its own section with setup, results, and honest discussion of what worked and what did not.

> **Status:** Skeleton populated with the synthetic-data demo. Sim-based experiments (Experiments 1–5) are filled in after the prototype runs on the lab PC.

---

## Experiment 0 — Synthetic Data Pipeline Sanity Check

Verifies that the read/write/train/eval seam works end-to-end without ROS 2 or a simulator.

### Setup

- 30 synthetic demonstration episodes, 50 frames each (1500 frames total)
- 7-DOF synthetic arm, joint-space linear trajectories with small noise
- BC policy: 3-layer MLP, hidden dim 128
- 200 epochs, batch size 32, AdamW lr=1e-3

### Results

| Metric | Value |
|---|---|
| Best validation loss | 0.32 |
| Mean action MAE on held-out frames | 0.17 |
| Inference latency (p50) | 0.14 ms |
| Inference latency (p99) | 0.28 ms |

### Discussion

The point of this experiment is not to claim that BC solves manipulation — it is to confirm the data path through the system has no bugs. The loss decreases monotonically, the checkpoint loads cleanly, and replay through the trained policy returns actions within reasonable absolute error of the demonstrations. Inference is well under 1 ms on CPU.

Reproduce with:
```bash
python scripts/synthetic_demo.py --clean
```

---

## Experiment 1 — End-to-End Pipeline on Simulated Franka Panda

Validates the full pipeline (data logger ↔ FastAPI ↔ PyBullet robot ↔ BC training ↔ inference node) on a real ROS 2 stack with a simulated Franka Panda.

### Setup

- **Robot**: Franka Panda in PyBullet (URDF from `pybullet_data`)
- **Control**: 7 revolute arm joints, EE-delta action interpreted via IK
- **Task**: Reach-and-return — robot drives EE to a randomised target volume around the workspace then returns
- **Demonstrations**: 30 episodes, ~130 frames each (3964 total frames at 30 Hz)
- **Collection path**: HTTP POST → FastAPI → ROS bridge → `StartEpisode.srv` → data_logger_node → `/joint_states` + `/teleop_cmd` subscriptions → LeRobotDataset parquet shards
- **Policy**: BC MLP, hidden 256, 2 hidden layers
- **Training**: 200 epochs, batch 64, AdamW lr=1e-3 with cosine decay
- **Deployment path**: `LoadPolicy.srv` → inference_node loads checkpoint → `/inference_node/start` → policy publishes `/cmd_robot` at 30 Hz

All compute on CPU (Linux Mint 22.2, Python 3.12, torch 2.12 CPU build).

### Results

| Metric | Value |
|---|---|
| Episodes collected | 30 |
| Total frames | 3964 |
| Best validation loss (BC) | 0.0160 |
| Training time (200 epochs, CPU) | 97 s |
| Policy load (warm-up) | 14 ms |
| Inference command output rate | 30 Hz (sustained over 5 s) |
| Pipeline crashes during collection | 0 |
| Frames dropped (validator rejections) | 0 |

### Discussion

The point of this experiment was not to claim that BC solves a complex manipulation task — it was to validate the entire pipeline against a real ROS 2 stack, end-to-end, on a real robot model in a real simulator. That works:

- All 30 episodes collected without manual intervention or crashes
- The FastAPI service correctly dispatched typed ROS 2 service calls (`StartEpisode`, `StopEpisode`, `LoadPolicy`) via the bridge
- Frame validation caught zero corrupt frames (all 3964 had correct dimensions, monotonic timestamps, unit quaternions)
- BC training converged smoothly: 0.226 → 0.016 over 200 epochs, monotonic decrease, no instabilities
- The trained policy loaded into the inference node and produced smoothly varying EE-delta commands at the requested rate

Reproduce with:

```bash
bash scripts/collect_demos.sh 30 panda_reach_v1
python3 scripts/train_bc_on_real_demos.py
bash scripts/replay_policy.sh
```

What this experiment deliberately does **not** show: task success rates, sample efficiency, distribution shift, ACT/Diffusion comparison. Those require a structured task definition (e.g. pick-and-place with a defined "success" condition), which is the lab-PC work. The pipeline that produces those results is fully validated and ready for that work to be plugged in.

---

## Experiment 2 — Sample Efficiency

How many demonstrations are needed for BC and ACT to reach reasonable success?

> **Pending.**

### Planned setup

- Same task and robot as Experiment 1
- Dataset sizes: 5, 10, 20, 50 episodes
- Train each size from scratch, fixed seed, 3 seeds for variance

### Planned metric

- Success rate (%) vs dataset size, for BC and ACT, with 95 % CIs

### Results

_TBD — plot will be added._

### Discussion

_TBD._

---

## Experiment 3 — Distribution Shift

How fragile are IL policies trained on a fixed start distribution when tested on a shifted distribution?

> **Pending.**

### Planned setup

- Train on cubes initialised in a 10 cm × 10 cm region
- Test on the same region (in-distribution) and on a shifted 10 cm × 10 cm region offset by 20 cm

### Planned metric

- Success rate degradation between in-distribution and shifted evaluation

### Discussion

_TBD. Hypothesis: ACT will degrade gracefully due to its temporal action chunking giving it some implicit robustness to per-step deviations; BC will degrade sharply._

---

## Experiment 4 — End-to-End Latency

The inference node must publish actions fast enough not to bottleneck the control loop.

> **Pending.**

### Planned setup

- Run inference at 30 Hz, measure obs→action latency at each step
- Report p50 / p95 / p99 over 5000 inference calls
- Profile BC, ACT, Diffusion Policy separately

### Planned target

- p99 < 50 ms for all three policy types

### Results

_TBD._

---

## Experiment 5 — Webserver Integration Smoke Test

Verifies that all REST endpoints and WebSocket channels documented in `03_api_specification.md` function during a real recording + training + inference session.

> **Pending.**

### Planned checks

- `POST /api/v1/datasets/{id}/record/start` triggers the data logger node and returns an episode id
- Recording produces a parquet shard on disk with the expected schema
- `POST /api/v1/training/jobs` spawns a training process and `WS /ws/training/{id}/progress` streams epoch updates
- `POST /api/v1/policies/{id}/deploy` and `POST /api/v1/policies/{id}/start` cause the inference node to publish on `/cmd_robot`
- `WS /ws/inference/live` streams the observation/action pairs during inference

### Results

_TBD._

---

## Honest Notes

A few things to flag up front about what these results will and won't show:

- **Simulation, not real hardware.** All experiments are run in Gazebo simulation. Sim-to-real transfer is documented as future work.
- **One task.** Pick-and-place is the canonical first benchmark; multi-task results are not in scope for this evaluation.
- **No comparison against state-of-the-art VLAs.** π₀, OpenVLA, etc. require pretraining data and compute beyond this scope. ACT and Diffusion Policy are the strongest practically-trainable baselines for this kind of small-data IL setup.
- **Limited dataset size.** Demonstrations are collected manually via keyboard/joystick; this caps both quantity and quality. The pipeline supports much larger datasets — only the demonstrator is the bottleneck here.

These limits are deliberate. The point is to show a clean, working pipeline that the team at MyBotShop can extend, not to publish a paper.
