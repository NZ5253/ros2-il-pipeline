# Imitation Learning Pipeline — Technical Concept

**Author:** Naeem Zain Uddin
**Date:** May 2026

## 1. Goal

Extend the MyBotShop robotic webserver with three capabilities, all exposed through the existing ROS 2 + WebSocket surface: record demonstrations from the teleop stream, train a manipulation policy on those demonstrations, and deploy the trained policy to drive the robot through the same controller a human teleoperator drives. Designed to be hardware-agnostic — the same nodes run on any 6-/7-DOF arm whose driver respects standard `sensor_msgs` and `geometry_msgs` topics.

## 2. Scope

| Decision | Choice | Rationale |
|---|---|---|
| Task | Pick-and-place with random object pose | Canonical IL benchmark, transferable across the platform's arms |
| Reference robot | Franka Panda in PyBullet | Used in ACT, Robomimic, ALOHA; URDF-driven so any arm is a config swap |
| IL framework | LeRobot (HuggingFace) | ACT and Diffusion Policy built-in, parquet-based dataset format |
| Primary policy | ACT (Action Chunking Transformer) | Strong on small datasets; chunk-based deployment handles multi-phase manipulation |
| Comparison policy | Diffusion Policy | Second SOTA point, same dataset and feature split |
| Dataset format | LeRobotDataset (parquet) | Standard format, HF Hub-shareable, no custom conversion |
| Web layer | FastAPI + WebSocket | Matches the platform's existing pattern, additive (runs alongside) |

## 3. Architecture

Three new ROS 2 nodes and a FastAPI service. Full diagram in [`02_architecture.md`](02_architecture.md).

- **`data_logger_node`** — records demonstration episodes during teleop. Subscribes to `/joint_states`, `/cartesian_pose`, `/cube_pose` (task object), and `/teleop_cmd`. Writes LeRobotDataset parquet shards.
- **`inference_node`** — closed-loop deployment. Same subscriptions, plus `/inference/load_policy` and `/inference/start|stop` services. Publishes to `/cmd_robot` — the same topic the teleop controller already drives.
- **`pybullet_robot_node`** — simulated Franka Panda stand-in. Respects the same topic contracts as a real robot driver, so swapping it for the MyBotShop platform is a topic-remap, not a code change.
- **FastAPI service** — REST + WebSocket layer that the webserver UI calls. Dispatches typed service calls through a ROS 2 bridge to the nodes above. Full spec in [`03_api.md`](03_api.md).

Custom interfaces are kept to a minimum. Two custom services (`StartEpisode`, `LoadPolicy`) carry payloads that standard `std_srvs/Trigger` couldn't.

## 4. Observation and action

`observation.state` per frame: joint positions (7) + joint velocities (7) + EE pose (xyz + quat, 7) + task object xyz (3) = **24-D**. Including the object's pose follows the Robomimic and ALOHA conventions and is necessary for random-spawn manipulation.

`action` per step: 7-D delta-EE Twist (linear xyz + angular xyz + gripper). Published as `geometry_msgs/Twist`.

Full schema and field order in [`04_dataset.md`](04_dataset.md).

## 5. Policy

**ACT** is the primary deployed policy. Configuration matches the original ACT paper for chunk-based manipulation on small datasets:

- 5.85 M parameters; transformer encoder-decoder with VAE prior
- Chunk size 50 (1.67 s planning horizon at 30 Hz)
- Inputs split into STATE (joint pos + vel, 14-D) and ENV (EE pose + object xyz, 10-D)
- Deployment uses **canonical temporal ensembling** (`temporal_ensemble_coeff=0.01`, `n_action_steps=1`, ensembler reset on each rollout) — predictions for the current timestep are exponentially-weighted-averaged across all overlapping chunks
- Trained 500 epochs, AdamW, lr 1e-4, batch 32

**Diffusion Policy** is the comparison policy under the same dataset, feature split, and training budget. UNet sized to ~4.5 M params (`down_dims=(64,128,256)`) to match ACT; the LeRobot default of ~250 M would overfit at this dataset size. Inference runs 10 DDIM denoising steps per chunk.

Both policies are selectable by a single `--policy` flag in `scripts/train.py` and a single `policy_type` parameter at the inference node.

## 6. Web API

The webserver gains a small REST + WebSocket surface for orchestrating the pipeline. Full spec in [`03_api.md`](03_api.md). Endpoints are additive to the platform's existing surface — they run alongside, not on top of, the existing functionality.

```
POST   /api/v1/datasets/{id}/record/start    begin recording an episode
POST   /api/v1/datasets/{id}/record/stop     end the current episode
POST   /api/v1/training/jobs                 start a training job
GET    /api/v1/training/jobs/{id}            job status + metrics
POST   /api/v1/policies/{id}/deploy          load policy into inference node
POST   /api/v1/policies/{id}/start           begin inference
POST   /api/v1/policies/{id}/stop
```

WebSocket channels for live training progress and inference telemetry are at `ws://.../training/{job_id}/progress` and `ws://.../inference/live`.

## 7. Results

Closed-loop evaluation over 20 rollouts on freshly randomised cube poses:

| Policy | Success |
|---|---:|
| ACT | 19 / 20 = 95 % |
| Diffusion Policy | 18 / 20 = 90 % |

End-to-end command rate sustained at 30 Hz, per-step inference latency under 30 ms on RTX 4060. Demo collection through the full HTTP → ROS bridge → data logger → parquet pipeline produced 78/80 successful episodes (scripted phase-based expert). Numbers and per-experiment detail in [`05_results.md`](05_results.md).

## 8. Platform integration

The MyBotShop platform ships pre-installed on the company's robot hardware (no public install path was found). The integration model is therefore **against documented contracts, not internals**:

- The `pybullet_robot_node` stand-in plays the role the customer robot plays inside the MyBotShop platform: publishes `/joint_states`, `/cartesian_pose`, `/cube_pose`; listens on `/cmd_robot`. Any robot driver that respects these standard `sensor_msgs` and `geometry_msgs` topics is a drop-in replacement.
- The FastAPI service runs adjacent to the platform's WebSocket layer rather than replacing it. The endpoints in [`03_api.md`](03_api.md) are the surface a real deployment exposes to the webserver UI.
- The new ROS 2 nodes use standard message types plus three custom services in `il_pipeline_msgs`. They make no assumptions about the platform's internals — only its published interfaces.

## 9. Future work

Mentioned for scope; not implemented here.

- **Vision-conditioned policies** — the data logger already subscribes to `/camera/image_raw`; switching to image-conditioned ACT / DP is a config change in `train.py` + a recollection.
- **Multi-task / language conditioning** — the dataset carries task strings; CLIP-conditioned policies (OpenVLA, π₀) are a direct upgrade with no schema change.
- **PPO fine-tuning** — the trained IL policy is a natural initialisation for PPO against a sparse success reward. The hook is described in §6.3 of the original concept; left for the workstation environment.
- **Sim-to-real** — domain randomisation on physics, textures, lighting in simulation; the ROS 2 interface layer is identical on sim and hardware.
