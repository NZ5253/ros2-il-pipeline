# Imitation Learning Pipeline for Manipulation
### Integration Concept for the MyBotShop ROS 2 Robotic Webserver Platform

**Author:** Naeem Zain Uddin
**Date:** May 2026
**Status:** Technical concept and architecture proposal

---

## 1. Problem Statement

The MyBotShop robotic webserver platform provides a hardware-agnostic interface for monitoring and controlling ROS 2 robots, with existing functionality for teleoperation, waypoint navigation, speech-to-action workflows, monitoring, diagnostics, and simulation visualisation. The platform is positioned as a unified control surface across humanoids, mobile manipulators, mobile platforms, and standalone arms.

The proposed evaluation task is to design and prototype a pipeline that adds **imitation learning for manipulation** to this platform. Concretely: a user should be able to (1) collect demonstration data through the platform's teleoperation interface, (2) have those demonstrations persist as a structured dataset accessible from the webserver, and (3) train and deploy a policy that reproduces the demonstrated behaviour on the same robot.

A clean extension to the existing platform — not a parallel system — is the design goal. The pipeline should respect three properties that the underlying platform already exhibits: **hardware-agnosticism**, **modular ROS 2 integration**, and **WebSocket-based real-time observability**.

---

## 2. Scope Decisions

The CEO's brief explicitly leaves scope open. The decisions made for this proposal:

| Decision | Choice | Rationale |
|---|---|---|
| Use case | Pick-and-place with a 6/7-DOF arm | Standard manipulation IL benchmark; works for any of the platform's supported arms |
| Reference robot | Franka Panda or UR5e (in simulation) | Widely supported in ROS 2; URDF and Gazebo plugins are mature; transferable to real platforms |
| Simulation | Gazebo (Harmonic if ROS 2 Jazzy, Fortress if Humble) | ROS 2 native, matches existing platform support |
| Demonstration input | Keyboard / joy teleop initially, VR teleop noted as future work | Already supported by the platform |
| IL framework | **LeRobot** (HuggingFace) | Modern, supports ACT and Diffusion Policy out of the box, parquet-based dataset format with built-in visualisation, growing community |
| Primary policy | **ACT (Action Chunking Transformer)** | Strong sample efficiency, well-suited to short manipulation trajectories from small datasets |
| Stretch policy | **Diffusion Policy** | Comparison baseline if time allows |
| Optional RL | **PPO via Stable-Baselines3**, initialised from the IL policy | Closes the loop with my thesis background on PPO + PyTorch |
| Web integration | **FastAPI** with WebSocket bridging into the platform's ROS 2 layer | Matches existing platform pattern of WebSocket-based real-time exchange |
| Dataset format | **LeRobotDataset** (parquet shards) | Compatible with `datasets` and HuggingFace Hub for shareable datasets |

The pipeline is designed to be deployable on any of the platform's supported robots without code changes — only configuration (URDF, joint names, observation space).

---

## 3. System Architecture

The proposed system adds three logical components to the existing platform, all communicating over ROS 2 interfaces:

1. **Data Logger** — records demonstration episodes during teleoperation
2. **Training Service** — trains a policy from logged datasets (off-line)
3. **Inference Node** — executes the trained policy by publishing action commands

The webserver exposes a small REST + WebSocket API for managing datasets, training jobs, and policy deployment. All three components are platform-agnostic and depend only on ROS 2 message contracts.

See [`02_architecture_diagrams.md`](02_architecture_diagrams.md) for the full system diagram, ROS 2 node graph, and data flow.

---

## 4. ROS 2 Node Structure

The pipeline introduces the following nodes:

### 4.1 `data_logger_node`

Subscribes to:
- `/joint_states` — `sensor_msgs/JointState`
- `/cartesian_pose` — `geometry_msgs/PoseStamped` (end-effector pose, if available)
- `/teleop_cmd` — `geometry_msgs/Twist` or `sensor_msgs/JointJog` (the teleop action stream from the existing platform)
- `/camera/image_raw` — `sensor_msgs/Image` (optional, if a wrist or scene camera is configured)

Exposes:
- `/data_logger/start_episode` — `std_srvs/Trigger`
- `/data_logger/stop_episode` — service taking a status payload (success / discard)
- `/data_logger/save_dataset` — service finalising the current dataset to disk

Output: appends frames to the active LeRobot-format dataset on disk, with episode boundaries and metadata.

### 4.2 `training_service_node` (off-line, ROS-aware but not real-time)

Runs as a long-lived process on the lab PC GPU. Exposes ROS 2 actions for training job lifecycle:
- `/training/start` — `Training.action` with parameters (dataset path, policy type, epochs, etc.)
- `/training/status` — feedback on epoch progress and validation loss
- `/training/cancel` — preempt the job

This node delegates to the LeRobot trainer; ROS 2 is used only as the control plane.

### 4.3 `inference_node`

Subscribes to:
- `/joint_states`
- `/cartesian_pose` (optional)
- `/camera/image_raw` (optional)

Publishes to:
- `/cmd_robot` — the same action command topic the controller already accepts (so the policy is a drop-in replacement for the teleop stream)

Service interface:
- `/inference/load_policy` — load a named policy by checkpoint path
- `/inference/start` — begin inference at a configured rate (typically 10–30 Hz for ACT)
- `/inference/stop`

### 4.4 Message and Service Contracts

Custom interfaces are kept to a minimum; standard ROS 2 messages cover most of the surface. Two custom service definitions are introduced:

```
# il_pipeline_msgs/srv/StartEpisode.srv
string episode_name
string task_description
---
bool success
string episode_id
string message
```

```
# il_pipeline_msgs/srv/LoadPolicy.srv
string checkpoint_path
string policy_type   # "act" | "diffusion" | "bc"
float64 inference_rate_hz
---
bool success
string message
```

See [`03_api_specification.md`](03_api_specification.md) for the full web-layer API.

---

## 5. Dataset Format

The dataset uses **LeRobotDataset** as its on-disk schema. This decision is load-bearing for several reasons: it is parquet-based (efficient and shardable), has a stable schema, ships with visualisation tools, integrates with HuggingFace Hub for dataset sharing, and is the format used by an increasing share of modern open-source manipulation IL work.

The full schema specification, including encoding choices for images, joint conventions, and metadata fields, is in [`04_dataset_schema.md`](04_dataset_schema.md).

A single episode contains a sequence of frames, each with:
- `observation.state` — joint positions + velocities + end-effector pose
- `observation.image.{cam_name}` — RGB frame (encoded as `Image` features with optional video compression)
- `action` — the commanded action at this step (delta-joint or delta-end-effector depending on configuration)
- `timestamp`, `frame_index`, `episode_index` — provenance metadata
- `task` — natural-language task description (enables future language-conditioned policies)

---

## 6. Learning Pipeline

### 6.1 Baseline: Behaviour Cloning (BC)

A small MLP that maps observation to action, trained with L1 or MSE loss. This serves as a sanity check: it should converge on tens of demonstrations and provides a reference point for evaluating more complex policies. If BC fails on a task, the pipeline (not the policy class) is the problem.

### 6.2 Primary: ACT (Action Chunking Transformer)

The chosen primary policy. ACT predicts a chunk of `k` future actions (typically `k=10–100`) from the current observation, conditioned through a transformer encoder over a short observation history. The chunking strategy is what makes it sample-efficient: the policy learns temporally extended behaviours from individual demonstrations rather than per-step decisions.

Training:
- Optimiser: AdamW, learning rate 1e-4 with cosine decay
- Batch size: 32–64
- Epochs: 1000–5000 depending on dataset size
- Validation: held-out episodes, success-rate evaluation in simulation

### 6.3 Stretch: Diffusion Policy

If time allows, train a Diffusion Policy on the same dataset and compare success rates. Diffusion has been shown to outperform ACT on more complex tasks at the cost of more compute and slower inference.

### 6.4 Optional: PPO Fine-Tuning

The IL policy can be wrapped as a Gym-style environment policy and fine-tuned with PPO (Stable-Baselines3) against a reward function in simulation. The reward signal can be sparse (task success) or shaped (distance-to-goal + grasp-stability). This is where my thesis work on PPO and reward shaping carries over directly.

The integration point is a `policy_bridge` module that exposes:
- `predict(observation) -> action` — for IL inference
- `act(observation, deterministic) -> action, log_prob, value` — for RL fine-tuning

---

## 7. Web-Layer Integration

The webserver gains a small surface for orchestrating the pipeline. Endpoints are designed to be additive to the platform's existing WebSocket interface — they live alongside, not on top of, the existing functionality.

**REST endpoints:**

```
GET    /api/v1/datasets               list available datasets
POST   /api/v1/datasets/record/start  begin recording an episode
POST   /api/v1/datasets/record/stop   end the current episode
GET    /api/v1/datasets/{id}          dataset metadata + episode count
DELETE /api/v1/datasets/{id}

POST   /api/v1/training/jobs          start a training job
GET    /api/v1/training/jobs/{id}     job status + metrics
DELETE /api/v1/training/jobs/{id}     cancel job

GET    /api/v1/policies               list trained policies
POST   /api/v1/policies/{id}/deploy   load policy into inference node
POST   /api/v1/policies/{id}/start    begin inference
POST   /api/v1/policies/{id}/stop
```

**WebSocket channels:**

```
ws://.../training/{job_id}/progress   epoch / loss / validation metrics
ws://.../inference/live               action commands + observation for replay
```

The full OpenAPI specification is in [`03_api_specification.md`](03_api_specification.md).

---

## 8. Evaluation Plan

To make claims about the pipeline being usable rather than just structurally sound, the following evaluation is performed on the prototype:

1. **Pipeline correctness** — collect 20 demonstrations of a single pick-and-place task in simulation; train BC; verify the policy executes the task in simulation with success rate measurable.
2. **Sample efficiency** — repeat with 5, 10, 20, 50 demonstrations; report success rate vs. dataset size for BC and ACT.
3. **Distribution shift** — train on demonstrations of one object pose distribution; evaluate on a shifted distribution; document degradation.
4. **End-to-end latency** — measure observation-to-action latency through the inference node; target under 50 ms for control rate compatibility.
5. **Webserver integration** — verify all REST endpoints function and WebSocket streams emit data during training and inference.

Results are documented honestly, including negative findings.

---

## 9. What This Concept Deliberately Does Not Do

A few choices were considered and explicitly rejected:

- **Bundling everything inside the platform's existing codebase.** Keeping the pipeline as a separate ROS 2 package that depends only on standard interfaces makes it testable in isolation and trivially reusable for other platforms.
- **A custom dataset format.** LeRobotDataset is already standard. Inventing a new format would create maintenance burden and isolation from the broader community.
- **Vision-language-action models (VLAs) like π₀ or OpenVLA.** These are powerful but require large pretraining datasets and high VRAM, and would not be evaluable within this scope. They are appropriate future work once the basic pipeline is in place.
- **Real-robot training.** The CEO mentioned the platform's visualisation environment for simulation; staying in simulation for this evaluation keeps the scope tractable while still demonstrating the full pipeline.

---

## 10. Future Work

Within the same architectural skeleton, several extensions are straightforward:

- **VR teleoperation** for richer demonstrations (Oculus / Vision Pro bridges to ROS 2 exist)
- **Language conditioning** — the dataset already carries task strings; CLIP-conditioned policies are a small extension
- **VLA-class policies** (OpenVLA, π₀) once data and compute are available
- **Multi-task policies** trained on a mixture of demonstration datasets
- **Sim-to-real transfer** with domain randomisation
- **Active learning** — the policy queries the user for demonstrations on failure cases through the webserver UI

---

## 11. Risks and Open Questions

| Item | Status | Mitigation |
|---|---|---|
| Platform's exact ROS 2 distribution | Open — docs list multiple supported versions | Pipeline targets ROS 2 interface contracts, not implementation. Will confirm by inspection on lab PC. |
| Platform's teleop topic and message types | Open — assumed `Twist` / `JointJog` | Will confirm and adapt the logger subscriber accordingly. Trivial to change. |
| Specific manipulator the platform expects | Open — sent clarifying email to CEO | Defaults to Franka Panda in simulation; URDF-driven so substitution is config-only. |
| Webserver plugin extension mechanism | Plugin-style mentioned in docs, not specified | The proposed FastAPI service runs adjacent to the platform; integration depth can grow once internals are understood. |
| Outdated documentation | Acknowledged by the CEO | Will document gaps encountered and propose doc updates as a side benefit. |

---

## 12. Summary

The proposed pipeline adds imitation learning for manipulation to the MyBotShop webserver as three new ROS 2 nodes (data logger, training service, inference) plus a small REST + WebSocket API layer. It uses LeRobotDataset as the on-disk format, ACT as the primary policy with BC as a sanity-check baseline and Diffusion Policy as a stretch comparison, and reserves a clean integration point for PPO fine-tuning that connects directly to my thesis background. The design is hardware-agnostic, distribution-agnostic, and modular against the existing platform.

The prototype implementation, demonstration data, trained policies, evaluation results, and a short demo video accompany this document.
