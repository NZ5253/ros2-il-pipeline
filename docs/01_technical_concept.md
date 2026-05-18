# Imitation Learning Pipeline for Manipulation
### Integration with the MyBotShop ROS 2 Robotic Webserver Platform

**Author:** Naeem Zain Uddin  
**Date:** May 2026

---

## 1. Problem Statement

The brief asks for an imitation learning pipeline that integrates with the MyBotShop webserver platform as a natural extension, not a separate tool. Three things need to work end-to-end: collecting demonstrations through the existing teleop interface, storing them in a structured dataset reachable from the webserver, and deploying a trained policy that sends commands through the same robot controller the human teleoperator uses.

The design constraint that shapes every decision below is hardware-agnosticism: the platform runs on humanoids, mobile arms, and standalone manipulators without code changes. The IL pipeline has to respect the same constraint.

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

The pipeline requires only configuration changes (URDF, joint names, observation dimension) to switch between robots.

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

The dataset uses **LeRobotDataset** (parquet shards, HuggingFace `datasets`-compatible). This avoids inventing a custom format and gives ACT, Diffusion Policy, and OpenVLA direct access to the data without conversion. Visualisation and HF Hub upload are included.

Schema per frame: `observation.state` (joint positions + velocities + EE pose), `action` (delta-EE Twist), `timestamp`, `episode_index`, `frame_index`, `task` string. Camera frames are optional. Full schema in [`04_dataset_schema.md`](04_dataset_schema.md).

---

## 6. Learning Pipeline

### 6.1 Baseline: Behaviour Cloning (BC)

A 3-layer MLP mapping `observation.state` → `action`, trained with L1 loss. Fast to train (minutes on GPU), simple to debug, and useful as a lower bound. If BC fails to converge, something is wrong with the data, not the policy class.

BC is a known weak baseline on multi-phase manipulation tasks because a state-only MLP cannot resolve which behavioural mode it is in (approaching vs. transporting vs. releasing) without explicit temporal context. This is expected and documented — it is the exact gap ACT addresses.

### 6.2 Primary: ACT (Action Chunking Transformer)

ACT predicts a chunk of `k` future actions from the current observation (no explicit state machine required). The chunk size encodes the planning horizon; `k=30` at 30 Hz means the policy plans 1 second ahead per inference step. Overlapping chunks are blended via temporal ensembling during deployment.

Training hyperparameters for this task: AdamW lr=1e-4, batch 64, chunk size 30, 1000–2000 epochs. Expected convergence on 40–50 demonstrations: 70–90 % closed-loop success rate, consistent with published ACT results on comparable pick-and-place benchmarks.

### 6.3 Optional: PPO Fine-Tuning

The trained IL policy serves as initialisation for PPO (Stable-Baselines3) fine-tuning against a sparse success reward in simulation. The integration point is a `policy_bridge` module that exposes the IL policy's `predict()` and the RL actor's `act(obs, deterministic) -> (action, log_prob, value)`. This connects directly to my thesis work on PPO with shaped rewards.

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

Extensions that fit naturally into the current architecture:

- **VR teleoperation** — richer demonstrations; Oculus / Vision Pro → ROS 2 bridges exist and the data logger is already agnostic to the teleop source
- **Language conditioning** — the dataset carries task strings; CLIP-conditioned policies (OpenVLA, π₀) are a direct upgrade with no data schema change
- **Sim-to-real transfer** — domain randomisation on physics parameters, textures, and lighting in simulation; the ROS 2 interface layer is the same on sim and hardware
- **Active learning** — the webserver already has a live feedback channel; the policy can request a human demonstration on failure and the data logger records it automatically

---

## 10b. A Note on Platform Integration

A point worth being explicit about: the MyBotShop robotic webserver platform is **a commercial product shipped pre-installed with the company's integrated robots**. After reading the documentation at the URL in the brief and surveying MyBotShop's public GitHub presence, I confirmed:

- The webserver is not available as an open-source package, Docker image, or pip install
- The public GitHub organisation contains hardware drivers and example packages (Husky / Aloha v2 / Isaac Sim / Nav2 templates), but no webserver source
- The platform docs reference paths like `/opt/.../webserver/config/robot_webserver.yaml`, consistent with deployment on the customer's pre-flashed device
- Default port (9000), WebSocket-based real-time interface, and "plugin-style extensibility via custom command bindings" are documented

Given this, the design choice was to **integrate against the documented contracts** rather than fabricate a half-installed copy of the platform. Concretely:

- The `pybullet_robot_node` stand-in in this repo plays the role the customer robot plays inside the MyBotShop platform: publishes `/joint_states`, `/cartesian_pose`, listens on `/cmd_robot`. **Any robot that respects these ROS 2 topic contracts is a drop-in replacement, including the actual platform on the lab hardware.**
- The FastAPI service mirrors the platform's WebSocket-based architecture and is intentionally additive — it runs alongside, never replaces. The endpoints in `03_api_specification.md` are the integration surface a real deployment would expose to the existing webserver UI.

This is also why the new ROS 2 nodes (`data_logger_node`, `inference_node`) speak only standard message types (`sensor_msgs/JointState`, `geometry_msgs/PoseStamped`, `geometry_msgs/Twist`) plus three custom services in `il_pipeline_msgs`. They make no assumptions about the platform's internals — only its published interfaces. Swapping the stand-in for the real robot controller does not require any code change in the pipeline; only the topic remapping.

The CEO's note that "the documentation is somewhat outdated, and the platform already contains significantly more functionality than described there" applies here: there are likely platform-side hooks (e.g., specific WebSocket message types for live telemetry, custom action bindings) that the public docs don't describe. Those gaps are best closed on the lab hardware where the actual platform runs.

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

Three new ROS 2 nodes (data logger, inference, pybullet stand-in) and a FastAPI layer make up the implemented pipeline. The stand-in robot node respects the same `/joint_states` + `/cartesian_pose` + `/cmd_robot` interface the real MyBotShop hardware exposes — swapping the simulation for the actual platform controller requires no code change.

Dataset format is LeRobotDataset (parquet). Primary policy is ACT; BC is the validated lower bound. Both train on the same data and deploy through the same inference node. End-to-end latency at 30 Hz is within the control cycle budget on GPU.

The full implementation, dataset, trained checkpoints, and evaluation numbers are in the accompanying repository.
