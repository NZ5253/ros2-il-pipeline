# Imitation Learning Pipeline for the MyBotShop Robotic Webserver

A ROS 2-native imitation learning pipeline for manipulation, designed as a modular extension to the MyBotShop robotic webserver platform.

**Status:** Working end-to-end on Windows 11 (WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060). All three nodes verified live, 23/23 unit tests passing, BC + ACT both train and deploy through the inference node. Real demonstration collection and GPU training in progress.

---

## What This Is

A complete pipeline for:

1. **Collecting demonstrations** through the existing teleoperation interface
2. **Storing them** as a standard LeRobotDataset (HuggingFace-compatible)
3. **Training** behaviour cloning, ACT, or Diffusion Policy on the collected data
4. **Deploying** the trained policy back through the same control interface

Optional extension: PPO-based RL fine-tuning of the IL policy.

The pipeline integrates with the MyBotShop platform through ROS 2 service and topic contracts only — no platform-internal code is modified. A `pybullet_robot_node` stand-in is included so the pipeline can be developed and validated without the actual platform installed.

## Verified End-to-End

| Step | Machine | Result |
|---|---|---|
| Synthetic pipeline check (30 eps, BC, 200 epochs) | Windows / RTX 4060 | val loss 0.192, MAE 0.124, p50 latency 0.08 ms |
| 3-node live pipeline (pybullet + data_logger + inference) | WSL / ROS 2 Humble | 30 Hz sustained, all topics verified |
| FastAPI health + dataset + policy endpoints | Windows (dry-run) | all return correct JSON |
| BC + ACT checkpoints load through inference_node | WSL | ✓ |
| 23 unit tests | Windows | passing |
| 50-ep real demo collection + GPU training | in progress | — |

---

## Repository Layout

```
mybotshop_evaluation/
├── PLAN.md                              project plan
├── README.md                            this file
├── WORKSTATION_RUNBOOK.md               exact lab PC commands
├── Projektübersicht.md                  brief project overview for the application
├── pyproject.toml                       repo-wide pytest + ruff config
├── requirements.txt                     non-ROS Python deps
├── docs/                                technical documentation
│   ├── 01_technical_concept.md          main technical document
│   ├── 02_architecture_diagrams.md      6 mermaid diagrams
│   ├── 03_api_specification.md          REST + WebSocket API spec
│   ├── 04_dataset_schema.md             LeRobotDataset schema details
│   ├── 05_evaluation_results.md         real numbers from CPU validation
│   ├── 06_demo_storyboard.md            video recording plan
│   └── diagrams/                        training-curve PNGs
├── il_pipeline/                         ROS 2 ament_python package
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/il_pipeline             ament index marker
│   ├── launch/pipeline.launch.py
│   └── il_pipeline/                     the Python package
│       ├── nodes/                       data_logger, inference, pybullet_robot
│       ├── dataset/                     LeRobot parquet writer + frame validator
│       ├── training/                    BC policy + LeRobot dataset adapter
│       ├── inference/                   policy loader + normaliser
│       └── web_api/                     FastAPI service + ROS bridge
├── il_pipeline_msgs/                    ROS 2 ament_cmake package (.srv/.msg/.action)
├── configs/                             YAML parameter files
├── scripts/                             CLI utilities (collect, train, evaluate, plot, demo)
└── tests/                               23 unit tests
```

---

## Reading Order

If you want the high-level concept, read in this order:

1. [`docs/01_technical_concept.md`](docs/01_technical_concept.md) — the main document
2. [`docs/02_architecture_diagrams.md`](docs/02_architecture_diagrams.md) — visual overview
3. [`docs/04_dataset_schema.md`](docs/04_dataset_schema.md) — dataset format
4. [`docs/03_api_specification.md`](docs/03_api_specification.md) — web API

If you want to read code: start at the ROS 2 nodes in [`il_pipeline/il_pipeline/nodes/`](il_pipeline/il_pipeline/nodes/), then the FastAPI service in [`il_pipeline/il_pipeline/web_api/app.py`](il_pipeline/il_pipeline/web_api/app.py).

---

## Setup (Lab PC)

### Requirements

- Ubuntu 22.04 or 24.04
- ROS 2 (Humble or Jazzy)
- Python 3.10+
- CUDA-capable GPU (for ACT / Diffusion training)
- `lerobot` HuggingFace library
- PyTorch 2.x

### Install

```bash
# Clone this repo
git clone <repo-url> mybotshop_evaluation
cd mybotshop_evaluation

# Python deps (lab PC)
pip install -r requirements.txt

# LeRobot
pip install lerobot

# The MyBotShop platform itself is a commercial product that ships
# pre-installed with their robot hardware (no public install path).
# See docs/01_technical_concept.md section 10b for the integration model.
```

### Build the ROS 2 packages

```bash
mkdir -p ~/il_ws/src
ln -sfn $PWD/il_pipeline ~/il_ws/src/il_pipeline
ln -sfn $PWD/il_pipeline_msgs ~/il_ws/src/il_pipeline_msgs
cd ~/il_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select il_pipeline_msgs
source install/setup.bash
colcon build --packages-select il_pipeline
source install/setup.bash
```

### Run

```bash
# Single-command bring-up (data logger + inference + pybullet sim + FastAPI):
ros2 launch il_pipeline pipeline.launch.py

# Or each node individually:
ros2 run il_pipeline pybullet_robot_node      # simulated Franka stand-in
ros2 run il_pipeline data_logger_node
ros2 run il_pipeline inference_node
python3 -m uvicorn il_pipeline.web_api.app:app --host 0.0.0.0 --port 8011
```

OpenAPI docs at `http://localhost:8011/docs`.

---

## Quick Walkthrough

### 1. Collect demonstrations

```bash
# Through the web UI, or via curl
curl -X POST http://localhost:8000/api/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{"name": "panda_pickplace_v1", "robot_model": "franka_panda", ...}'

curl -X POST http://localhost:8000/api/v1/datasets/<id>/record/start \
  -H 'Content-Type: application/json' \
  -d '{"episode_name": "demo_001"}'

# Drive the robot through the demonstration with the existing teleop UI

curl -X POST http://localhost:8000/api/v1/datasets/<id>/record/stop \
  -H 'Content-Type: application/json' \
  -d '{"outcome": "success"}'
```

### 2. Train a policy

```bash
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "<id>",
    "policy_type": "act",
    "epochs": 2000,
    "chunk_size": 50
  }'

# Watch progress via WebSocket
wscat -c ws://localhost:8000/ws/training/<job_id>/progress
```

### 3. Deploy and run

```bash
curl -X POST http://localhost:8000/api/v1/policies/<policy_id>/deploy
curl -X POST http://localhost:8000/api/v1/policies/<policy_id>/start

# Watch the robot execute the task in simulation or on hardware
```

---

## Evaluation Methodology

The pipeline is evaluated on a pick-and-place task in simulation:

| Experiment | Setup | Metric |
|---|---|---|
| Pipeline correctness | 20 demos, BC | Task success in sim |
| Sample efficiency | 5/10/20/50 demos | Success rate per dataset size |
| Distribution shift | Train and eval on shifted object poses | Success rate degradation |
| Latency | End-to-end obs→action | Should be < 50 ms |

Results, including negative findings, are documented in `docs/05_evaluation_results.md` (added after the prototype runs on the lab PC).

---

## Status Checklist

- ✅ Technical concept, architecture diagrams, API spec, dataset schema
- ✅ ROS 2 nodes (data_logger, inference, pybullet_robot) wired and verified live
- ✅ Custom interfaces (`il_pipeline_msgs`) built and registered with ROS 2
- ✅ Dataset writer, frame validator, LeRobot parquet round-trip
- ✅ Training pipeline (BC and ACT both)
- ✅ FastAPI web layer with live ROS bridge
- ✅ Synthetic end-to-end validated (data → train → eval, GPU)
- ✅ Live ROS 2 pipeline verified (3 nodes, 30 Hz)
- ✅ BC and ACT train and load through inference_node
- 🟡 50-episode real demo collection (running)
- 🟡 GPU training on real demos (BC + ACT)
- 🟡 Closed-loop evaluation rollouts
- 🟡 Demo video

---

## Contact

Naeem Zain Uddin · naeemzainuddin5253@gmail.com · +49 176 43277891
M.Sc. Automation & Robotics, TU Dortmund (graduated May 2026)
Portfolio: nz5253.github.io
