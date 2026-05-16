# Imitation Learning Pipeline for the MyBotShop Robotic Webserver

A ROS 2-native imitation learning pipeline for manipulation, designed as a modular extension to the MyBotShop robotic webserver platform.

**Status:** Working end-to-end. Full pipeline validated on a real ROS 2 stack with a simulated Franka Panda solving pick-and-place. ACT trained on CPU as proof-of-concept; full ACT training is the workstation step (see [WORKSTATION_RUNBOOK.md](WORKSTATION_RUNBOOK.md)).

---

## What This Is

A complete pipeline for:

1. **Collecting demonstrations** through the existing teleoperation interface
2. **Storing them** as a standard LeRobotDataset (HuggingFace-compatible)
3. **Training** behaviour cloning, ACT, or Diffusion Policy on the collected data
4. **Deploying** the trained policy back through the same control interface

Optional extension: PPO-based RL fine-tuning of the IL policy.

The pipeline integrates with the MyBotShop platform through ROS 2 service and topic contracts only — no platform-internal code is modified. A `pybullet_robot_node` stand-in is included so the pipeline can be developed and validated without the actual platform installed.

## Verified End-to-End (this dev box, CPU)

| Step | Result |
|---|---|
| 40 pick-and-place demonstrations via HTTP → ROS → parquet | 40 / 40 success |
| BC training (200 epochs, 22K frames) | 4 min, val loss 0.0099 |
| BC closed-loop rollouts | 1 / 10 (baseline for the multi-phase task) |
| ACT training (LeRobot 0.5.x, 5.8M params, CPU) | trains end-to-end |
| ACT checkpoint loads through inference_node | ✓ |
| FastAPI REST + WebSocket layer dispatches typed ROS services | ✓ |
| 23 unit tests | passing |

---

## Repository Layout

```
mybotshop_evaluation/
├── PLAN.md                              project plan
├── README.md                            this file
├── docs/
│   ├── 01_technical_concept.md          main technical document
│   ├── 02_architecture_diagrams.md      system, ROS 2, data, training, inference diagrams
│   ├── 03_api_specification.md          REST + WebSocket API spec
│   └── 04_dataset_schema.md             LeRobotDataset schema details
├── src/
│   ├── il_pipeline/
│   │   ├── nodes/
│   │   │   ├── data_logger_node.py      records demonstrations
│   │   │   └── inference_node.py        runs the trained policy
│   │   ├── dataset/
│   │   │   ├── lerobot_writer.py        parquet shard writer
│   │   │   └── frame_validator.py       per-frame validation
│   │   ├── training/
│   │   │   ├── train.py                 training entry point
│   │   │   ├── policy_factory.py        BC/ACT/Diffusion dispatch
│   │   │   └── lerobot_torch_dataset.py PyTorch dataset adapter
│   │   └── inference/
│   │       ├── policy_loader.py         checkpoint loader
│   │       └── normaliser.py            input/output normalisation
│   └── web_api/
│       └── app.py                       FastAPI service
├── configs/                             example launch configs and scenes
├── scripts/                             helper scripts (data collection, eval)
└── tests/                               unit tests
```

---

## Reading Order

If you want the high-level concept, read in this order:

1. [`docs/01_technical_concept.md`](docs/01_technical_concept.md) — the main document
2. [`docs/02_architecture_diagrams.md`](docs/02_architecture_diagrams.md) — visual overview
3. [`docs/04_dataset_schema.md`](docs/04_dataset_schema.md) — dataset format
4. [`docs/03_api_specification.md`](docs/03_api_specification.md) — web API

If you want to read code: start at the ROS 2 nodes in [`src/il_pipeline/nodes/`](src/il_pipeline/nodes/), then the FastAPI service in [`src/web_api/app.py`](src/web_api/app.py).

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

# Install the MyBotShop platform locally (see their docs)
# https://docs.mybotshop.de/projects/product_robot_webserver/html/index.html
```

### Run

```bash
# Terminal 1 — start the MyBotShop platform (according to their docs)

# Terminal 2 — start the data logger
ros2 run il_pipeline data_logger_node --ros-args \
    --params-file configs/data_logger.yaml

# Terminal 3 — start the inference node (idle until a policy is loaded)
ros2 run il_pipeline inference_node --ros-args \
    --params-file configs/inference.yaml

# Terminal 4 — start the web API
uvicorn il_pipeline.web_api.app:app --host 0.0.0.0 --port 8000
```

OpenAPI docs are available at `http://localhost:8000/api/v1/docs`.

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
- ✅ 40-episode pick-and-place dataset collected (100 % expert success)
- ✅ BC trained and evaluated end-to-end
- ✅ ACT (LeRobot 0.5.x) trains and loads through inference node
- 🟡 Full ACT training (workstation GPU, see WORKSTATION_RUNBOOK.md)
- 🟡 Final evaluation rollouts on workstation-trained ACT
- 🟡 Demo video

---

## Contact

Naeem Zain Uddin · naeemzainuddin5253@gmail.com · +49 176 43277891
M.Sc. Automation & Robotics, TU Dortmund (graduated May 2026)
Portfolio: nz5253.github.io
