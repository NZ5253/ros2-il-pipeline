# Imitation Learning Pipeline — MyBotShop Webserver Integration

A ROS 2 imitation-learning pipeline for manipulation, designed as a modular extension to the MyBotShop robotic webserver. Demonstrations are recorded through the platform's teleop stream, stored as LeRobotDataset shards, and a trained policy is deployed back through the same `/cmd_robot` topic the teleoperator drove.

## Headline result

| Policy | Closed-loop success | Inference latency | Notes |
|---|---:|---:|---|
| **ACT** (primary) | **19 / 20 = 95 %** | < 20 ms | LeRobot ACT, temporal ensembling |
| **Diffusion Policy** | **18 / 20 = 90 %** | < 30 ms | LeRobot DP, 10-step DDIM at deploy |

Task: pick a randomly-spawned red cube and place it on a green target zone (8 cm radius) using a 7-DOF Franka Panda in PyBullet. 80 demonstrations, 30 Hz control loop, RTX 4060.

Demo video: [`demo.mp4`](demo.mp4) — 2:45, five consecutive rollouts.

## Repository layout

```
docs/
  01_concept.md             technical concept
  02_architecture.md        node + topic diagram
  03_api.md                 REST + WebSocket API
  04_dataset.md             LeRobotDataset schema
  05_results.md             experiment numbers
il_pipeline/                ROS 2 ament_python package
  il_pipeline/nodes/        data_logger, inference, pybullet_robot
  il_pipeline/dataset/      LeRobot parquet writer + frame validator
  il_pipeline/training/     policies (BC, ACT, Diffusion) + dataset loader
  il_pipeline/inference/    policy loader + normaliser
  il_pipeline/web_api/      FastAPI service + ROS bridge
il_pipeline_msgs/           ROS 2 .srv / .msg / .action definitions
scripts/                    collect / train / evaluate / record_demo
tests/                      25 unit tests
WORKSTATION_RUNBOOK.md      end-to-end reproduction
```

## Run the demo

Requires ROS 2 Humble, Python 3.10, CUDA, the `lerobot` library.

```bash
# Bring up the full stack
ros2 launch il_pipeline pipeline.launch.py

# Record a fresh demo.mp4 with the trained ACT policy
bash scripts/record_demo.sh
```

Full reproduction (collect → train → evaluate) in [`WORKSTATION_RUNBOOK.md`](WORKSTATION_RUNBOOK.md).

## Contact

Naeem Zain Uddin · naeemzainuddin5253@gmail.com
M.Sc. Automation & Robotics, TU Dortmund
