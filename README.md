# MyBotShop IL Pipeline

Imitation-learning pipeline for the MyBotShop robotic webserver. Records demos through the existing teleop stream, trains a policy on them, and deploys it back through the same controller the teleoperator drove.

## Results

20 closed-loop rollouts on pick-and-place (random cube spawn, Franka Panda in PyBullet, RTX 4060):

- ACT — 19/20 (95%)
- Diffusion Policy — 18/20 (90%)

Demo: [`demo.mp4`](demo.mp4) — 5 rollouts, ~2:45.

## Repo

```
docs/
  concept.md       what it is, how it's put together, why these choices
  api.md           REST + WebSocket spec
  results.md       numbers
il_pipeline/       ROS 2 ament_python package
il_pipeline_msgs/  three custom .srv files (StartEpisode, StopEpisode, LoadPolicy)
scripts/           collect, train, evaluate, record
tests/             25 unit tests
RUNBOOK.md         reproduce end-to-end on a fresh machine
```

## Run

```bash
# bring up everything
ros2 launch il_pipeline pipeline.launch.py

# or just record a new demo with the trained ACT policy
bash scripts/record_demo.sh
```

Full reproduction in [`RUNBOOK.md`](RUNBOOK.md).

---

Naeem Zain Uddin · naeemzainuddin5253@gmail.com
M.Sc. Automation & Robotics, TU Dortmund
