# Runbook

Reproduce the whole thing on a fresh machine. Validated on Windows 11 + WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060.

## Prereqs

Ubuntu 22.04 (or WSL), ROS 2 Humble, Python 3.10, CUDA. The Python deps are in `requirements.txt`; `lerobot`, `pybullet`, `torch`, `pyarrow`, `fastapi` are the load-bearing ones.

If you're on a fresh WSL, run `bash scripts/setup_ros2_wsl.sh` — it installs ROS 2, the Python deps, and runs the colcon build.

## Build

```bash
mkdir -p ~/il_ws/src
ln -sfn $PWD/il_pipeline      ~/il_ws/src/il_pipeline
ln -sfn $PWD/il_pipeline_msgs ~/il_ws/src/il_pipeline_msgs
cd ~/il_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select il_pipeline_msgs
source install/setup.bash
colcon build --packages-select il_pipeline
source install/setup.bash
cd -
```

Sanity check: `ros2 pkg executables il_pipeline` lists `data_logger_node`, `inference_node`, `pybullet_robot_node`.

## Collect demos

```bash
DATASET_ROOT=/mnt/c/$USER/mybotshop_eval/dataset \
    bash scripts/collect_demos.sh 80 panda_pickplace_v2
```

Expect ~78/80 expert successes. Picks a persistent path so the dataset survives a WSL restart (`/tmp` doesn't).

## Finalise the dataset

```bash
python3 -c "
import sys; sys.path.insert(0, 'il_pipeline')
from il_pipeline.dataset.lerobot_writer import LeRobotShardWriter
from pathlib import Path
LeRobotShardWriter(
    root=Path('/mnt/c/$USER/mybotshop_eval/dataset'),
    dataset_name='panda_pickplace_v2'
).finalise(['observation.state', 'action'])
"
```

This writes `meta/stats.json` (mean / std / min / max per feature).

## Train

ACT (~140 min on RTX 4060):

```bash
python3 scripts/train.py --policy act \
    --dataset /mnt/c/$USER/mybotshop_eval/dataset/panda_pickplace_v2 \
    --output  runs/panda_act_v2 \
    --epochs 500 --batch-size 32 --chunk-size 50 --lr 1e-4 --device cuda:0
```

Diffusion Policy (~70 min):

```bash
python3 scripts/train.py --policy diffusion \
    --dataset /mnt/c/$USER/mybotshop_eval/dataset/panda_pickplace_v2 \
    --output  runs/panda_diffusion_v2 \
    --epochs 500 --batch-size 64 --horizon 16 --lr 1e-4 --device cuda:0
```

## Evaluate

```bash
EVAL_DEVICE=cuda:0 bash scripts/evaluate.sh runs/panda_act_v2/best.pt       act       20
EVAL_DEVICE=cuda:0 bash scripts/evaluate.sh runs/panda_diffusion_v2/best.pt diffusion 20
```

Reference numbers: ACT 19/20, DP 18/20.

## Record the demo video

PyBullet's GUI under WSL needs a virtual framebuffer. `record_demo.sh` sets up Xvfb + Mesa + ffmpeg:

```bash
sudo apt-get install -y xvfb ffmpeg     # once
bash scripts/record_demo.sh             # writes demo.mp4 (1024x720, 5 rollouts)
```

## Time budget

| Step | RTX 4060 |
|---|---:|
| Build | 10 min |
| Collect 80 demos | 30 min |
| Stats | < 1 min |
| Train ACT | ~140 min |
| Train DP | ~70 min |
| Evaluate (20 × 2 policies) | ~20 min |
| Record demo | 5 min |
| **Total** | **~4.5 h** |
