# Workstation Runbook

End-to-end reproduction. Validated on Windows 11 + WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060.

## Prerequisites

- Ubuntu 22.04 (or WSL Ubuntu 22.04)
- ROS 2 Humble
- Python 3.10+ with `lerobot`, `torch>=2.0`, `pyarrow`, `pybullet`, `fastapi`
- CUDA-capable GPU

Fresh WSL install:
```bash
bash scripts/setup_ros2_wsl.sh    # ROS 2, Python deps, colcon build
```

## 1. Build the ROS 2 packages

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

Verify: `ros2 pkg executables il_pipeline` → `data_logger_node`, `inference_node`, `pybullet_robot_node`.

## 2. Collect demonstrations

```bash
DATASET_ROOT=/mnt/c/$USER/mybotshop_eval/dataset \
    bash scripts/collect_demos.sh 80 panda_pickplace_v2
```

Expected: `successes=78/80` (97.5 % expert success). The script handles dataset creation, episode start/stop, scripted expert execution, and per-episode success labelling. Use a persistent path (not `/tmp/`) so the dataset survives a WSL restart.

## 3. Finalise the dataset

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

Writes `meta/stats.json` (per-feature mean/std/min/max).

## 4. Train

ACT (~140 min on RTX 4060):
```bash
python3 scripts/train.py --policy act \
    --dataset /mnt/c/$USER/mybotshop_eval/dataset/panda_pickplace_v2 \
    --output  runs/panda_act_v2 \
    --epochs 500 --batch-size 32 --chunk-size 50 --lr 1e-4 --device cuda:0
```

Diffusion Policy (~70 min on RTX 4060):
```bash
python3 scripts/train.py --policy diffusion \
    --dataset /mnt/c/$USER/mybotshop_eval/dataset/panda_pickplace_v2 \
    --output  runs/panda_diffusion_v2 \
    --epochs 500 --batch-size 64 --horizon 16 --lr 1e-4 --device cuda:0
```

## 5. Evaluate

```bash
EVAL_DEVICE=cuda:0 bash scripts/evaluate.sh runs/panda_act_v2/best.pt       act       20
EVAL_DEVICE=cuda:0 bash scripts/evaluate.sh runs/panda_diffusion_v2/best.pt diffusion 20
```

20 closed-loop rollouts each. Reference results: ACT 19/20 = 95 %, DP 18/20 = 90 %.

## 6. Record the demo video

PyBullet's GUI in WSL needs a virtual framebuffer; `record_demo.sh` sets up Xvfb + Mesa + ffmpeg and writes `demo.mp4`:

```bash
sudo apt-get install -y xvfb ffmpeg    # once
bash scripts/record_demo.sh            # writes demo.mp4 (1024x720, 5 rollouts)
```

## 7. Submission deliverables

1. `docs/01_concept.md` — technical concept
2. `docs/05_results.md` — evaluation numbers
3. `runs/panda_act_v2/best.pt` — the 95 % checkpoint
4. `demo.mp4` — 2:45 walkthrough
5. Git repository link

## Time budget on RTX 4060

| Step | Time |
|---|---:|
| Build + verify | 10 min |
| Collect 80 demos | 30 min |
| Stats | < 1 min |
| Train ACT (500 ep) | ~140 min |
| Train Diffusion (500 ep) | ~70 min |
| Evaluate (20 rollouts × 2 policies) | ~20 min |
| Record demo | 5 min |
| **Total** | **~4.5 hours** |
