# Workstation Runbook

Steps to reproduce the full pipeline. Validated on Windows 11 + WSL Ubuntu 22.04 + ROS 2 Humble + RTX 4060.

---

## 0. Prerequisites

- Ubuntu 22.04 (or WSL Ubuntu 22.04 on Windows)
- ROS 2 Humble
- Python 3.10+
- CUDA-capable GPU
- ~10 GB free disk space

If starting from scratch on WSL:
```bash
bash scripts/setup_ros2_wsl.sh   # installs ROS 2, Python deps, builds both packages
```

---

## 1. Clone and set up

```bash
git clone <repo-url> mybotshop_evaluation
cd mybotshop_evaluation

pip install -r requirements.txt
pip install lerobot
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Build the ROS 2 packages

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
cd -
```

Verify:

```bash
ros2 pkg list | grep il_pipeline       # both packages present
ros2 interface show il_pipeline_msgs/srv/StartEpisode
ros2 pkg executables il_pipeline       # data_logger_node, inference_node, pybullet_robot_node
ros2 launch il_pipeline pipeline.launch.py --show-args
```

If `ros2 pkg list` shows `il_pipeline_msgs` but not `il_pipeline`, your colcon-ros may have a hook-generation quirk (seen on Linux Mint with colcon-ros 0.5.0). Workaround:

```bash
export AMENT_PREFIX_PATH=$(pwd)/ros2_ws/install/il_pipeline:$AMENT_PREFIX_PATH
```

Stock Ubuntu 24.04 + ROS Jazzy installs work without this workaround.

---

## 3. Collect demonstrations

The PyBullet stand-in is included; for the workstation submission you can
either (a) use it as on the dev box, or (b) swap in the actual MyBotShop
platform if installed. The data logger and FastAPI service don't change
either way.

```bash
# Source both ROS layers
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Collect 80 pick-and-place demos (matches the published ACT v2 results)
DATASET_ROOT=/mnt/c/$USER/mybotshop_eval/dataset \
    bash scripts/collect_demos.sh 80 panda_pickplace_v2
```

Expected output: `successes=N/80` with most episodes succeeding (78/80 = 97.5 % on the reference run). The script handles dataset creation, episode start/stop, scripted expert execution, and per-episode success labelling. The `enable_object_pose=true` default means `/cube_pose` is recorded as part of `observation.state` — needed for the object-aware policies in Section 5.

---

## 4. Finalise the dataset (compute normalisation stats)

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, "il_pipeline")
from il_pipeline.dataset.lerobot_writer import LeRobotShardWriter
from pathlib import Path
w = LeRobotShardWriter(root=Path("/path/to/dataset"), dataset_name="panda_pickplace_v2")
w.finalise(["observation.state", "action"])
print("stats.json written")
EOF
```

Use a persistent path (not `/tmp/`) so data survives WSL restarts. The dataset at `/mnt/c/...` or `/root/` is recommended.

---

## 5. Train the policies

### BC baseline (~5 minutes on GPU)

```bash
python3 scripts/train.py \
    --dataset /path/to/dataset/panda_pickplace_v2 \
    --output runs/panda_bc \
    --policy bc \
    --epochs 500 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda:0
```

### ACT primary policy (~80 minutes on GPU)

```bash
python3 scripts/train.py \
    --dataset /path/to/dataset/panda_pickplace_v2 \
    --output runs/panda_act \
    --policy act \
    --epochs 500 \
    --batch-size 32 \
    --lr 1e-4 \
    --chunk-size 50 \
    --device cuda:0
```

`--epochs 500` is sufficient for convergence on this dataset with RTX 4060. Use `--epochs 2000` only if you have ~5 hours and want to squeeze out extra performance.

### Diffusion Policy (~110 minutes on GPU)

```bash
python3 scripts/train.py \
    --dataset /path/to/dataset/panda_pickplace_v2 \
    --output runs/panda_diffusion \
    --policy diffusion \
    --epochs 500 \
    --batch-size 64 \
    --horizon 16 \
    --lr 1e-4 \
    --device cuda:0
```

UNet sized to ~4.5 M params (`down_dims=(64,128,256)` in `build_diffusion_policy`) to match ACT for a fair comparison. Inference runs 10 DDIM steps per chunk so per-step latency stays within the 30 Hz control budget.

---

## 6. Evaluate via closed-loop rollouts

```bash
# CPU inference is fine even on the workstation — saves GPU for training
EVAL_DEVICE=cuda:0 bash scripts/evaluate.sh runs/panda_bc/best.pt   bc   20
EVAL_DEVICE=cuda:0 bash scripts/evaluate.sh runs/panda_act/best.pt  act  20
```

The script resets the sim between rollouts, deploys the policy via
`/inference_node/load_policy`, watches `/task_status`, and reports
success rate.

Measured results on the 80-demo object-aware dataset (RTX 4060, 20 rollouts):

| Policy | Success rate | Notes |
|---|---|---|
| BC | 0/20 = 0 % | Exposes causal confusion — see `docs/05_evaluation_results.md` Experiment 4 |
| ACT | 19/20 = 95 % | Top of published range; canonical temporal ensembling |
| Diffusion Policy | (see eval doc) | Same dataset, ~4.5 M params |

For the object-blind v1 baseline (no cube pose in observation), BC scored 15 % and ACT scored 35 %. The v1→v2 jump for ACT (35 % → 95 %) is purely from including the task object's pose in `observation.state` — no architectural change.

---

## 7. Record demo video

On WSL, PyBullet's GUI window requires a virtual framebuffer. `record_demo.sh` handles everything — Xvfb, Mesa software renderer, ffmpeg capture, and the demo session — in one command:

```bash
# Install once if not already present
sudo apt-get install -y xvfb ffmpeg

# Record 5 rollouts with GUI to demo.mp4
bash scripts/record_demo.sh
```

Alternatively, to record manually (native Linux with real X11):

```bash
PYBULLET_GUI=1 bash scripts/run_demo_session.sh &
ffmpeg -video_size 1280x720 -framerate 30 -f x11grab -i :0.0 \
       -t 180 -c:v libx264 -pix_fmt yuv420p demo.mp4
```

A storyboard for the recording is in `docs/06_demo_storyboard.md`.

---

## 8. Final submission

The deliverable for the CEO:

1. **`docs/01_technical_concept.md`** — the main document
2. **Git repository link** — public or shared
3. **`runs/panda_act/best.pt`** — the trained policy checkpoint
4. **`demo.mp4`** — 2–3 minute walkthrough
5. **`docs/05_evaluation_results.md`** — populated with the real experiment results

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: em` during colcon build | `sudo apt install python3-empy` and force `Python3_EXECUTABLE=/usr/bin/python3` in colcon cmake args |
| Path with spaces breaks colcon | Build under `/tmp/mybotshop_ws` and symlink |
| `lerobot` import errors | `pip install --upgrade lerobot huggingface-hub` |
| Inference latency too high | Drop to `execution_mode: temporal_ensemble` instead of `first_action`; reduce `inference_rate_hz` |
| Cube not gripped reliably | Constraint-based grasp threshold in `pybullet_robot_node.py` (`_maybe_attach_grasp`) — tune the 0.08 m distance |

---

## What is *not* on the workstation critical path

The following were already verified end-to-end on the dev box and need no rerun:

- ROS 2 service contracts (`StartEpisode`, `StopEpisode`, `LoadPolicy`) on Jazzy
- HTTP → ROS bridge dispatching typed service calls through FastAPI
- LeRobotDataset parquet round-trip (read / write / normalise stats)
- Frame validator and dataset writer unit tests (23 passing)
- BC training loop convergence on real pick-and-place data
- BC closed-loop rollouts: pipeline successful, 1/10 success on CPU baseline
- ACT (LeRobot 0.5.x) training end-to-end on CPU — 5.8M params, gradients flow,
  checkpoint loads through the inference node's policy loader
- 40-episode pick-and-place dataset collected with 100 % expert success rate
- Constraint-based grasping in the PyBullet stand-in robot

Only re-run those if a workstation-specific dependency breaks them.

---

## Estimated workstation time budget

| Step | Time (measured on RTX 4060) |
|---|---|
| Clone + install deps | 20 min |
| Build msgs + verify nodes | 10 min |
| Collect 80 demos | 30 min (78/80 expert success on reference run) |
| Compute stats (finalise) | < 1 min |
| BC training (500 epochs, GPU) | ~10 min |
| ACT training (500 epochs, GPU) | ~140 min (longer with 2× data vs v1) |
| Diffusion training (500 epochs, GPU) | ~110 min |
| Evaluation rollouts (20 each × 3 policies) | ~30 min |
| Demo video recording | 30 min |
| Polish + push | 20 min |
| **Total (BC+ACT only)** | **~4 hours** |
| **Total (with Diffusion Policy)** | **~5.5 hours** |

Note: ACT with `--epochs 2000` on RTX 4060 takes ~5 hours, not 30–60 min. Use `--epochs 500` for a 80-minute run that still converges on this dataset.
