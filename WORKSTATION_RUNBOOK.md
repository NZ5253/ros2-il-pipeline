# Workstation Runbook

Exact commands to reproduce the full deliverable on the university workstation. Designed so the workstation work is purely "more compute" — every script has been validated CPU-side on the dev box.

**Goal:** under one workday of workstation time.

---

## 0. Prerequisites

The workstation should have:

- Ubuntu 22.04 or 24.04
- ROS 2 (Humble or Jazzy — Jazzy preferred to match the dev box)
- Python 3.10+
- A CUDA-capable GPU with recent driver
- ~10 GB free disk space for the dataset, checkpoints, and lerobot models

---

## 1. Clone and set up

```bash
git clone <repo-url> mybotshop_evaluation
cd mybotshop_evaluation

# Python deps (system or venv — match what the ROS 2 nodes will use)
pip install -r requirements.txt
pip install lerobot                         # ACT + Diffusion Policy
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA torch
```

Verify the GPU is visible:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Build the ROS 2 packages

The repo contains two ROS 2 packages: `il_pipeline_msgs` (custom interfaces, ament_cmake) and `il_pipeline` (nodes + launch, ament_python).

```bash
mkdir -p ros2_ws/src
cp -r il_pipeline_msgs ros2_ws/src/
cp -r il_pipeline      ros2_ws/src/
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
cd ..
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

# Collect 40 pick-and-place demos
bash scripts/collect_demos.sh 40 panda_pickplace_v1
```

Expected output: `successes=N/40` with most episodes succeeding. The
script handles dataset creation, episode start/stop, scripted expert
execution, and per-episode success labelling.

---

## 4. Finalise the dataset (compute normalisation stats)

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, "src")
from il_pipeline.dataset.lerobot_writer import LeRobotShardWriter
from pathlib import Path
w = LeRobotShardWriter(root=Path("/tmp/mybotshop_demos"), dataset_name="panda_pickplace_v1")
w.finalise(["observation.state", "action"])
print("stats.json written")
EOF
```

---

## 5. Train the policies

### BC baseline (~2 minutes on GPU)

```bash
python3 scripts/train.py \
    --dataset /tmp/mybotshop_demos/panda_pickplace_v1 \
    --output runs/panda_bc \
    --policy bc \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda:0
```

### ACT primary policy (~30–60 minutes on GPU)

```bash
python3 scripts/train.py \
    --dataset /tmp/mybotshop_demos/panda_pickplace_v1 \
    --output runs/panda_act \
    --policy act \
    --epochs 2000 \
    --batch-size 32 \
    --lr 1e-4 \
    --chunk-size 50 \
    --device cuda:0
```

### Optional: Diffusion Policy comparison

(Same `train.py` once a Diffusion path is added to `policy_factory.py` — left
as stretch goal.)

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

Expected results (with full GPU training):
- BC baseline: 10–30 % (limited by single-step MLP on multi-phase task)
- ACT primary: 70–90 % (action chunking captures the phase structure)

---

## 7. Record demo video

```bash
# In one terminal: launch the full stack with PyBullet GUI on
PYBULLET_GUI=1 bash scripts/run_demo_session.sh

# In another terminal: record the screen
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 \
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

| Step | Time |
|---|---|
| Clone + install deps | 20 min |
| Build msgs + verify nodes | 10 min |
| Collect 40 demos | 15 min |
| BC training | 2 min |
| ACT training (2000 epochs, GPU) | 30–60 min |
| Diffusion Policy training (stretch) | 60–120 min |
| Evaluation rollouts (20 each, BC + ACT) | 20 min |
| Demo video recording | 30 min |
| Polish + push | 30 min |
| **Total** | **~4 hours** |

Comfortable inside one workday.
