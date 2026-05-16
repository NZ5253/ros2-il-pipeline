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

## 2. Build the custom ROS 2 interfaces

```bash
mkdir -p ros2_ws/src
cp -r il_pipeline_msgs ros2_ws/src/
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select il_pipeline_msgs \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
cd ..
```

Verify:

```bash
ros2 interface show il_pipeline_msgs/srv/StartEpisode
```

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
bash scripts/evaluate.sh runs/panda_bc/best.pt   bc   20    # BC, 20 rollouts
bash scripts/evaluate.sh runs/panda_act/best.pt  act  20    # ACT, 20 rollouts
```

The script resets the sim between rollouts, deploys the policy via
`/inference_node/load_policy`, watches `/task_status`, and reports
success rate.

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

The following were already verified on the dev box and need no rerun:

- ROS 2 service contracts (`StartEpisode`, `StopEpisode`, `LoadPolicy`)
- HTTP → ROS bridge for the FastAPI layer
- LeRobotDataset parquet round-trip (read/write/normalise)
- Frame validator and dataset writer unit tests (23 passing)
- BC training loop convergence

Only re-run those if a workstation-specific dependency breaks them.
