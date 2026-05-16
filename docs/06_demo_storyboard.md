# Demo Video Storyboard

A 2–3 minute screen recording that walks the CEO through the deliverable. Recorded on the workstation after the full ACT training run completes.

---

## Target

- **Length**: 2–3 minutes
- **Audience**: technical (CEO + engineering team)
- **Goal**: prove the design described in `docs/01_technical_concept.md` actually runs

---

## Storyboard

### Scene 1 — Intro (15 s)

- Screen: terminal + open `docs/01_technical_concept.md` side-by-side
- Voiceover: name, role applying for, one-line summary of the task
- "I built an imitation learning pipeline that plugs into your ROS 2 webserver platform via three new nodes and a small REST + WebSocket API. Here is what it does."

### Scene 2 — Architecture map (20 s)

- Screen: `docs/02_architecture_diagrams.md` open at the system diagram
- Voiceover: walk through the three new nodes (data logger, training service, inference) and how they sit alongside the existing platform
- Highlight: "Nothing in the platform is modified — only new ROS 2 topics and services."

### Scene 3 — Data collection through the API (30 s)

- Screen: terminal split — left pane runs the full stack, right pane curls the API
- Show:
  1. `POST /api/v1/datasets` → returns a dataset id
  2. `POST /api/v1/datasets/{id}/record/start` → ROS bridge dispatches `StartEpisode.srv`
  3. PyBullet window: Franka Panda executing a pick-and-place demonstration
  4. `POST /api/v1/datasets/{id}/record/stop` → returns frame count and parquet path
- Voiceover: "The data logger node turns the existing teleop stream into a structured LeRobotDataset. Episodes are addressable via REST and discoverable via WebSocket."

### Scene 4 — Dataset on disk (15 s)

- Screen: file tree of `/tmp/mybotshop_demos/panda_pickplace_v1/`
- Quick `ls` of `data/chunk-000/`, then `cat meta/info.json` to show schema
- Voiceover: "Parquet shards, one per episode. Stats and episode index in metadata. HuggingFace-compatible — datasets are shareable as-is."

### Scene 5 — Training (15 s)

- Screen: `scripts/train.py` invocation with output streaming
- Voiceover: "Training is one script. Same call signature for BC and ACT, same script runs on CPU and GPU. The workstation runs it with `--device cuda:0` and longer steps."

### Scene 6 — Inference deployment (30 s)

- Screen: PyBullet GUI + terminal showing `/inference_node/load_policy` and `/inference_node/start` service calls
- Voiceover narrating as it runs:
  - "Reset the robot and respawn the cube"
  - "Load the trained policy via service call"
  - "Start inference at 30 Hz"
  - Cube gets picked up and delivered to the green target
- Highlight: "The policy publishes the same Twist action topic the teleop used. The robot doesn't know whether a human or a model is driving."

### Scene 7 — Evaluation results (15 s)

- Screen: `docs/05_evaluation_results.md` Experiment 1 section
- Voiceover: success rate over 20 rollouts (numbers filled in from `scripts/evaluate.sh`), inference latency, training time
- Highlight: honest reporting, including any failure modes observed

### Scene 8 — How to extend (20 s)

- Screen: the `WORKSTATION_RUNBOOK.md` quickly scrolled
- Voiceover:
  - "To use this with the actual platform, replace `pybullet_robot_node` with your platform — the interface contracts are the same."
  - "ACT and Diffusion Policy are wired via the LeRobot library, swappable through one flag in `train.py`."
  - "Optional RL fine-tuning uses my thesis-stack PPO; the hook is already in place."

### Scene 9 — Outro (10 s)

- Screen: Git repo file tree
- Voiceover: "Code, docs, and tests are in the repo. Happy to walk through any part in more detail."

---

## Recording setup

- 1920×1080 at 30 fps
- Microphone: use built-in or a USB mic; record a separate clean voiceover and mix in post if quality matters
- Quiet room, no notifications visible
- `ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 -t 180 -c:v libx264 -pix_fmt yuv420p demo.mp4`

---

## Editing pass (if needed)

- Cut the dead air between phase transitions
- Speed up by 1.25× for any segment where PyBullet is just running and nothing changes
- Subtitles for the voiceover — clearer for non-native English listeners

---

## Final review checklist

- [ ] PyBullet window visible and large
- [ ] Terminal text readable at 1080p (font size ≥ 14)
- [ ] No personal data, paths with username, or credentials visible
- [ ] Success rate stated clearly in scene 7
- [ ] Total length under 3 minutes
