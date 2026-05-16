# MyBotShop Technical Evaluation — Project Plan

**Brief:** Design (and implement, where feasible) an imitation learning pipeline for manipulation, integrated with MyBotShop's existing ROS 2 robotic webserver platform.

**Target deliverable level:** Working prototype with real IL training + live demo + full technical write-up + Git repository.

**Documentation reference:** https://docs.mybotshop.de/projects/product_robot_webserver/html/index.html

---

## 1. What "Best Possible" Looks Like

A complete deliverable package:

1. **Working prototype** — their platform installed locally, IL pipeline running end-to-end in simulation
2. **Real training** — actual demonstrations collected via teleop, real policy trained, real evaluation
3. **Live demo** — trained policy executing a manipulation task (pick-and-place baseline)
4. **Technical document (~10–15 pages)** — problem framing, architecture, design choices, results, future work
5. **Architecture diagrams** — system-level, ROS 2 node graph, data flow, training pipeline
6. **Git repository** — clean code, well-documented, runnable
7. **Demo video** — 2–3 minute walkthrough

This signals senior-level systems thinking, not just a homework assignment.

---

## 2. Recommended Technology Stack

| Layer | Choice | Why |
|---|---|---|
| **ROS distribution** | ROS 2 Humble or Jazzy (match their platform) | Their platform is ROS 2-based; check docs for exact version |
| **Simulation** | Gazebo Harmonic (or Isaac Sim if they support it) | ROS 2 native, well-supported; Isaac if GPU sim is wanted |
| **Robot model** | Franka Panda OR UR5e | Research-standard, abundant URDFs, good for manipulation IL |
| **IL framework** | LeRobot (HuggingFace) | Modern, supports ACT / Diffusion Policy / pi0, growing community, good dataset format |
| **Policy architecture** | ACT first, Diffusion Policy as stretch | ACT is lighter to train and converges fast; Diffusion is SOTA but heavier |
| **Dataset format** | LeRobotDataset (parquet-based) | Standard, integrates with HF Hub, has visualisation tools built in |
| **Training** | PyTorch on lab PC GPU | Existing strength |
| **RL (optional)** | Stable-Baselines3 PPO | Thesis carry-over, well-tested |
| **Teleoperation** | Keyboard / joy + Foxglove for monitoring | Simple, demonstrates the concept; can extend to VR if time |
| **Webserver integration** | FastAPI + WebSocket bridge to ROS 2 | Lightweight, matches "robotic webserver" framing |
| **Visualization** | Foxglove Studio | Modern ROS 2 visualization, good for evaluation streaming |
| **Repo / docs** | Git + README + architecture markdown | Clean, professional |

---

## 3. Architecture (High Level)

```
┌─────────────────────────────────────────────────────────────────┐
│                     MyBotShop Webserver Platform                │
│   (existing: teleop, waypoint nav, monitoring, visualization)   │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
               │ adds:                            │ adds:
               ▼                                  ▼
       ┌───────────────┐               ┌──────────────────────┐
       │ Data Logger   │               │ Inference Node       │
       │ Node (ROS 2)  │               │ (ROS 2, loads policy)│
       └───────┬───────┘               └──────────┬───────────┘
               │                                  │
               ▼                                  │
       ┌───────────────┐                          │
       │ Dataset       │                          │
       │ (LeRobot      │                          │
       │  Parquet)     │                          │
       └───────┬───────┘                          │
               │                                  │
               ▼                                  │
       ┌───────────────┐                          │
       │ Training      │       trained policy     │
       │ Pipeline      │─────────────────────────►│
       │ (PyTorch +    │                          │
       │  LeRobot)     │                          │
       └───────────────┘                          │
                                                  ▼
                                          ┌──────────────┐
                                          │  Robot Arm   │
                                          │  (sim/real)  │
                                          └──────────────┘

Web layer (FastAPI):
  /datasets/  CRUD on collected episodes
  /train/     start/stop training jobs
  /policies/  list, download, deploy trained policies
  /eval/      live evaluation streaming
```

**Three new components added to their platform:**
1. **Data Logger Node** — subscribes to robot state + teleop commands, writes LeRobot-format dataset
2. **Inference Node** — loads trained policy, publishes action commands during deployment
3. **Web API extensions** — endpoints for dataset, training, policy, evaluation management

---

## 4. ROS 2 Node Structure

```
/teleop_input            (existing — their teleop)
   │
   ▼
/cmd_robot               (action commands to robot)
   │
   ├──► /robot_controller (existing — their controller)
   │
   └──► /data_logger      (new — records episodes)

/robot_state             (existing — joint states, EE pose)
   │
   └──► /data_logger      (new)

/episode_control         (new — start/stop/save episode)

# Deployment side:
/observation             (state + optional camera)
   │
   ▼
/il_inference            (new — runs trained policy)
   │
   ▼
/cmd_robot               (publishes to existing controller)
```

**Custom messages / services:**
- `EpisodeControl.srv` — start, stop, save with metadata
- `PolicyDeploy.srv` — load policy by name, set inference rate
- `Observation.msg` — joint state + EE pose + camera (optional)
- `Action.msg` — joint velocity or end-effector delta

---

## 5. Dataset Format

Use **LeRobotDataset** schema (HuggingFace standard):

```
episode_index
frame_index
timestamp
observation.state         [joint positions, velocities, EE pose]
observation.image.cam_0   [RGB if camera used]
action                    [delta joint or delta EE pose]
next.reward               [optional, for RL extension]
next.done                 [episode boundary]
task                      [task name/instruction]
```

Stored as **parquet shards** (~episode-per-shard). Compatible with `datasets` library, HF Hub upload, LeRobot visualisation tools out of the box.

---

## 6. Learning Pipeline

**Baseline (must-have):**
- Behavior Cloning with MLP — sanity check, fast to train
- ACT (Action Chunking Transformer) — primary IL policy

**Stretch (if time allows):**
- Diffusion Policy — compare against ACT
- RL fine-tuning with SB3 PPO using the BC/ACT policy as initialisation

**Training pipeline structure:**
```
LeRobotDataset (parquet)
   ↓
PyTorch DataLoader (normalise, shuffle, batch)
   ↓
ACT model (transformer encoder + temporal action chunking)
   ↓
Loss: L1 on action chunks + KL on latent (if VAE variant)
   ↓
Checkpoint per N steps
   ↓
Best-checkpoint export for inference node
```

---

## 7. Phase Breakdown (Post-Defense, ~12 days)

| Phase | Days | Goal | Deliverable |
|---|---|---|---|
| 1. Docs + research | 1 | Read MyBotShop docs, IL framework survey, lock scope | Brief design note |
| 2. Architecture design | 1 | Diagrams, node structure, API spec, dataset schema | Architecture doc |
| 3. Local platform setup | 1–2 | Install their webserver, ROS 2 env, sim with robot arm | Platform running, teleop verified |
| 4. Data collection pipeline | 2 | Data logger node, dataset format, collect 20–50 demos | Real dataset on disk |
| 5. Training pipeline | 2–3 | BC + ACT training, evaluation in sim | Trained policy + eval numbers |
| 6. Inference + deployment | 2 | Inference node, live execution in sim, web integration | Working live demo |
| 7. Polish + deliverable | 1–2 | Technical write-up, README, demo video, Git push | Submission package |

**Total: 10–12 days of focused work.**

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Their platform docs are outdated (CEO said so) — install pain | High | Allocate Day 3 buffer; contact CEO if blocked > 0.5 day |
| Simulation setup with their robot abstraction is tricky | Medium | Fall back to a generic Gazebo + UR5e setup decoupled from their platform |
| Training data quality from keyboard teleop is poor | Medium | Use scripted demonstrations as augmentation; mention this as future work (VR teleop) |
| ACT/Diffusion training doesn't converge on small dataset | Medium | Start with BC baseline; document negative results honestly |
| 12 days isn't enough for full prototype | Low–medium | Ship architecture + partial implementation rather than rush |

**Bigger picture:** Even if the prototype isn't perfect, the technical document + architecture + node structure + dataset schema + Git repo of clean partial code is already a strong submission. The CEO said "no strict requirement regarding format" — depth of thinking matters more than feature completeness.

---

## 9. What Goes on the Lab PC

| Task | Lab PC | Current machine |
|---|---|---|
| Reading docs, design thinking | OK | ✅ |
| Architecture diagrams (drawio, mermaid) | OK | ✅ |
| Writing technical document | OK | ✅ |
| Installing their webserver platform | Lab PC | ❌ |
| ROS 2 simulation | Lab PC | ❌ |
| GPU training | Lab PC | ❌ |
| Code editing | Either | ✅ (sync via Git) |

So: phases 1, 2, and part of 7 can be done anywhere. Phases 3–6 need the lab PC.

---

## 10. First Things to Do on Lab PC

1. `git init` a clean repository for this evaluation
2. Read https://docs.mybotshop.de/projects/product_robot_webserver/html/index.html cover-to-cover, take notes
3. Identify ROS 2 version their platform uses
4. Install their platform locally following docs (note where docs fail)
5. Set up a parallel Gazebo + Franka/UR5e environment as fallback
6. Email CEO with one clarifying question (see Section 11)

---

## 11. Clarifying Question to Send (Optional, Smart Move)

A short, well-placed email to the CEO before starting full work signals seriousness:

> Sehr geehrter Herr Kottlarz,
>
> vielen Dank für die schnelle Rückmeldung und das spannende technische Konzeptthema. Bevor ich mit der Ausarbeitung beginne, eine kurze Klärung:
>
> Gibt es einen bevorzugten Roboter / Manipulator (z. B. UR5e, Franka, eigene Plattform), an dem ich den Use Case orientieren soll, oder ist die Wahl frei, solange das Konzept hardware-agnostisch bleibt?
>
> Mit freundlichen Grüßen,
> Naeem Zain Uddin

Send after defense (May 18), start work May 19/20.

---

## 12. Final Submission Package Checklist

- [ ] Technical document (PDF, 10–15 pages)
- [ ] Architecture diagrams (system, ROS 2 nodes, data flow)
- [ ] Git repository (public or shared link)
- [ ] README with setup instructions
- [ ] Dataset schema specification
- [ ] API specification (OpenAPI / markdown)
- [ ] Demo video (2–3 min, screen recording)
- [ ] One-page executive summary
- [ ] Optional: live link to webserver running locally for a demo session
