# Architecture Diagrams

All diagrams use Mermaid. They render natively on GitHub, GitLab, VS Code, and Foxglove docs.

---

## 1. System-Level Architecture

How the proposed components plug into the existing MyBotShop webserver platform.

```mermaid
flowchart TB
    subgraph existing["MyBotShop Webserver Platform (existing)"]
        UI["Web UI<br/>Teleop, Monitoring, Diagnostics"]
        WS["WebSocket Bridge"]
        ROSBridge["ROS 2 Bridge<br/>Topic/Service/Action Bindings"]
        Viz["3D Visualisation"]
    end

    subgraph new["IL Pipeline (proposed)"]
        WebAPI["FastAPI Layer<br/>REST + WebSocket"]
        Logger["Data Logger Node"]
        Trainer["Training Service Node"]
        Inferer["Inference Node"]
        Storage[("LeRobot Dataset<br/>parquet shards")]
        Ckpts[("Policy Checkpoints")]
    end

    subgraph robot["Robot (sim or real)"]
        Controller["ROS 2 Controller"]
        Sensors["Joint States, EE Pose, Cameras"]
    end

    UI --> WS --> ROSBridge
    UI --> WebAPI
    WebAPI --> Logger
    WebAPI --> Trainer
    WebAPI --> Inferer
    Logger --> Storage
    Trainer --> Storage
    Trainer --> Ckpts
    Inferer --> Ckpts
    Sensors --> Logger
    Sensors --> Inferer
    ROSBridge --> Logger
    Inferer --> Controller
    ROSBridge --> Controller
```

The IL pipeline is a sibling subsystem to the existing platform: it speaks the same ROS 2 interfaces, exposes its own web layer that the existing UI can call, and never modifies the platform's core code.

---

## 2. ROS 2 Node Graph

The runtime topology of nodes, topics, and services.

```mermaid
flowchart LR
    teleop["teleop_node<br/>(existing)"]
    robot["robot_controller<br/>(existing)"]
    logger["data_logger_node<br/>(new)"]
    inferer["inference_node<br/>(new)"]
    trainer["training_service_node<br/>(new, off-line)"]

    teleop -->|"/teleop_cmd"| logger
    teleop -->|"/teleop_cmd"| robot
    robot -->|"/joint_states"| logger
    robot -->|"/joint_states"| inferer
    robot -->|"/cartesian_pose"| logger
    robot -->|"/cartesian_pose"| inferer

    cam["camera_driver"] -->|"/camera/image_raw"| logger
    cam -->|"/camera/image_raw"| inferer

    inferer -->|"/cmd_robot"| robot

    logger -.->|writes| ds[("dataset/")]
    ds -.->|reads| trainer
    trainer -.->|writes| ck[("checkpoints/")]
    ck -.->|loads| inferer

    classDef new fill:#e1f5ff,stroke:#0066cc
    classDef existing fill:#f0f0f0,stroke:#666
    class logger,inferer,trainer new
    class teleop,robot,cam existing
```

Solid arrows are ROS 2 topics; dashed arrows are filesystem reads/writes.

---

## 3. Data Collection Flow

What happens when a user records a demonstration through the webserver UI.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI
    participant Logger as data_logger_node
    participant Teleop as teleop_node
    participant Robot as Robot

    User->>UI: click "Start Recording"
    UI->>API: POST /datasets/record/start
    API->>Logger: srv: StartEpisode
    Logger-->>API: episode_id

    loop while user is teleoperating
        User->>UI: joystick / keyboard input
        UI->>Teleop: command
        Teleop->>Robot: /teleop_cmd
        Robot->>Logger: /joint_states, /cartesian_pose
        Teleop->>Logger: /teleop_cmd (recorded as action)
    end

    User->>UI: click "Stop"
    UI->>API: POST /datasets/record/stop
    API->>Logger: srv: StopEpisode
    Logger->>Logger: append frames to parquet shard
    Logger-->>API: frame_count, duration
    API-->>UI: episode summary
```

---

## 4. Training Pipeline

How a logged dataset becomes a deployed policy.

```mermaid
flowchart TB
    ds[("LeRobotDataset<br/>parquet shards")]
    norm["Normaliser<br/>fit on dataset"]
    dl["PyTorch DataLoader<br/>action chunking, shuffle"]
    policy["ACT Policy<br/>Transformer encoder<br/>+ action chunk decoder"]
    loss["Loss<br/>L1 on action chunks<br/>+ KL (if VAE)"]
    opt["AdamW + Cosine LR"]
    ckpt[("Checkpoint<br/>per N steps")]
    eval["Sim Evaluation<br/>success rate"]
    best[("Best Checkpoint")]

    ds --> norm
    norm --> dl
    dl --> policy
    policy --> loss
    loss --> opt
    opt --> policy
    policy --> ckpt
    ckpt --> eval
    eval -->|select| best
```

---

## 5. Inference Flow

What happens when a trained policy is deployed.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as FastAPI
    participant Inferer as inference_node
    participant Robot as Robot

    User->>UI: select policy, click "Deploy"
    UI->>API: POST /policies/{id}/deploy
    API->>Inferer: srv: LoadPolicy(ckpt_path, "act", 30Hz)
    Inferer->>Inferer: torch.load + warm-up
    Inferer-->>API: ready

    User->>UI: click "Start"
    UI->>API: POST /policies/{id}/start
    API->>Inferer: srv: StartInference

    loop at inference_rate_hz
        Robot->>Inferer: /joint_states, /cartesian_pose, /image_raw
        Inferer->>Inferer: build observation tensor
        Inferer->>Inferer: policy.forward()
        Inferer->>Robot: /cmd_robot (action chunk head)
    end

    User->>UI: click "Stop"
    UI->>API: POST /policies/{id}/stop
    API->>Inferer: srv: StopInference
```

ACT predicts chunks of `k` future actions per forward pass; the inference node executes the first action and re-predicts each cycle, or executes the full chunk with temporal ensembling — both modes are supported.

---

## 6. Optional RL Fine-Tuning Loop

How the IL policy is extended into an RL fine-tuning stage. This component carries over my thesis work on PPO directly.

```mermaid
flowchart LR
    il["Trained IL Policy<br/>(ACT or BC)"]
    bridge["policy_bridge<br/>predict() & act()"]
    env["Gym-style Env<br/>wraps simulation"]
    ppo["PPO (SB3)<br/>actor-critic"]
    reward["Reward Function<br/>sparse or shaped"]
    final["Fine-tuned Policy"]

    il -->|init weights| bridge
    bridge --> ppo
    env --> bridge
    bridge --> env
    env --> reward
    reward --> ppo
    ppo -->|updates| bridge
    ppo --> final
```

The `policy_bridge` abstraction keeps IL and RL training decoupled from the deployment node — the same trained network can be loaded by `inference_node` regardless of which stage produced it.
