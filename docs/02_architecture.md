# Architecture

## System view

How the IL pipeline plugs into the existing MyBotShop webserver platform.

```mermaid
flowchart TB
    subgraph existing["MyBotShop Webserver (existing)"]
        UI["Web UI"]
        ROSBridge["ROS 2 / WebSocket Bridge"]
    end

    subgraph new["IL Pipeline"]
        WebAPI["FastAPI<br/>REST + WS"]
        Logger["data_logger_node"]
        Inferer["inference_node"]
        Storage[("LeRobotDataset<br/>parquet")]
        Ckpts[("Policy<br/>checkpoints")]
    end

    subgraph robot["Robot (sim or real)"]
        Controller["ROS 2 Controller"]
        Sensors["Joints / EE / Object pose"]
    end

    UI --> WebAPI
    UI --> ROSBridge
    WebAPI --> Logger
    WebAPI --> Inferer
    Logger --> Storage
    Storage --> Ckpts
    Ckpts --> Inferer
    Sensors --> Logger
    Sensors --> Inferer
    Inferer --> Controller
    ROSBridge --> Controller
```

The IL pipeline is a sibling subsystem: same ROS 2 interfaces, separate FastAPI surface that the existing UI calls, never modifies the platform's core code.

## ROS 2 nodes and topics

```mermaid
flowchart LR
    teleop["teleop_node<br/>(existing)"]
    robot["robot_controller<br/>(existing)"]
    logger["data_logger_node<br/>(new)"]
    inferer["inference_node<br/>(new)"]

    teleop -->|"/teleop_cmd"| logger
    teleop -->|"/teleop_cmd"| robot
    robot -->|"/joint_states"| logger
    robot -->|"/joint_states"| inferer
    robot -->|"/cartesian_pose"| logger
    robot -->|"/cartesian_pose"| inferer
    robot -->|"/cube_pose"| logger
    robot -->|"/cube_pose"| inferer
    inferer -->|"/cmd_robot"| robot

    logger -.->|writes| ds[("dataset/")]
    ds -.->|reads| trainer["train.py<br/>(offline)"]
    trainer -.->|writes| ck[("checkpoints/")]
    ck -.->|loads| inferer

    classDef new fill:#e1f5ff,stroke:#0066cc
    class logger,inferer,trainer new
```

Solid arrows are ROS 2 topics. Dashed arrows are filesystem reads/writes. The `inference_node` publishes to the same `/cmd_robot` topic the human teleoperator drives, so the deployed policy is a drop-in replacement for the teleop stream.

## Node summary

| Node | Subscribes | Publishes / Services |
|---|---|---|
| `data_logger_node` | `/joint_states`, `/cartesian_pose`, `/cube_pose`, `/teleop_cmd` | `~/start_episode` (`StartEpisode.srv`), `~/stop_episode` (`StopEpisode.srv`) |
| `inference_node` | `/joint_states`, `/cartesian_pose`, `/cube_pose` | `/cmd_robot` (`Twist`), `~/load_policy` (`LoadPolicy.srv`), `~/start` / `~/stop` (`Trigger`) |
| `pybullet_robot_node` (sim stand-in) | `/cmd_robot` | `/joint_states`, `/cartesian_pose`, `/cube_pose`, `/task_status`, `~/reset` |

The simulator and a real robot driver expose the same topic contract — switching from PyBullet to the MyBotShop platform on real hardware is a topic-remap, not a code change.
