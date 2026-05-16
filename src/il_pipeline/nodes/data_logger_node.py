"""
Data logger node.

Subscribes to robot state, teleop commands, and (optionally) cameras during an
episode. Buffers frames in memory until the episode is stopped, then writes a
parquet shard following the LeRobotDataset schema described in
docs/04_dataset_schema.md.

Episode lifecycle is driven by service calls from the FastAPI layer so the
webserver UI can start and stop recordings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Twist
from il_pipeline_msgs.srv import StartEpisode, StopEpisode

from il_pipeline.dataset.lerobot_writer import LeRobotShardWriter
from il_pipeline.dataset.frame_validator import FrameValidator


@dataclass
class EpisodeBuffer:
    """In-memory buffer for the active episode."""

    episode_id: str
    task_description: str
    started_at: float
    frames: list[dict] = field(default_factory=list)

    def append(self, frame: dict) -> None:
        self.frames.append(frame)

    @property
    def duration_s(self) -> float:
        return time.time() - self.started_at


class DataLoggerNode(Node):
    """
    Records demonstration episodes.

    Subscriptions (configured by parameters):
        /joint_states       sensor_msgs/JointState
        /cartesian_pose     geometry_msgs/PoseStamped   (optional)
        /teleop_cmd         geometry_msgs/Twist         (action source)
        /camera/image_raw   sensor_msgs/Image           (optional)

    Services:
        ~/start_episode     std_srvs/Trigger    (TODO: use StartEpisode.srv with payload)
        ~/stop_episode      std_srvs/Trigger
    """

    def __init__(self) -> None:
        super().__init__("data_logger_node")

        # Declare parameters — all topic names and dataset paths are config-driven
        # so the same node works for Panda, UR5e, or a custom arm.
        self.declare_parameter("dataset_root", "/data/lerobot_datasets")
        self.declare_parameter("dataset_name", "default_dataset")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cartesian_pose_topic", "/cartesian_pose")
        self.declare_parameter("teleop_cmd_topic", "/teleop_cmd")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("enable_camera", False)
        self.declare_parameter("expected_fps", 30.0)
        self.declare_parameter("expected_n_joints", 7)
        self.declare_parameter("action_dim", 7)
        self.declare_parameter("action_mode", "delta_joint")  # or "delta_ee"

        self.dataset_root = Path(
            self.get_parameter("dataset_root").get_parameter_value().string_value
        )
        self.dataset_name = (
            self.get_parameter("dataset_name").get_parameter_value().string_value
        )
        self.expected_fps = (
            self.get_parameter("expected_fps").get_parameter_value().double_value
        )
        self.enable_camera = (
            self.get_parameter("enable_camera").get_parameter_value().bool_value
        )

        # Latest values, sampled at action publication rate to form a frame
        self._latest_joint_state: Optional[JointState] = None
        self._latest_pose: Optional[PoseStamped] = None
        self._latest_image: Optional[Image] = None
        self._latest_action: Optional[np.ndarray] = None
        self._latest_action_t: float = 0.0

        self._episode: Optional[EpisodeBuffer] = None
        self._writer = LeRobotShardWriter(
            root=self.dataset_root,
            dataset_name=self.dataset_name,
        )
        self._validator = FrameValidator(
            expected_n_joints=self.get_parameter("expected_n_joints").value,
            expected_action_dim=self.get_parameter("action_dim").value,
            expected_fps=self.expected_fps,
        )

        # ROS 2 wiring. QoS chosen for reliable but bounded queue depth.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub_joint = self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            sensor_qos,
        )
        self._sub_pose = self.create_subscription(
            PoseStamped,
            self.get_parameter("cartesian_pose_topic").value,
            self._on_pose,
            sensor_qos,
        )
        self._sub_teleop = self.create_subscription(
            Twist,
            self.get_parameter("teleop_cmd_topic").value,
            self._on_teleop,
            sensor_qos,
        )
        if self.enable_camera:
            self._sub_image = self.create_subscription(
                Image,
                self.get_parameter("image_topic").value,
                self._on_image,
                sensor_qos,
            )

        # Timer pulls the latest values into frames at the expected fps.
        # Using a timer rather than callback-chained sampling keeps frame
        # cadence consistent even if upstream topics jitter.
        self._frame_timer = self.create_timer(
            1.0 / self.expected_fps,
            self._tick,
        )

        self._srv_start = self.create_service(
            StartEpisode, "~/start_episode", self._handle_start
        )
        self._srv_stop = self.create_service(
            StopEpisode, "~/stop_episode", self._handle_stop
        )

        self.get_logger().info(
            f"data_logger_node ready. dataset={self.dataset_name} "
            f"fps={self.expected_fps} camera={self.enable_camera}"
        )

    # ── Subscription callbacks ────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest_pose = msg

    def _on_teleop(self, msg: Twist) -> None:
        # Convert Twist into the configured action representation.
        # For delta_ee, take linear and angular components directly.
        # For delta_joint, an upstream IK layer publishes the joint deltas
        # we should subscribe to instead — see project_TODO.md.
        self._latest_action = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z,
             msg.angular.x, msg.angular.y, msg.angular.z, 0.0],
            dtype=np.float32,
        )
        self._latest_action_t = time.time()

    def _on_image(self, msg: Image) -> None:
        self._latest_image = msg

    # ── Frame assembly ────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._episode is None:
            return
        if self._latest_joint_state is None or self._latest_action is None:
            # Skip frames until both streams have produced at least one message.
            return

        frame = self._build_frame()
        ok, reason = self._validator.validate(frame)
        if not ok:
            self.get_logger().warn(f"dropping frame: {reason}")
            return

        self._episode.append(frame)

    def _build_frame(self) -> dict:
        js = self._latest_joint_state
        pose = self._latest_pose

        # Concatenate joint pos, joint vel, EE pose into observation.state.
        # Order is documented in docs/04_dataset_schema.md.
        joint_pos = np.array(js.position, dtype=np.float32)
        joint_vel = np.array(js.velocity, dtype=np.float32) if js.velocity else np.zeros_like(joint_pos)
        if pose is not None:
            ee = np.array(
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                 pose.pose.orientation.x, pose.pose.orientation.y,
                 pose.pose.orientation.z, pose.pose.orientation.w],
                dtype=np.float32,
            )
        else:
            ee = np.zeros(7, dtype=np.float32)

        state = np.concatenate([joint_pos, joint_vel, ee])

        frame = {
            "observation.state": state,
            "action": self._latest_action.copy(),
            "timestamp": time.time() - self._episode.started_at,
            "frame_index": len(self._episode.frames),
            "episode_index": self._writer.next_episode_index(),
            "next.reward": 0.0,
            "next.done": False,
        }
        if self.enable_camera and self._latest_image is not None:
            frame["observation.images.wrist_cam"] = self._encode_image(self._latest_image)
        return frame

    @staticmethod
    def _encode_image(msg: Image) -> np.ndarray:
        # Lab-PC: use cv_bridge here. Stubbed for local development.
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    # ── Service handlers ──────────────────────────────────────────────────

    def _handle_start(self, request: StartEpisode.Request, response: StartEpisode.Response):
        if self._episode is not None:
            response.success = False
            response.episode_id = ""
            response.message = "an episode is already in progress; stop it first"
            return response

        episode_id = request.episode_name or f"ep-{int(time.time())}"
        self._episode = EpisodeBuffer(
            episode_id=episode_id,
            task_description=request.task_description,
            started_at=time.time(),
        )
        self.get_logger().info(
            f"started episode {episode_id} (task='{request.task_description}')"
        )
        response.success = True
        response.episode_id = episode_id
        response.message = ""
        return response

    def _handle_stop(self, request: StopEpisode.Request, response: StopEpisode.Response):
        if self._episode is None:
            response.success = False
            response.episode_id = ""
            response.frame_count = 0
            response.duration_s = 0.0
            response.saved_to = ""
            response.message = "no active episode"
            return response

        ep = self._episode
        self._episode = None

        if request.outcome == "discard":
            self.get_logger().info(f"discarded episode {ep.episode_id} ({len(ep.frames)} frames)")
            response.success = True
            response.episode_id = ep.episode_id
            response.frame_count = len(ep.frames)
            response.duration_s = ep.duration_s
            response.saved_to = ""
            response.message = "discarded"
            return response

        # Mark the final frame's done flag.
        if ep.frames:
            ep.frames[-1]["next.done"] = True

        path = self._writer.write_episode(ep.episode_id, ep.frames)
        msg = (
            f"saved episode {ep.episode_id} "
            f"frames={len(ep.frames)} duration={ep.duration_s:.1f}s -> {path}"
        )
        self.get_logger().info(msg)
        response.success = True
        response.episode_id = ep.episode_id
        response.frame_count = len(ep.frames)
        response.duration_s = ep.duration_s
        response.saved_to = str(path)
        response.message = msg
        return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
