"""
Inference node.

Loads a trained IL policy (BC, ACT, or Diffusion Policy) and runs it in
closed loop: subscribes to robot state, runs the policy at a configured rate,
and publishes action commands on the same topic the existing teleop stream
already uses, so the policy is a drop-in replacement for the human.

Execution modes for ACT:
    "first_action"        — execute only action[0] each step, re-predict
    "full_chunk"          — execute whole chunk, then re-predict
    "temporal_ensemble"   — weighted blend of overlapping chunks (ACT default)
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger
from il_pipeline_msgs.srv import LoadPolicy

# Project-local
from il_pipeline.inference.policy_loader import load_policy
from il_pipeline.inference.normaliser import Normaliser


class InferenceNode(Node):
    """Closed-loop policy inference."""

    def __init__(self) -> None:
        super().__init__("inference_node")

        # Parameters
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("policy_type", "act")  # "bc" | "act" | "diffusion"
        self.declare_parameter("inference_rate_hz", 30.0)
        self.declare_parameter("execution_mode", "temporal_ensemble")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cartesian_pose_topic", "/cartesian_pose")
        self.declare_parameter("cmd_topic", "/cmd_robot")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("enable_camera", False)
        self.declare_parameter("device", "cuda:0")

        self._policy = None
        self._normaliser: Optional[Normaliser] = None
        self._device = torch.device(
            self.get_parameter("device").get_parameter_value().string_value
        )
        self._inference_rate = (
            self.get_parameter("inference_rate_hz").get_parameter_value().double_value
        )
        self._execution_mode = (
            self.get_parameter("execution_mode").get_parameter_value().string_value
        )
        self._running = False

        # Latest sensor values
        self._latest_joint_state: Optional[JointState] = None
        self._latest_pose: Optional[PoseStamped] = None
        self._latest_image: Optional[Image] = None

        # Action chunk buffer for ACT temporal ensembling
        self._chunk_buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=50)
        self._step_index = 0

        # Wiring
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            sensor_qos,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("cartesian_pose_topic").value,
            self._on_pose,
            sensor_qos,
        )
        if self.get_parameter("enable_camera").value:
            self.create_subscription(
                Image,
                self.get_parameter("image_topic").value,
                self._on_image,
                sensor_qos,
            )

        self._pub_cmd = self.create_publisher(
            Twist,
            self.get_parameter("cmd_topic").value,
            10,
        )

        # Services
        self._srv_load = self.create_service(
            LoadPolicy, "~/load_policy", self._handle_load_policy
        )
        self._srv_start = self.create_service(
            Trigger, "~/start", self._handle_start
        )
        self._srv_stop = self.create_service(
            Trigger, "~/stop", self._handle_stop
        )

        self._timer = self.create_timer(1.0 / self._inference_rate, self._tick)

        # Auto-load if path was passed in at startup
        path = self.get_parameter("checkpoint_path").value
        if path:
            self._load(path)

        self.get_logger().info(
            f"inference_node ready. rate={self._inference_rate}Hz "
            f"mode={self._execution_mode} device={self._device}"
        )

    # ── Subscribers ───────────────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest_pose = msg

    def _on_image(self, msg: Image) -> None:
        self._latest_image = msg

    # ── Inference tick ────────────────────────────────────────────────────

    def _tick(self) -> None:
        if not self._running or self._policy is None:
            return
        if self._latest_joint_state is None:
            return

        t0 = time.time()
        observation = self._build_observation()
        if observation is None:
            return

        with torch.inference_mode():
            obs_tensor = self._normaliser.normalise_obs(observation).to(self._device)
            action_chunk = self._policy.predict_action_chunk(obs_tensor)
            action_chunk = self._normaliser.denormalise_action(action_chunk).cpu().numpy()

        action = self._select_action(action_chunk)
        self._publish_action(action)

        latency_ms = (time.time() - t0) * 1000.0
        if latency_ms > 50:
            self.get_logger().warn(f"inference latency {latency_ms:.1f}ms > 50ms")
        self._step_index += 1

    def _build_observation(self) -> Optional[dict]:
        js = self._latest_joint_state
        pose = self._latest_pose
        if js is None:
            return None

        joint_pos = np.array(js.position, dtype=np.float32)
        joint_vel = (
            np.array(js.velocity, dtype=np.float32)
            if js.velocity else np.zeros_like(joint_pos)
        )
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

        obs = {"observation.state": state}
        if self._latest_image is not None:
            obs["observation.images.wrist_cam"] = self._encode_image(self._latest_image)
        return obs

    @staticmethod
    def _encode_image(msg: Image) -> np.ndarray:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def _select_action(self, action_chunk: np.ndarray) -> np.ndarray:
        """
        Pick which action(s) from the predicted chunk to execute this step.

        For ACT, the chunk has shape [chunk_size, action_dim]. Different modes
        trade reactivity against re-prediction cost.
        """
        if self._execution_mode == "first_action":
            return action_chunk[0]

        if self._execution_mode == "full_chunk":
            # Step through the buffered chunk; only re-predict when exhausted.
            if not self._chunk_buffer or self._chunk_buffer[0][0] != self._step_index:
                self._chunk_buffer.clear()
                for i, a in enumerate(action_chunk):
                    self._chunk_buffer.append((self._step_index + i, a))
            _, action = self._chunk_buffer.popleft()
            return action

        # temporal_ensemble: weighted average over all chunks that include this step.
        self._chunk_buffer.append((self._step_index, action_chunk))
        actions = []
        weights = []
        for chunk_step, chunk in list(self._chunk_buffer):
            offset = self._step_index - chunk_step
            if 0 <= offset < len(chunk):
                actions.append(chunk[offset])
                # Exponential weighting: more recent predictions get higher weight.
                weights.append(np.exp(-0.01 * offset))
        actions = np.array(actions)
        weights = np.array(weights) / np.sum(weights)
        return np.sum(actions * weights[:, None], axis=0)

    def _publish_action(self, action: np.ndarray) -> None:
        # For now, assumes a Twist-shaped action; for delta_joint actions a
        # JointJog publisher would replace this. Topic remains the same so
        # the existing controller picks it up unchanged.
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(action[0]), float(action[1]), float(action[2])
        msg.angular.x, msg.angular.y, msg.angular.z = float(action[3]), float(action[4]), float(action[5])
        self._pub_cmd.publish(msg)

    # ── Policy lifecycle ──────────────────────────────────────────────────

    def _load(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        policy_type = self.get_parameter("policy_type").value
        self._policy, self._normaliser = load_policy(path, policy_type, self._device)
        self.get_logger().info(f"loaded {policy_type} policy from {path}")

    def _handle_load_policy(
        self, request: LoadPolicy.Request, response: LoadPolicy.Response
    ):
        t0 = time.time()
        try:
            # Honour the request payload; fall back to parameter values where blank.
            ckpt_path = request.checkpoint_path or self.get_parameter("checkpoint_path").value
            policy_type = request.policy_type or self.get_parameter("policy_type").value

            # Override runtime rate and execution mode if provided.
            if request.inference_rate_hz > 0:
                self._inference_rate = request.inference_rate_hz
                self._timer.cancel()
                self._timer = self.create_timer(1.0 / self._inference_rate, self._tick)
            if request.execution_mode:
                self._execution_mode = request.execution_mode

            # Stash policy_type as a parameter so _load picks it up.
            self.set_parameters([
                rclpy.parameter.Parameter("policy_type", value=policy_type),
                rclpy.parameter.Parameter("checkpoint_path", value=ckpt_path),
            ])
            self._load(ckpt_path)

            response.success = True
            response.warm_up_ms = (time.time() - t0) * 1000.0
            response.message = f"loaded {policy_type} from {ckpt_path}"
        except Exception as e:  # noqa: BLE001
            response.success = False
            response.warm_up_ms = (time.time() - t0) * 1000.0
            response.message = str(e)
        return response

    def _handle_start(self, request, response):
        if self._policy is None:
            response.success = False
            response.message = "no policy loaded"
            return response
        self._running = True
        self._step_index = 0
        self._chunk_buffer.clear()
        response.success = True
        response.message = "running"
        return response

    def _handle_stop(self, request, response):
        self._running = False
        response.success = True
        response.message = "stopped"
        return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
