"""
PyBullet Franka Panda ROS 2 bridge node.

Acts as a stand-in for the real robot platform on hosts without Gazebo or
hardware. Loads a Franka Panda URDF in PyBullet, steps the simulation at
240 Hz, and exposes ROS 2 topics that the rest of the pipeline already
expects:

    /joint_states       sensor_msgs/JointState      (published)
    /cartesian_pose     geometry_msgs/PoseStamped   (published)
    /cmd_robot          geometry_msgs/Twist         (subscribed — EE delta)
    /teleop_cmd         geometry_msgs/Twist         (subscribed — same shape)

The /cmd_robot subscription accepts EE delta commands; the bridge resolves
them to joint target positions via inverse kinematics. /teleop_cmd has the
same shape and is treated identically, so the data logger and inference node
can use the same topic regardless of whether the human or the policy is
driving.

This node is the local equivalent of MyBotShop's robot controller. On the
lab PC, swap it out for their actual platform — no other code needs to
change.
"""

from __future__ import annotations

import time
from pathlib import Path

import pybullet as p
import pybullet_data
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Twist


# Franka Panda has 9 movable joints, of which the first 7 are revolute arm
# joints and 9, 10 are gripper fingers. Joint 7 is a fixed link.
ARM_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
ARM_JOINT_NAMES = [f"panda_joint{i+1}" for i in range(7)]
EE_LINK_INDEX = 11  # panda_grasptarget_hand


class PyBulletRobotNode(Node):
    """ROS 2 wrapper around a PyBullet Franka Panda simulation."""

    def __init__(self) -> None:
        super().__init__("pybullet_robot_node")

        self.declare_parameter("sim_step_hz", 240.0)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("gui", False)
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("position_gain", 0.5)
        self.declare_parameter("velocity_gain", 1.0)
        self.declare_parameter("max_velocity", 1.0)

        self.sim_step_dt = 1.0 / self.get_parameter("sim_step_hz").value
        self.publish_dt = 1.0 / self.get_parameter("publish_hz").value

        # Connect to PyBullet
        mode = p.GUI if self.get_parameter("gui").value else p.DIRECT
        self._pb = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_step_dt)

        # Ground plane and robot
        p.loadURDF("plane.urdf")
        urdf_path = self.get_parameter("urdf_path").value or "franka_panda/panda.urdf"
        self._robot = p.loadURDF(urdf_path, useFixedBase=True)

        # Start at a reasonable home configuration
        home_q = [0.0, -0.5, 0.0, -1.8, 0.0, 1.6, 0.7]
        for idx, q in zip(ARM_JOINT_INDICES, home_q):
            p.resetJointState(self._robot, idx, q)
        self._target_q = list(home_q)

        # ROS 2 wiring
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_joints = self.create_publisher(JointState, "/joint_states", 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/cartesian_pose", 10)

        self.create_subscription(Twist, "/cmd_robot", self._on_cmd, sensor_qos)
        self.create_subscription(Twist, "/teleop_cmd", self._on_cmd, sensor_qos)

        # Physics step timer (240 Hz)
        self._step_timer = self.create_timer(self.sim_step_dt, self._step)
        # Publication timer (30 Hz)
        self._pub_timer = self.create_timer(self.publish_dt, self._publish)

        self.get_logger().info(
            f"pybullet_robot_node ready. urdf={urdf_path} "
            f"sim_hz={1.0/self.sim_step_dt:.0f} pub_hz={1.0/self.publish_dt:.0f}"
        )

    # ── ROS callbacks ─────────────────────────────────────────────────────

    def _on_cmd(self, msg: Twist) -> None:
        """
        Interpret an EE Twist command as a small step in end-effector space.

        Strategy: compute target EE pose by adding a small linear delta to the
        current EE position, then solve IK to a target joint configuration that
        the position controller drives toward. This is intentionally simple —
        it gives the policy a hardware-agnostic action surface (Twist) without
        baking in a specific IK library.
        """
        step = 0.01  # metres per command, scales with rate
        ee_state = p.getLinkState(self._robot, EE_LINK_INDEX)
        ee_xyz = list(ee_state[0])
        ee_xyz[0] += step * msg.linear.x
        ee_xyz[1] += step * msg.linear.y
        ee_xyz[2] += step * msg.linear.z
        ee_orn = ee_state[1]
        ik = p.calculateInverseKinematics(
            self._robot, EE_LINK_INDEX, ee_xyz, ee_orn,
            maxNumIterations=20,
            residualThreshold=1e-3,
        )
        self._target_q = list(ik[:7])

    # ── Timers ────────────────────────────────────────────────────────────

    def _step(self) -> None:
        # Drive joints toward target with PD position control
        p.setJointMotorControlArray(
            self._robot,
            ARM_JOINT_INDICES,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self._target_q,
            positionGains=[self.get_parameter("position_gain").value] * 7,
            velocityGains=[self.get_parameter("velocity_gain").value] * 7,
            forces=[200.0] * 7,
        )
        p.stepSimulation()

    def _publish(self) -> None:
        # Joint state
        states = p.getJointStates(self._robot, ARM_JOINT_INDICES)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(ARM_JOINT_NAMES)
        js.position = [s[0] for s in states]
        js.velocity = [s[1] for s in states]
        js.effort = [s[3] for s in states]
        self._pub_joints.publish(js)

        # Cartesian EE pose
        ee_state = p.getLinkState(self._robot, EE_LINK_INDEX)
        pose = PoseStamped()
        pose.header.stamp = js.header.stamp
        pose.header.frame_id = "panda_link0"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = ee_state[0]
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = ee_state[1]
        self._pub_pose.publish(pose)

    def destroy_node(self) -> bool:
        try:
            p.disconnect(self._pb)
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PyBulletRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
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
