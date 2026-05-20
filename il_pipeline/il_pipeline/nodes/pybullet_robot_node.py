"""
PyBullet Franka Panda ROS 2 bridge node.

Acts as a stand-in for the real robot platform on hosts without Gazebo or
hardware. Loads a Franka Panda URDF in PyBullet, steps the simulation at
240 Hz, and exposes ROS 2 topics that the rest of the pipeline already
expects:

    /joint_states       sensor_msgs/JointState      (published)
    /cartesian_pose     geometry_msgs/PoseStamped   (published)
    /cube_pose          geometry_msgs/PoseStamped   (published — task object)
    /task_status        std_msgs/Bool               (published — success flag)
    /cmd_robot          geometry_msgs/Twist         (subscribed — EE delta)
    /teleop_cmd         geometry_msgs/Twist         (subscribed — same shape)
    ~/reset             std_srvs/Trigger            (service — resets task)

The /cmd_robot subscription accepts EE delta commands; the bridge resolves
them to joint target positions via inverse kinematics. /teleop_cmd has the
same shape and is treated identically, so the data logger and inference node
can use the same topic regardless of whether the human or the policy is
driving.

Task: pick-and-place. A cube spawns at a randomised position in front of the
robot; success is declared when the cube ends up within a target zone.

This node is the local equivalent of MyBotShop's robot controller. On the
lab PC, swap it out for their actual platform — no other code needs to
change because the topic + service contracts stay the same.
"""

from __future__ import annotations

import contextlib
import random

import pybullet as p
import pybullet_data
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

# Franka Panda has 12 joints in the URDF, of which the first 7 are the
# revolute arm joints. 9 and 10 are gripper fingers (prismatic).
ARM_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
ARM_JOINT_NAMES = [f"panda_joint{i+1}" for i in range(7)]
GRIPPER_JOINT_INDICES = [9, 10]
EE_LINK_INDEX = 11

# Task: pick-and-place. Cube spawns on the table at a randomised pose; the
# policy must grasp it and deliver it into the target zone.
CUBE_SIZE = 0.04
CUBE_SPAWN_X_RANGE = (0.40, 0.55)
CUBE_SPAWN_Y_RANGE = (-0.15, 0.05)
CUBE_SPAWN_Z = CUBE_SIZE / 2 + 1e-3
CUBE_MASS = 0.05

# Target zone: a fixed disc to the left of the spawn region so the policy
# always learns a left-ward delivery.
TARGET_XY = (0.40, 0.25)
TARGET_RADIUS = 0.08
TARGET_Z_MAX = 0.10           # cube must end below this (i.e. on the table)

HOME_Q = [0.0, -0.5, 0.0, -1.8, 0.0, 1.6, 0.7]


class PyBulletRobotNode(Node):
    """ROS 2 wrapper around a PyBullet pick-and-place Franka Panda simulation."""

    def __init__(self) -> None:
        super().__init__("pybullet_robot_node")

        self.declare_parameter("sim_step_hz", 240.0)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("gui", False)
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("position_gain", 0.5)
        self.declare_parameter("velocity_gain", 1.0)
        self.declare_parameter("max_velocity", 1.0)
        self.declare_parameter("ee_step_scale", 0.01)
        self.declare_parameter("gripper_open", True)
        self.declare_parameter("seed", 0)
        # Cube spawn range. Defaults match the training distribution; an
        # OOD eval can override these to spawn outside the training range.
        self.declare_parameter("cube_spawn_x_min", CUBE_SPAWN_X_RANGE[0])
        self.declare_parameter("cube_spawn_x_max", CUBE_SPAWN_X_RANGE[1])
        self.declare_parameter("cube_spawn_y_min", CUBE_SPAWN_Y_RANGE[0])
        self.declare_parameter("cube_spawn_y_max", CUBE_SPAWN_Y_RANGE[1])

        self.sim_step_dt = 1.0 / self.get_parameter("sim_step_hz").value
        self.publish_dt = 1.0 / self.get_parameter("publish_hz").value
        self._ee_step = self.get_parameter("ee_step_scale").value
        self._rng = random.Random(self.get_parameter("seed").value)

        # Connect to PyBullet
        mode = p.GUI if self.get_parameter("gui").value else p.DIRECT
        self._pb = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_step_dt)

        # Camera + GUI setup. PyBullet's default camera is too far out and
        # opens "Synthetic Camera RGB/Depth/Seg" debug panels that take a
        # quarter of the screen — useless for a demo recording. Tighten the
        # view onto the workspace and hide the panels.
        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.3,
                cameraYaw=50,
                cameraPitch=-30,
                cameraTargetPosition=[0.45, 0.05, 0.15],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Ground plane and robot
        p.loadURDF("plane.urdf")
        urdf_path = self.get_parameter("urdf_path").value or "franka_panda/panda.urdf"
        self._robot = p.loadURDF(urdf_path, useFixedBase=True)

        # Initialise the robot at home
        for idx, q in zip(ARM_JOINT_INDICES, HOME_Q, strict=False):
            p.resetJointState(self._robot, idx, q)
        self._target_q = list(HOME_Q)
        self._gripper_target = 0.04 if self.get_parameter("gripper_open").value else 0.0
        # Track grasp state and an optional cube-to-gripper constraint that
        # gives reliable pickup in simulation (standard IL-sim trick used by
        # Robomimic, ALOHA, etc.).
        self._grasp_constraint = None
        self._grasp_attached = False

        # Spawn the task object (cube). Stored on self so we can reset it.
        self._cube_id = None
        self._spawn_cube()

        # Visual marker for the target zone — purely cosmetic, helps with GUI demos
        target_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=TARGET_RADIUS,
            length=0.002,
            rgbaColor=[0.0, 1.0, 0.0, 0.35],
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_vis,
            basePosition=[TARGET_XY[0], TARGET_XY[1], 0.001],
        )

        # ROS 2 wiring
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_joints = self.create_publisher(JointState, "/joint_states", 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/cartesian_pose", 10)
        self._pub_cube = self.create_publisher(PoseStamped, "/cube_pose", 10)
        self._pub_status = self.create_publisher(Bool, "/task_status", 10)

        self.create_subscription(Twist, "/cmd_robot", self._on_cmd, sensor_qos)
        self.create_subscription(Twist, "/teleop_cmd", self._on_cmd, sensor_qos)

        # Service to reset the task between episodes
        self.create_service(Trigger, "~/reset", self._handle_reset)

        # Physics step timer (240 Hz) and publication timer (30 Hz)
        self._step_timer = self.create_timer(self.sim_step_dt, self._step)
        self._pub_timer = self.create_timer(self.publish_dt, self._publish)

        self.get_logger().info(
            f"pybullet_robot_node ready. urdf={urdf_path} "
            f"sim_hz={1.0/self.sim_step_dt:.0f} pub_hz={1.0/self.publish_dt:.0f}"
        )

    # ── Task helpers ──────────────────────────────────────────────────────

    def _spawn_cube(self) -> None:
        """Spawn (or respawn) the task cube on the table at a randomised pose."""
        if self._cube_id is not None:
            p.removeBody(self._cube_id)
        x = self._rng.uniform(
            self.get_parameter("cube_spawn_x_min").value,
            self.get_parameter("cube_spawn_x_max").value,
        )
        y = self._rng.uniform(
            self.get_parameter("cube_spawn_y_min").value,
            self.get_parameter("cube_spawn_y_max").value,
        )
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_SIZE / 2] * 3)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_SIZE / 2] * 3,
            rgbaColor=[1.0, 0.2, 0.2, 1.0],
        )
        self._cube_id = p.createMultiBody(
            baseMass=CUBE_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, CUBE_SPAWN_Z],
        )
        # Bump friction up so the gripper can hold the cube reliably on CPU
        # without us having to fine-tune the entire grasp controller.
        p.changeDynamics(self._cube_id, -1, lateralFriction=1.0, spinningFriction=0.1)

    def _task_success(self) -> bool:
        """Cube within target xy disc and on the table (low z)."""
        if self._cube_id is None:
            return False
        pos, _ = p.getBasePositionAndOrientation(self._cube_id)
        dx = pos[0] - TARGET_XY[0]
        dy = pos[1] - TARGET_XY[1]
        return (dx * dx + dy * dy) ** 0.5 < TARGET_RADIUS and pos[2] < TARGET_Z_MAX

    # ── ROS callbacks ─────────────────────────────────────────────────────

    def _on_cmd(self, msg: Twist) -> None:
        """
        Interpret an EE Twist command as a step in end-effector space.

        - linear.{x,y,z}: cartesian delta (metres, scaled by ee_step_scale)
        - angular.x:      gripper command (positive = open, negative = close)
                          A separate dedicated channel could be cleaner; using
                          angular.x keeps a single 6-DOF action space for the
                          policy.
        """
        ee_state = p.getLinkState(self._robot, EE_LINK_INDEX)
        ee_xyz = list(ee_state[0])
        ee_xyz[0] += self._ee_step * msg.linear.x
        ee_xyz[1] += self._ee_step * msg.linear.y
        ee_xyz[2] += self._ee_step * msg.linear.z
        ee_orn = ee_state[1]
        ik = p.calculateInverseKinematics(
            self._robot, EE_LINK_INDEX, ee_xyz, ee_orn,
            maxNumIterations=20,
            residualThreshold=1e-3,
        )
        self._target_q = list(ik[:7])

        # Gripper: msg.angular.x in [-1, 1]; positive opens, negative closes
        if msg.angular.x > 0.1:
            self._gripper_target = 0.04
            self._release_grasp()
        elif msg.angular.x < -0.1:
            self._gripper_target = 0.0
            self._maybe_attach_grasp()

    def _handle_reset(self, request, response):
        for idx, q in zip(ARM_JOINT_INDICES, HOME_Q, strict=False):
            p.resetJointState(self._robot, idx, q)
        self._target_q = list(HOME_Q)
        self._gripper_target = 0.04
        self._release_grasp()
        self._spawn_cube()
        response.success = True
        response.message = "robot and cube reset"
        return response

    # ── Grasp helper (constraint-based, standard IL-sim trick) ───────────

    def _maybe_attach_grasp(self) -> None:
        """If the gripper is closing and the cube is near the EE, attach it."""
        if self._grasp_attached or self._cube_id is None:
            return
        ee_pos = p.getLinkState(self._robot, EE_LINK_INDEX)[0]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self._cube_id)
        dx = ee_pos[0] - cube_pos[0]
        dy = ee_pos[1] - cube_pos[1]
        dz = ee_pos[2] - cube_pos[2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        # Threshold: gripper close enough that a real Franka would have grasped.
        # 8cm covers steady-state error and fingertip offset.
        if d > 0.08:
            return
        # Create a fixed constraint between the EE link and the cube. The
        # transform between them is the current relative pose, so the cube
        # stays where it is relative to the EE.
        ee_state = p.getLinkState(self._robot, EE_LINK_INDEX)
        ee_pos_w, ee_orn_w = ee_state[0], ee_state[1]
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos_w, ee_orn_w)
        rel_pos, rel_orn = p.multiplyTransforms(
            inv_ee_pos, inv_ee_orn, cube_pos, cube_orn
        )
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self._robot,
            parentLinkIndex=EE_LINK_INDEX,
            childBodyUniqueId=self._cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        self._grasp_attached = True
        self.get_logger().info(f"grasp attached (cube-EE distance was {d:.3f}m)")

    def _release_grasp(self) -> None:
        if self._grasp_attached and self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint)
            self._grasp_constraint = None
            self._grasp_attached = False
            self.get_logger().info("grasp released")

    # ── Timers ────────────────────────────────────────────────────────────

    def _step(self) -> None:
        # Drive arm joints toward target with PD position control. Forces are
        # set high enough that the manipulator actually reaches commanded
        # poses within reasonable steady-state error.
        p.setJointMotorControlArray(
            self._robot,
            ARM_JOINT_INDICES,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self._target_q,
            positionGains=[self.get_parameter("position_gain").value] * 7,
            velocityGains=[self.get_parameter("velocity_gain").value] * 7,
            forces=[500.0] * 7,
        )
        # Drive gripper fingers symmetrically.
        p.setJointMotorControlArray(
            self._robot,
            GRIPPER_JOINT_INDICES,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[self._gripper_target, self._gripper_target],
            forces=[40.0, 40.0],
        )
        p.stepSimulation()

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()

        # Joint state
        states = p.getJointStates(self._robot, ARM_JOINT_INDICES)
        js = JointState()
        js.header.stamp = stamp
        js.name = list(ARM_JOINT_NAMES)
        js.position = [s[0] for s in states]
        js.velocity = [s[1] for s in states]
        js.effort = [s[3] for s in states]
        self._pub_joints.publish(js)

        # Cartesian EE pose
        ee_state = p.getLinkState(self._robot, EE_LINK_INDEX)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "panda_link0"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = ee_state[0]
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = ee_state[1]
        self._pub_pose.publish(pose)

        # Cube pose
        if self._cube_id is not None:
            cube_pos, cube_orn = p.getBasePositionAndOrientation(self._cube_id)
            cube_msg = PoseStamped()
            cube_msg.header.stamp = stamp
            cube_msg.header.frame_id = "panda_link0"
            cube_msg.pose.position.x, cube_msg.pose.position.y, cube_msg.pose.position.z = cube_pos
            cube_msg.pose.orientation.x, cube_msg.pose.orientation.y, cube_msg.pose.orientation.z, cube_msg.pose.orientation.w = cube_orn
            self._pub_cube.publish(cube_msg)

        # Task status
        status = Bool()
        status.data = self._task_success()
        self._pub_status.publish(status)

    def destroy_node(self) -> bool:
        with contextlib.suppress(Exception):
            p.disconnect(self._pb)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PyBulletRobotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
