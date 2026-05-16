"""
Scripted pick-and-place expert.

Subscribes to /cube_pose and /cartesian_pose, publishes /teleop_cmd Twist
commands that step the end-effector through a phase-based pick-and-place
trajectory:

    APPROACH    move EE to (cube_xy, hover_z) with gripper open
    DESCEND     lower EE to grasp_z over the cube
    GRASP       close gripper (Twist.angular.x = -1) and hold
    LIFT        raise EE back to hover_z while gripped
    TRANSPORT   move EE to (target_xy, hover_z)
    DELIVER     descend EE to place_z
    RELEASE     open gripper (Twist.angular.x = +1) and hold
    RETREAT     raise EE back to hover_z and stop

This script plays the role of a human teleoperator during data collection.
The data logger records both /joint_states and /teleop_cmd, so the learned
policy sees exactly the demonstrations a real human would have produced
through MyBotShop's teleop UI.

The phase machine uses time-based transitions tuned for the 30 Hz command
rate and the PyBullet position controller gains. On a real robot, the same
state machine works but the timings would need tuning.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


# Target zone — must match pybullet_robot_node.py
TARGET_XY = (0.40, 0.25)
HOVER_Z = 0.25           # safe transit height (above cube + clearance)
GRASP_Z_OFFSET = 0.02    # offset above cube centre when grasping
                         # Panda EE link is panda_grasptarget_hand which sits
                         # roughly at the gripper-finger midline; +2cm above
                         # the 4cm cube centre puts the fingers around it.
PLACE_Z = 0.05           # release height over target (low so cube settles)

# Phase durations (seconds at 30 Hz). Tuned for the PyBullet position controller
# with gains 0.5; values are conservative so each phase actually completes
# before advancing.
PHASE_DURATION = {
    "APPROACH":  2.5,
    "DESCEND":   2.0,
    "GRASP":     1.5,
    "LIFT":      2.0,
    "TRANSPORT": 2.5,
    "DELIVER":   2.0,
    "RELEASE":   1.5,
    "RETREAT":   1.0,
}
PHASE_ORDER = ["APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "DELIVER", "RELEASE", "RETREAT"]


@dataclass
class State:
    ee_xyz: Optional[tuple] = None
    cube_xyz: Optional[tuple] = None


class ScriptedPickPlace(Node):
    def __init__(self, rate_hz: float = 30.0, max_gain: float = 1.0) -> None:
        super().__init__("scripted_pickplace")
        self.pub = self.create_publisher(Twist, "/teleop_cmd", 10)
        self.create_subscription(PoseStamped, "/cube_pose", self._on_cube, 10)
        self.create_subscription(PoseStamped, "/cartesian_pose", self._on_ee, 10)

        self.state = State()
        self.rate_hz = rate_hz
        self.max_gain = max_gain  # caps |Twist.linear| to keep motion smooth

        self._phase_idx = 0
        self._phase_start_t: Optional[float] = None
        self._cube_locked: Optional[tuple] = None  # cube_xyz snapshot at GRASP

        # Wait until both subscriptions have produced one message
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info("scripted_pickplace running")

    # ── Subscriptions ────────────────────────────────────────────────────

    def _on_cube(self, msg: PoseStamped) -> None:
        self.state.cube_xyz = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def _on_ee(self, msg: PoseStamped) -> None:
        self.state.ee_xyz = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    # ── Phase machine ────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self.state.ee_xyz is None or self.state.cube_xyz is None:
            return
        if self._phase_start_t is None:
            self._phase_start_t = time.time()

        if self._phase_idx >= len(PHASE_ORDER):
            self.pub.publish(Twist())  # stop
            self._finished = True
            rclpy.shutdown()
            return

        phase = PHASE_ORDER[self._phase_idx]
        elapsed = time.time() - self._phase_start_t
        if elapsed >= PHASE_DURATION[phase]:
            # Diagnostic: report EE and cube positions at the moment we leave
            # this phase, so we can see if waypoints were actually reached.
            ee = self.state.ee_xyz
            cube = self.state.cube_xyz
            self.get_logger().info(
                f"end {phase}: ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                f"cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f})"
            )
            self._phase_idx += 1
            self._phase_start_t = time.time()
            if self._phase_idx < len(PHASE_ORDER):
                next_phase = PHASE_ORDER[self._phase_idx]
                self.get_logger().info(f"→ {next_phase}")
            return

        # Compute target EE xyz for this phase
        cube_x, cube_y, cube_z = self.state.cube_xyz
        if phase in ("LIFT", "TRANSPORT", "DELIVER", "RELEASE", "RETREAT"):
            # After grasping, the policy should track the cube's last known
            # pre-grasp position rather than the actual cube (which is moving
            # with the gripper). Lock the reference at GRASP.
            if self._cube_locked is not None:
                cube_x, cube_y, cube_z = self._cube_locked

        if phase == "APPROACH":
            target = (cube_x, cube_y, HOVER_Z)
            gripper = +1.0
        elif phase == "DESCEND":
            target = (cube_x, cube_y, cube_z + GRASP_Z_OFFSET)
            gripper = +1.0
        elif phase == "GRASP":
            target = (cube_x, cube_y, cube_z + GRASP_Z_OFFSET)
            gripper = -1.0
            # Lock cube reference so subsequent phases reference the pre-pick pose.
            if self._cube_locked is None:
                self._cube_locked = self.state.cube_xyz
        elif phase == "LIFT":
            target = (cube_x, cube_y, HOVER_Z)
            gripper = -1.0
        elif phase == "TRANSPORT":
            target = (TARGET_XY[0], TARGET_XY[1], HOVER_Z)
            gripper = -1.0
        elif phase == "DELIVER":
            target = (TARGET_XY[0], TARGET_XY[1], PLACE_Z)
            gripper = -1.0
        elif phase == "RELEASE":
            target = (TARGET_XY[0], TARGET_XY[1], PLACE_Z)
            gripper = +1.0
        elif phase == "RETREAT":
            target = (TARGET_XY[0], TARGET_XY[1], HOVER_Z)
            gripper = +1.0
        else:
            target = self.state.ee_xyz
            gripper = 0.0

        # Compute clamped delta toward the target
        ee_x, ee_y, ee_z = self.state.ee_xyz
        dx, dy, dz = target[0] - ee_x, target[1] - ee_y, target[2] - ee_z

        # Velocity gain: drive proportional to distance, capped
        kp = 5.0
        vx, vy, vz = kp * dx, kp * dy, kp * dz
        norm = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
        if norm > self.max_gain:
            vx, vy, vz = (vx / norm) * self.max_gain, (vy / norm) * self.max_gain, (vz / norm) * self.max_gain

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.x = float(gripper)
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--max-gain", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = ScriptedPickPlace(rate_hz=args.rate, max_gain=args.max_gain)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy._rclpy_pybind11.RCLError):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
