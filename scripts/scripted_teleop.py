"""
Scripted teleoperator.

Publishes /teleop_cmd Twist messages that drive the simulated robot through
a parameterised reach-and-return motion. Used in place of a human teleop
operator for unattended data collection — it produces deterministic, easily
varied demonstrations that look reasonable in joint space.

For real demonstrations, replace this script with keyboard / joy teleop
through the MyBotShop UI. The data logger sees the same /teleop_cmd stream
either way, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ScriptedTeleop(Node):
    def __init__(
        self,
        duration_s: float,
        rate_hz: float,
        target_xyz: tuple[float, float, float],
        return_to_start: bool,
    ) -> None:
        super().__init__("scripted_teleop")
        self.pub = self.create_publisher(Twist, "/teleop_cmd", 10)
        self.duration_s = duration_s
        self.rate_hz = rate_hz
        self.target_xyz = target_xyz
        self.return_to_start = return_to_start

        self._start_t = time.time()
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _tick(self) -> None:
        t = (time.time() - self._start_t) / self.duration_s
        if t >= 1.0:
            zero = Twist()
            self.pub.publish(zero)
            self.get_logger().info("scripted teleop finished")
            rclpy.shutdown()
            return

        # Smoothstep velocity profile: starts at 0, peaks mid-motion, ends at 0
        # to make trajectories that look like human demonstrations.
        if self.return_to_start:
            # Out then back: two smoothsteps in opposite directions
            phase = t * 2.0
            if phase < 1.0:
                sign = 1.0
            else:
                phase -= 1.0
                sign = -1.0
            v = sign * (6.0 * phase * (1.0 - phase))  # velocity = derivative
        else:
            v = 6.0 * t * (1.0 - t)  # derivative of smoothstep

        msg = Twist()
        msg.linear.x = v * self.target_xyz[0]
        msg.linear.y = v * self.target_xyz[1]
        msg.linear.z = v * self.target_xyz[2]
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Total motion duration in seconds")
    parser.add_argument("--rate", type=float, default=30.0,
                        help="Command rate in Hz (match the data logger fps)")
    parser.add_argument("--target", type=float, nargs=3, default=[0.4, 0.0, -0.2],
                        metavar=("X", "Y", "Z"),
                        help="EE delta in metres to drive toward")
    parser.add_argument("--return-to-start", action="store_true", default=True,
                        help="Move out then back, producing a closed-loop demo")
    args = parser.parse_args()

    rclpy.init()
    node = ScriptedTeleop(
        duration_s=args.duration,
        rate_hz=args.rate,
        target_xyz=tuple(args.target),
        return_to_start=args.return_to_start,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy._rclpy_pybind11.RCLError:
        # raised when shutdown() is called inside the timer; ignore
        pass
    finally:
        with contextlib.suppress(Exception):
            node.destroy_node()


if __name__ == "__main__":
    main()
