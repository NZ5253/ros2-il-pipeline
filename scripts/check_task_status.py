"""
Read /task_status for a short window and exit 0 if any True is observed.

Used by collect_demos.sh after the scripted expert finishes, to decide
whether the episode is a success or should be discarded. Reading multiple
messages avoids the trap where `ros2 topic echo --once` returns a cached
False from before the policy reached the target.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class StatusReader(Node):
    def __init__(self, timeout_s: float) -> None:
        super().__init__("task_status_reader")
        self._latest: bool = False
        self._any_true: bool = False
        self.create_subscription(Bool, "/task_status", self._on_status, 10)
        self._deadline = time.time() + timeout_s

    def _on_status(self, msg: Bool) -> None:
        self._latest = bool(msg.data)
        if msg.data:
            self._any_true = True

    def spin_until_deadline(self) -> None:
        while time.time() < self._deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=1.5,
                        help="seconds to listen on /task_status")
    parser.add_argument("--mode", choices=("latest", "any"), default="any",
                        help="any: success if any True observed in window; latest: success if latest is True")
    args = parser.parse_args()

    rclpy.init()
    node = StatusReader(args.timeout)
    try:
        node.spin_until_deadline()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    success = node._any_true if args.mode == "any" else node._latest
    print("True" if success else "False")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
