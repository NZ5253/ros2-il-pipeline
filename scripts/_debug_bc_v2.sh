#!/bin/bash
# Debug: spin up only the robot node and BC inference, log full observations
# during a single rollout, then verify the cube xyz is non-zero.
. /opt/ros/humble/setup.bash
. /root/il_ws/install/setup.bash
cd "$(dirname "$0")/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)/il_pipeline"

nohup /usr/bin/python3 -u il_pipeline/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=42 -p gui:=false > /tmp/dbg_pb.log 2>&1 &
PB=$!
sleep 4

echo "--- /cube_pose available? ---"
timeout 2 ros2 topic echo --once /cube_pose 2>&1 | grep -A2 'position:' | head -5

echo ""
echo "--- inline inference test ---"
/usr/bin/python3 - <<'PY'
import sys, time
sys.path.insert(0, "il_pipeline")
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import torch
import numpy as np
from il_pipeline.inference.policy_loader import load_policy
from pathlib import Path

rclpy.init()
node = rclpy.create_node("dbg")
js = [None]
ep = [None]
cp = [None]
def on_js(msg): js[0] = msg
def on_ep(msg): ep[0] = msg
def on_cp(msg): cp[0] = msg
node.create_subscription(JointState, "/joint_states", on_js, 10)
node.create_subscription(PoseStamped, "/cartesian_pose", on_ep, 10)
node.create_subscription(PoseStamped, "/cube_pose", on_cp, 10)
for _ in range(60):
    rclpy.spin_once(node, timeout_sec=0.05)
    if js[0] is not None and ep[0] is not None and cp[0] is not None:
        break

print(f"joint_states  received: {js[0] is not None}")
print(f"cartesian_pose received: {ep[0] is not None}")
print(f"cube_pose      received: {cp[0] is not None}")
if cp[0]:
    print(f"cube xyz: {cp[0].pose.position.x:.3f}, {cp[0].pose.position.y:.3f}, {cp[0].pose.position.z:.3f}")
if ep[0]:
    print(f"ee xyz:   {ep[0].pose.position.x:.3f}, {ep[0].pose.position.y:.3f}, {ep[0].pose.position.z:.3f}")

# Build the same observation the inference node would build
m = js[0]
joint_pos = np.array(m.position, dtype=np.float32)
joint_vel = np.array(m.velocity, dtype=np.float32) if m.velocity else np.zeros_like(joint_pos)
ee = np.array([
    ep[0].pose.position.x, ep[0].pose.position.y, ep[0].pose.position.z,
    ep[0].pose.orientation.x, ep[0].pose.orientation.y,
    ep[0].pose.orientation.z, ep[0].pose.orientation.w,
], dtype=np.float32)
obj = np.array([cp[0].pose.position.x, cp[0].pose.position.y, cp[0].pose.position.z], dtype=np.float32) if cp[0] else np.zeros(3, dtype=np.float32)
state = np.concatenate([joint_pos, joint_vel, ee, obj])
print(f"\nstate dim: {state.shape[0]}, last 3 (should match cube): {state[-3:]}")

# Load policy and predict
device = torch.device("cuda:0")
policy, normaliser = load_policy(Path("runs/panda_bc_v2/best.pt"), "bc", device)
print(f"normaliser obs_mean shape: {normaliser._obs_mean.shape}")

obs_t = normaliser.normalise_obs(state).to(device)
with torch.inference_mode():
    action_chunk = policy.predict_action_chunk(obs_t)
action_chunk = normaliser.denormalise_action(action_chunk).cpu().numpy()
print(f"action shape: {action_chunk.shape}")
print(f"action: {action_chunk[0]}")
PY

kill -TERM $PB 2>/dev/null || true
sleep 1
kill -KILL $PB 2>/dev/null || true
