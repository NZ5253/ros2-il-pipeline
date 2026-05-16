#!/bin/bash
#
# Replay the trained BC policy through the inference_node, with the
# PyBullet robot simulating the Franka.
#
# Components launched:
#   - pybullet_robot_node   (simulated Franka, publishes /joint_states + /cartesian_pose)
#   - inference_node        (loads BC checkpoint, publishes /cmd_robot)
#
# Then we log /cmd_robot output for ~5 seconds to measure inference latency and
# verify the policy is producing reasonable commands.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/jazzy/setup.bash
source /tmp/mybotshop_ws/install/setup.bash
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src"

CKPT="$REPO_ROOT/runs/panda_reach_v1_bc/best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

# Cleanup
pkill -f "pybullet_robot_node" 2>/dev/null || true
pkill -f "inference_node"       2>/dev/null || true
sleep 1

echo "=== Starting pybullet_robot_node ==="
/usr/bin/python3 src/il_pipeline/nodes/pybullet_robot_node.py > /tmp/replay_pb.log 2>&1 &
PB_PID=$!
sleep 3

echo "=== Starting inference_node ==="
/usr/bin/python3 src/il_pipeline/nodes/inference_node.py \
    --ros-args \
    -p checkpoint_path:="$CKPT" \
    -p policy_type:=bc \
    -p inference_rate_hz:=30.0 \
    -p execution_mode:=first_action \
    -p device:=cpu \
    > /tmp/replay_inf.log 2>&1 &
INF_PID=$!
sleep 2

echo "=== Loading policy via service call ==="
ros2 service call /inference_node/load_policy il_pipeline_msgs/srv/LoadPolicy \
    "{checkpoint_path: '$CKPT', policy_type: 'bc', inference_rate_hz: 30.0, execution_mode: 'first_action'}" \
    2>&1 | tail -3

echo ""
echo "=== Starting inference loop ==="
ros2 service call /inference_node/start std_srvs/srv/Trigger 2>&1 | tail -3

echo ""
echo "=== Recording /cmd_robot output for 5 seconds ==="
timeout 5 ros2 topic echo /cmd_robot --field linear > /tmp/replay_cmd.log 2>&1 || true

echo ""
echo "=== Stopping inference ==="
ros2 service call /inference_node/stop std_srvs/srv/Trigger 2>&1 | tail -3 || true

# Tail of joint states to verify the robot moved
echo ""
echo "=== Final joint state ==="
timeout 2 ros2 topic echo --once /joint_states --field position 2>&1 | head -10

# Cleanup
kill $INF_PID 2>/dev/null || true
kill $PB_PID  2>/dev/null || true
wait $INF_PID 2>/dev/null
wait $PB_PID  2>/dev/null

echo ""
echo "=== /cmd_robot output summary (commands published over 5s) ==="
wc -l /tmp/replay_cmd.log

echo ""
echo "=== inference_node log tail ==="
tail -10 /tmp/replay_inf.log
