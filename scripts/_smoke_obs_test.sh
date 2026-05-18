#!/bin/bash
# Smoke test for cube_pose observation extension.
set -e
. /opt/ros/humble/setup.bash
. /root/il_ws/install/setup.bash
cd "$(dirname "$0")/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)/il_pipeline"

echo "starting pybullet_robot_node"
nohup /usr/bin/python3 -u il_pipeline/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=42 -p gui:=false > /tmp/smoke_pb.log 2>&1 &
PB_PID=$!
echo "PB_PID=$PB_PID"
sleep 5

echo "--- topic list ---"
ros2 topic list | grep -E 'joint_states|cartesian|cube|task' || true

echo "--- cube_pose snapshot ---"
timeout 3 ros2 topic echo --once /cube_pose 2>&1 | head -15 || true

echo "--- joint_states snapshot (length) ---"
timeout 3 ros2 topic echo --once /joint_states 2>&1 | grep -c 'position\|velocity\|name' || true

kill -TERM $PB_PID 2>/dev/null || true
sleep 1
kill -KILL $PB_PID 2>/dev/null || true
echo "done"
