#!/bin/bash
#
# End-to-end integration test:
#   1. Boot data_logger_node (ROS 2)
#   2. Boot FastAPI service with live ROS bridge
#   3. Boot a fake publisher that emits /joint_states + /teleop_cmd
#   4. POST /datasets to register a dataset
#   5. POST /record/start (triggers data_logger via the bridge)
#   6. Wait while frames accumulate
#   7. POST /record/stop (triggers data_logger via the bridge, writes parquet)
#   8. Verify on-disk artefacts
#
# Run from the repo root.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Sourcing ROS 2 Jazzy ==="
source /opt/ros/jazzy/setup.bash
source /tmp/mybotshop_ws/install/setup.bash

export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src"

rm -rf /tmp/integration_test_datasets

echo ""
echo "=== Starting data_logger_node (background) ==="
/usr/bin/python3 "$REPO_ROOT/src/il_pipeline/nodes/data_logger_node.py" \
    --ros-args \
    -p dataset_root:=/tmp/integration_test_datasets \
    -p dataset_name:=integration_run \
    -p expected_fps:=30.0 \
    -p enable_camera:=false \
    > /tmp/integration_datalogger.log 2>&1 &
DL_PID=$!

echo ""
echo "=== Starting FastAPI service (background) ==="
/usr/bin/python3 -m uvicorn il_pipeline.web_api.app:app --host 127.0.0.1 --port 8011 --log-level warning \
    > /tmp/integration_fastapi.log 2>&1 &
API_PID=$!

# Wait for the API to be ready
echo ""
echo "=== Waiting for FastAPI to come up ==="
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8011/api/v1/health > /dev/null; then
        echo "  API ready after ${i} attempts"
        break
    fi
    sleep 0.2
done

echo ""
echo "=== Starting fake publisher (background) ==="
/usr/bin/python3 - <<'EOF' &
import time
import rclpy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

rclpy.init()
node = rclpy.create_node('fake_publisher_integration')
js_pub = node.create_publisher(JointState, '/joint_states', 10)
tc_pub = node.create_publisher(Twist, '/teleop_cmd', 10)
time.sleep(0.5)
for i in range(180):  # ~6 seconds at 30 Hz
    js = JointState()
    js.position = [0.1 + 0.001 * i] * 7
    js.velocity = [0.0] * 7
    js_pub.publish(js)
    tc = Twist()
    tc.linear.x = 0.001 * i
    tc_pub.publish(tc)
    rclpy.spin_once(node, timeout_sec=0.0)
    time.sleep(1.0 / 30)
rclpy.shutdown()
EOF
PUB_PID=$!

sleep 1

echo ""
echo "=== Creating dataset via FastAPI ==="
DS_RESP=$(curl -s -X POST http://127.0.0.1:8011/api/v1/datasets \
    -H 'Content-Type: application/json' \
    -d '{"name": "integration_run", "robot_model": "panda", "task_description": "synthetic pickplace"}')
echo "  → $DS_RESP"
DS_ID=$(echo "$DS_RESP" | /usr/bin/python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')
echo "  dataset_id: $DS_ID"

echo ""
echo "=== Starting recording (HTTP → ROS bridge → data_logger) ==="
START_RESP=$(curl -s -X POST "http://127.0.0.1:8011/api/v1/datasets/$DS_ID/record/start" \
    -H 'Content-Type: application/json' \
    -d '{"episode_name": "integration_ep_001", "task_description": "pickplace test"}')
echo "  → $START_RESP"

echo ""
echo "=== Recording for 2 seconds ==="
sleep 2

echo ""
echo "=== Stopping recording (HTTP → ROS bridge → data_logger → parquet) ==="
STOP_RESP=$(curl -s -X POST "http://127.0.0.1:8011/api/v1/datasets/$DS_ID/record/stop" \
    -H 'Content-Type: application/json' \
    -d '{"outcome": "success"}')
echo "  → $STOP_RESP"

# Cleanup
wait $PUB_PID 2>/dev/null
kill $DL_PID 2>/dev/null
kill $API_PID 2>/dev/null
wait $DL_PID 2>/dev/null
wait $API_PID 2>/dev/null

echo ""
echo "=== On-disk artefacts ==="
find /tmp/integration_test_datasets -type f | sort

echo ""
echo "=== Parquet contents (frame count + columns) ==="
/usr/bin/python3 - <<'EOF'
import pyarrow.parquet as pq
from pathlib import Path
for p in Path("/tmp/integration_test_datasets").rglob("*.parquet"):
    t = pq.read_table(p)
    print(f"  {p}")
    print(f"    columns: {t.column_names}")
    print(f"    frames:  {len(t)}")
EOF

echo ""
echo "=== Integration test complete ==="
