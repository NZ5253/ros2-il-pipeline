#!/bin/bash
#
# Collect N demonstration episodes through the full ROS 2 + FastAPI pipeline.
#
# Components launched:
#   - pybullet_robot_node  (simulated Franka Panda)
#   - data_logger_node     (records episodes to parquet)
#   - FastAPI service      (orchestration via ROS bridge)
#
# Then for each episode:
#   1. POST /datasets/{id}/record/start
#   2. Run scripted_teleop with randomised target
#   3. POST /datasets/{id}/record/stop
#
# Final dataset is at /tmp/mybotshop_demos/<dataset>/

set -e

N_EPISODES="${1:-20}"
DATASET_NAME="${2:-panda_pickplace_v1}"
DATASET_ROOT="/tmp/mybotshop_demos"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/jazzy/setup.bash
source /tmp/mybotshop_ws/install/setup.bash
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src"

rm -rf "${DATASET_ROOT:?}/${DATASET_NAME:?}"
mkdir -p "$DATASET_ROOT"

# Kill anything lingering from previous runs
pkill -f "pybullet_robot_node" 2>/dev/null || true
pkill -f "data_logger_node"    2>/dev/null || true
pkill -f "uvicorn.*il_pipeline" 2>/dev/null || true
sleep 1

echo "=== Starting pybullet_robot_node (background) ==="
/usr/bin/python3 src/il_pipeline/nodes/pybullet_robot_node.py \
    > /tmp/collect_pb.log 2>&1 &
PB_PID=$!

echo "=== Starting data_logger_node (background) ==="
/usr/bin/python3 src/il_pipeline/nodes/data_logger_node.py \
    --ros-args \
    -p dataset_root:="$DATASET_ROOT" \
    -p dataset_name:="$DATASET_NAME" \
    -p expected_fps:=30.0 \
    -p enable_camera:=false \
    > /tmp/collect_dl.log 2>&1 &
DL_PID=$!

echo "=== Starting FastAPI (background) ==="
/usr/bin/python3 -m uvicorn il_pipeline.web_api.app:app \
    --host 127.0.0.1 --port 8011 --log-level warning \
    > /tmp/collect_api.log 2>&1 &
API_PID=$!

# Wait for API
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8011/api/v1/health > /dev/null; then break; fi
    sleep 0.2
done
sleep 1

echo "=== Creating dataset ==="
DS_RESP=$(curl -s -X POST http://127.0.0.1:8011/api/v1/datasets \
    -H 'Content-Type: application/json' \
    -d "{\"name\": \"$DATASET_NAME\", \"robot_model\": \"franka_panda\", \"task_description\": \"reach-and-return\"}")
DS_ID=$(echo "$DS_RESP" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "  dataset_id: $DS_ID"

for i in $(seq 1 "$N_EPISODES"); do
    # Randomise a target within a small bounded volume
    TARGET_X=$(/usr/bin/python3 -c "import random; random.seed($i); print(0.25 + 0.10 * random.uniform(-1,1))")
    TARGET_Y=$(/usr/bin/python3 -c "import random; random.seed($i+100); print(0.10 * random.uniform(-1,1))")
    TARGET_Z=$(/usr/bin/python3 -c "import random; random.seed($i+200); print(-0.05 + 0.10 * random.uniform(-1,1))")

    EPISODE_NAME=$(printf "demo_%03d" "$i")
    echo "--- episode $i/$N_EPISODES  target=($TARGET_X, $TARGET_Y, $TARGET_Z) ---"

    curl -s -X POST "http://127.0.0.1:8011/api/v1/datasets/$DS_ID/record/start" \
        -H 'Content-Type: application/json' \
        -d "{\"episode_name\": \"$EPISODE_NAME\", \"task_description\": \"reach-and-return\"}" \
        > /dev/null

    /usr/bin/python3 scripts/scripted_teleop.py \
        --duration 3.0 --rate 30.0 \
        --target "$TARGET_X" "$TARGET_Y" "$TARGET_Z" \
        > /dev/null 2>&1 || true

    STOP_RESP=$(curl -s -X POST "http://127.0.0.1:8011/api/v1/datasets/$DS_ID/record/stop" \
        -H 'Content-Type: application/json' \
        -d '{"outcome": "success"}')
    FRAMES=$(echo "$STOP_RESP" | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("frame_count", d.get("detail", "?")))')
    echo "    frames=$FRAMES"

    # Small gap so the robot can settle back near home
    sleep 0.5
done

echo "=== Cleanup ==="
kill $API_PID 2>/dev/null || true
kill $DL_PID  2>/dev/null || true
kill $PB_PID  2>/dev/null || true
wait $API_PID 2>/dev/null
wait $DL_PID  2>/dev/null
wait $PB_PID  2>/dev/null

echo ""
echo "=== Dataset summary ==="
ls -la "$DATASET_ROOT/$DATASET_NAME/data/chunk-000/" 2>/dev/null | head -25
echo ""
echo "Episodes meta:"
cat "$DATASET_ROOT/$DATASET_NAME/meta/episodes.jsonl" 2>/dev/null
echo ""
echo "Info:"
cat "$DATASET_ROOT/$DATASET_NAME/meta/info.json" 2>/dev/null
