#!/bin/bash
#
# Collect N pick-and-place demonstration episodes through the full
# ROS 2 + FastAPI pipeline.
#
# Components launched once:
#   - pybullet_robot_node  (simulated Franka Panda + cube + target zone)
#   - data_logger_node     (records episodes to LeRobotDataset parquet)
#   - FastAPI service      (orchestration via the ROS bridge)
#
# Per episode:
#   1. Call /pybullet_robot_node/reset (randomises cube spawn, returns arm home)
#   2. POST /datasets/{id}/record/start
#   3. Run scripted_pickplace expert
#   4. Read final /task_status to label success/failure
#   5. POST /datasets/{id}/record/stop with outcome=success or discard
#
# Final dataset is at $DATASET_ROOT/<dataset>/.

set +e

N_EPISODES="${1:-30}"
DATASET_NAME="${2:-panda_pickplace_v2}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/mybotshop_demos}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/humble/setup.bash > /dev/null
source /root/il_ws/install/setup.bash > /dev/null
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/il_pipeline"

rm -rf "${DATASET_ROOT:?}/${DATASET_NAME:?}"
mkdir -p "$DATASET_ROOT"

echo "=== Starting pybullet_robot_node ==="
/usr/bin/python3 -u il_pipeline/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=0 \
    > /tmp/collect_pb.log 2>&1 &
PB_PID=$!

echo "=== Starting data_logger_node ==="
/usr/bin/python3 -u il_pipeline/il_pipeline/nodes/data_logger_node.py \
    --ros-args \
    -p dataset_root:="$DATASET_ROOT" \
    -p dataset_name:="$DATASET_NAME" \
    -p expected_fps:=30.0 \
    -p enable_camera:=false \
    > /tmp/collect_dl.log 2>&1 &
DL_PID=$!

echo "=== Starting FastAPI ==="
/usr/bin/python3 -u -m uvicorn il_pipeline.web_api.app:app \
    --host 127.0.0.1 --port 8011 --log-level warning \
    > /tmp/collect_api.log 2>&1 &
API_PID=$!

# Wait for API readiness
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8011/api/v1/health > /dev/null; then break; fi
    sleep 0.3
done
sleep 1

echo "=== Creating dataset ==="
DS_RESP=$(curl -s -X POST http://127.0.0.1:8011/api/v1/datasets \
    -H 'Content-Type: application/json' \
    -d "{\"name\": \"$DATASET_NAME\", \"robot_model\": \"franka_panda\", \"task_description\": \"pick-and-place red cube to target zone\"}")
DS_ID=$(echo "$DS_RESP" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "  dataset_id: $DS_ID"

SUCCESSES=0
FAILURES=0
for i in $(seq 1 "$N_EPISODES"); do
    EPISODE_NAME=$(printf "demo_%03d" "$i")
    echo "--- episode $i/$N_EPISODES ($EPISODE_NAME) ---"

    # Reset the simulated robot and respawn the cube at a new random pose
    ros2 service call /pybullet_robot_node/reset std_srvs/srv/Trigger > /dev/null 2>&1
    sleep 1

    # Start recording
    curl -s -X POST "http://127.0.0.1:8011/api/v1/datasets/$DS_ID/record/start" \
        -H 'Content-Type: application/json' \
        -d "{\"episode_name\": \"$EPISODE_NAME\", \"task_description\": \"pick-and-place\"}" \
        > /dev/null

    # Run scripted expert (blocking; ends when phase machine completes ~16 s)
    timeout --kill-after=2 20 /usr/bin/python3 -u scripts/scripted_pickplace.py > /tmp/collect_expert.log 2>&1

    # Read task_status for a short window: success if ANY True was observed
    # between expert finish and now. This avoids stale cached False values.
    STATUS=$(timeout 3 /usr/bin/python3 -u scripts/check_task_status.py --timeout 1.5 --mode any 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        OUTCOME="success"
        SUCCESSES=$((SUCCESSES + 1))
    else
        OUTCOME="discard"
        FAILURES=$((FAILURES + 1))
    fi

    STOP_RESP=$(curl -s -X POST "http://127.0.0.1:8011/api/v1/datasets/$DS_ID/record/stop" \
        -H 'Content-Type: application/json' \
        -d "{\"outcome\": \"$OUTCOME\"}")
    FRAMES=$(echo "$STOP_RESP" | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("frame_count", "?"))')
    echo "    outcome=$OUTCOME frames=$FRAMES"
done

echo ""
echo "=== Summary ==="
echo "successes=$SUCCESSES / $N_EPISODES"
echo "failures =$FAILURES"

echo ""
echo "=== Cleanup ==="
kill -TERM $API_PID 2>/dev/null
kill -TERM $DL_PID 2>/dev/null
kill -TERM $PB_PID 2>/dev/null
sleep 1
kill -KILL $API_PID 2>/dev/null
kill -KILL $DL_PID 2>/dev/null
kill -KILL $PB_PID 2>/dev/null

echo ""
echo "=== Dataset summary ==="
ls "$DATASET_ROOT/$DATASET_NAME/data/chunk-000/" 2>/dev/null | wc -l
echo "Info:"
cat "$DATASET_ROOT/$DATASET_NAME/meta/info.json" 2>/dev/null
