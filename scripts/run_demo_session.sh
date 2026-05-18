#!/bin/bash
#
# Demo session launcher for the CEO walkthrough.
#
# Starts the full IL pipeline stack, loads the trained ACT policy, and runs
# 5 pick-and-place rollouts with verbose output.
#
# GUI mode (PYBULLET_GUI=1) requires a native Linux desktop or Windows with
# WSLg properly configured (DISPLAY set). Headless mode (default) runs the
# same pipeline with no window.
#
# Usage:
#   bash scripts/run_demo_session.sh                # headless
#   PYBULLET_GUI=1 bash scripts/run_demo_session.sh # GUI (needs X11/WSLg)
#
# For screen recording:
#   ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 \
#          -t 180 -c:v libx264 -pix_fmt yuv420p demo.mp4

set +e

POLICY_CKPT="${POLICY_CKPT:-runs/panda_act/best.pt}"
POLICY_TYPE="${POLICY_TYPE:-act}"
N_ROLLOUTS="${N_ROLLOUTS:-5}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
PYBULLET_GUI="${PYBULLET_GUI:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/humble/setup.bash > /dev/null
source /root/il_ws/install/setup.bash > /dev/null
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/il_pipeline"

echo "=== MyBotShop IL Pipeline Demo ==="
echo "Policy: $POLICY_TYPE @ $POLICY_CKPT"
echo "Device: $EVAL_DEVICE"
echo "Rollouts: $N_ROLLOUTS"
echo "GUI: $PYBULLET_GUI"
echo ""

if [ ! -f "$POLICY_CKPT" ]; then
    echo "ERROR: checkpoint not found: $POLICY_CKPT"
    echo "Run scripts/train.py first."
    exit 1
fi

echo "=== Starting pybullet_robot_node ==="
/usr/bin/python3 -u il_pipeline/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=42 -p use_gui:="$PYBULLET_GUI" \
    > /tmp/demo_pb.log 2>&1 &
PB_PID=$!
sleep 3

echo "=== Starting inference_node ==="
/usr/bin/python3 -u il_pipeline/il_pipeline/nodes/inference_node.py \
    --ros-args \
    -p checkpoint_path:="$POLICY_CKPT" \
    -p policy_type:="$POLICY_TYPE" \
    -p inference_rate_hz:=30.0 \
    -p execution_mode:=first_action \
    -p device:="$EVAL_DEVICE" \
    > /tmp/demo_inf.log 2>&1 &
INF_PID=$!
sleep 3

echo "=== Loading policy ==="
ros2 service call /inference_node/load_policy il_pipeline_msgs/srv/LoadPolicy \
    "{checkpoint_path: '$POLICY_CKPT', policy_type: '$POLICY_TYPE', inference_rate_hz: 30.0, execution_mode: 'first_action'}" \
    > /dev/null 2>&1

SUCCESSES=0
for i in $(seq 1 "$N_ROLLOUTS"); do
    echo ""
    echo "--- rollout $i/$N_ROLLOUTS ---"
    ros2 service call /pybullet_robot_node/reset std_srvs/srv/Trigger > /dev/null 2>&1
    sleep 1
    ros2 service call /inference_node/start std_srvs/srv/Trigger > /dev/null 2>&1
    sleep 20
    ros2 service call /inference_node/stop std_srvs/srv/Trigger > /dev/null 2>&1
    STATUS=$(timeout 3 /usr/bin/python3 -u scripts/check_task_status.py --timeout 1.5 --mode any 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        SUCCESSES=$((SUCCESSES + 1))
        echo "    RESULT: success"
    else
        echo "    RESULT: fail"
    fi
done

echo ""
echo "=== Demo complete ==="
echo "Policy: $POLICY_TYPE"
echo "Successes: $SUCCESSES / $N_ROLLOUTS"

kill -TERM $INF_PID $PB_PID 2>/dev/null
sleep 1
kill -KILL $INF_PID $PB_PID 2>/dev/null
