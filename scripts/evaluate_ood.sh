#!/bin/bash
#
# Out-of-distribution evaluation. Same protocol as evaluate.sh but the
# cube spawn range is shifted outside the training distribution.
#
# Training: x in [0.40, 0.55], y in [-0.15, 0.05]
# OOD test: x in [0.55, 0.65], y in [ 0.05, 0.15]   (no overlap)

set +e

CKPT="${1:?usage: $0 <checkpoint.pt> <bc|act|diffusion> <n_rollouts>}"
POLICY_TYPE="${2:?usage: $0 <checkpoint.pt> <bc|act|diffusion> <n_rollouts>}"
N_ROLLOUTS="${3:-20}"
TIMEOUT_PER_ROLLOUT="${TIMEOUT_PER_ROLLOUT:-25}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/humble/setup.bash > /dev/null
source /root/il_ws/install/setup.bash > /dev/null
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/il_pipeline"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "=== Starting pybullet_robot_node (OOD spawn range) ==="
/usr/bin/python3 -u il_pipeline/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=2000 \
    -p cube_spawn_x_min:=0.55 -p cube_spawn_x_max:=0.65 \
    -p cube_spawn_y_min:=0.05 -p cube_spawn_y_max:=0.15 \
    > /tmp/eval_ood_pb.log 2>&1 &
PB_PID=$!
sleep 3

echo "=== Starting inference_node ==="
/usr/bin/python3 -u il_pipeline/il_pipeline/nodes/inference_node.py \
    --ros-args \
    -p checkpoint_path:="$CKPT" \
    -p policy_type:="$POLICY_TYPE" \
    -p inference_rate_hz:=30.0 \
    -p execution_mode:=first_action \
    -p device:="$EVAL_DEVICE" \
    > /tmp/eval_ood_inf.log 2>&1 &
INF_PID=$!
sleep 3

ros2 service call /inference_node/load_policy il_pipeline_msgs/srv/LoadPolicy \
    "{checkpoint_path: '$CKPT', policy_type: '$POLICY_TYPE', inference_rate_hz: 30.0, execution_mode: 'first_action'}" \
    > /dev/null 2>&1

SUCCESSES=0
for i in $(seq 1 "$N_ROLLOUTS"); do
    echo "--- rollout $i/$N_ROLLOUTS ---"
    ros2 service call /pybullet_robot_node/reset std_srvs/srv/Trigger > /dev/null 2>&1
    sleep 1
    ros2 service call /inference_node/start std_srvs/srv/Trigger > /dev/null 2>&1
    sleep "$TIMEOUT_PER_ROLLOUT"
    ros2 service call /inference_node/stop std_srvs/srv/Trigger > /dev/null 2>&1
    STATUS=$(timeout 3 /usr/bin/python3 -u scripts/check_task_status.py --timeout 1.5 --mode any 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        SUCCESSES=$((SUCCESSES + 1))
        echo "    success"
    else
        echo "    fail"
    fi
done

echo ""
echo "=== OOD Result ==="
echo "Policy: $POLICY_TYPE @ $CKPT"
echo "Spawn shift: x[0.55,0.65] y[0.05,0.15]  (training was x[0.40,0.55] y[-0.15,0.05])"
echo "Rollouts: $N_ROLLOUTS"
echo "Successes: $SUCCESSES"
RATE=$(/usr/bin/python3 -c "print(f'{$SUCCESSES / $N_ROLLOUTS * 100:.1f}')")
echo "Success rate: $RATE%"

kill -TERM $INF_PID $PB_PID 2>/dev/null
sleep 1
kill -KILL $INF_PID $PB_PID 2>/dev/null
