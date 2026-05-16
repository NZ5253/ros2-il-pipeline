#!/bin/bash
#
# Closed-loop policy evaluation: run N rollouts of the trained policy in the
# PyBullet sim, report success rate.
#
# Usage:
#   bash scripts/evaluate.sh <checkpoint.pt> <bc|act> <n_rollouts>
#
# Per rollout:
#   1. Call /pybullet_robot_node/reset (random cube spawn)
#   2. /inference_node/load_policy with the given checkpoint
#   3. /inference_node/start
#   4. Wait fixed time (~18 s) for the policy to attempt the task
#   5. /inference_node/stop
#   6. Read /task_status and count as success if any True seen
#
# Final output: success rate over N rollouts.

set +e

CKPT="${1:?usage: $0 <checkpoint.pt> <bc|act> <n_rollouts>}"
POLICY_TYPE="${2:?usage: $0 <checkpoint.pt> <bc|act> <n_rollouts>}"
N_ROLLOUTS="${3:-20}"
TIMEOUT_PER_ROLLOUT="${TIMEOUT_PER_ROLLOUT:-18}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /opt/ros/jazzy/setup.bash > /dev/null
source /tmp/mybotshop_ws/install/setup.bash > /dev/null
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "=== Starting pybullet_robot_node ==="
/usr/bin/python3 -u src/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=1000 \
    > /tmp/eval_pb.log 2>&1 &
PB_PID=$!
sleep 3

echo "=== Starting inference_node ==="
/usr/bin/python3 -u src/il_pipeline/nodes/inference_node.py \
    --ros-args \
    -p checkpoint_path:="$CKPT" \
    -p policy_type:="$POLICY_TYPE" \
    -p inference_rate_hz:=30.0 \
    -p execution_mode:=first_action \
    -p device:=cpu \
    > /tmp/eval_inf.log 2>&1 &
INF_PID=$!
sleep 3

echo "=== Loading policy ==="
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
echo "=== Result ==="
echo "Policy: $POLICY_TYPE @ $CKPT"
echo "Rollouts: $N_ROLLOUTS"
echo "Successes: $SUCCESSES"
RATE=$(/usr/bin/python3 -c "print(f'{$SUCCESSES / $N_ROLLOUTS * 100:.1f}')")
echo "Success rate: $RATE%"

kill -TERM $INF_PID 2>/dev/null
kill -TERM $PB_PID 2>/dev/null
sleep 1
kill -KILL $INF_PID 2>/dev/null
kill -KILL $PB_PID 2>/dev/null
