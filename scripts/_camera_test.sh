#!/bin/bash
# Test the new PyBullet GUI camera view: start the robot node briefly
# under Xvfb, capture a single frame, then exit.
. /opt/ros/humble/setup.bash
. /root/il_ws/install/setup.bash
cd "$(dirname "$0")/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)/il_pipeline"

Xvfb :99 -screen 0 1280x720x24 > /dev/null 2>&1 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1

nohup /usr/bin/python3 -u il_pipeline/il_pipeline/nodes/pybullet_robot_node.py \
    --ros-args -p seed:=42 -p gui:=true > /tmp/cam_test_pb.log 2>&1 &
PB=$!
sleep 6
import -window root /tmp/cam_test.png 2>/dev/null || \
    ffmpeg -y -video_size 1280x720 -f x11grab -i $DISPLAY -frames:v 1 \
        /mnt/c/Users/smnazain/mybotshop_eval/cam_test.png 2>&1 | tail -1
kill -TERM $PB 2>/dev/null
sleep 1
kill -KILL $PB 2>/dev/null
kill -TERM $XVFB_PID 2>/dev/null
ls -lh /mnt/c/Users/smnazain/mybotshop_eval/cam_test.png 2>&1 | tail -1
