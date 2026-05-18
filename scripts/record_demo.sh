#!/bin/bash
#
# Records the CEO demo video using a virtual framebuffer (Xvfb + Mesa SW).
#
# Requires: xvfb, ffmpeg  (sudo apt-get install -y xvfb ffmpeg)
#
# Usage:
#   bash scripts/record_demo.sh                     # produces demo.mp4
#   OUTPUT=my_demo.mp4 bash scripts/record_demo.sh

set +e

POLICY_CKPT="${POLICY_CKPT:-runs/panda_act/best.pt}"
POLICY_TYPE="${POLICY_TYPE:-act}"
N_ROLLOUTS="${N_ROLLOUTS:-5}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-demo.mp4}"
DISPLAY_NUM=":99"
RESOLUTION="1280x720"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Starting virtual display ($RESOLUTION) ==="
Xvfb "$DISPLAY_NUM" -screen 0 "${RESOLUTION}x24" &
XVFB_PID=$!
sleep 2
export DISPLAY="$DISPLAY_NUM"
export LIBGL_ALWAYS_SOFTWARE=1

echo "=== Starting ffmpeg recorder ==="
ffmpeg -y \
    -video_size "$RESOLUTION" -framerate 30 \
    -f x11grab -i "${DISPLAY_NUM}.0" \
    -c:v libx264 -preset fast -pix_fmt yuv420p \
    "$OUTPUT" \
    > /tmp/ffmpeg_record.log 2>&1 &
FFMPEG_PID=$!
sleep 1

echo "=== Launching demo session (GUI enabled) ==="
PYBULLET_GUI=1 EVAL_DEVICE="$EVAL_DEVICE" N_ROLLOUTS="$N_ROLLOUTS" \
    bash scripts/run_demo_session.sh

echo ""
echo "=== Stopping recorder ==="
kill -INT "$FFMPEG_PID" 2>/dev/null
wait "$FFMPEG_PID" 2>/dev/null
kill -TERM "$XVFB_PID" 2>/dev/null

echo "Video written: $OUTPUT"
ls -lh "$OUTPUT"
