#!/bin/bash
export DATASET_ROOT=/mnt/c/Users/smnazain/mybotshop_eval/dataset
N=${1:-5}
NAME=${2:-panda_pickplace_v2_smoke}
bash "$(dirname "$0")/collect_demos.sh" "$N" "$NAME"
