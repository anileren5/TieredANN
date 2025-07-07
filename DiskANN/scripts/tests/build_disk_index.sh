#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DIST_FN="l2"
R=64
L=128
B=0.003
M=1
PQ_CHUNKS=0
CONFIG_TAG="R${R}_L${L}_A1.2_PQ${PQ_CHUNKS}"  # Alpha is fixed at 1.2
DATA_PATH="./data/$DATASET/${DATASET}_base.bin"
INDEX_PREFIX="./index/$DATASET/disk_index_${DATASET}_learn_${CONFIG_TAG}"

# Run the command
./build/apps/build_disk_index \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --data_path "$DATA_PATH" \
  --index_path_prefix "$INDEX_PREFIX" \
  -R "$R" \
  -L "$L" \
  -B "$B" \
  -M "$M" \
  #--build_PQ_bytes "$PQ_CHUNKS"
