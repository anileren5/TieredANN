#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DIST_FN="l2"
R=32
L=50
ALPHA=1.2
PQ_CHUNKS=16
CONFIG_TAG="R${R}_L${L}_A${ALPHA}_PQ${PQ_CHUNKS}"

DATA_PATH="./data/$DATASET/${DATASET}_base.bin"
INDEX_PREFIX="./index/$DATASET/index_${DATASET}_learn_${CONFIG_TAG}"

# Run the command
./build/apps/build_memory_index \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --data_path "$DATA_PATH" \
  --index_path_prefix "$INDEX_PREFIX" \
  -R "$R" \
  -L "$L" \
  --alpha "$ALPHA" \
  --build_PQ_bytes "$PQ_CHUNKS"
