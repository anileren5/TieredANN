#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DIST_FN="l2"
R=32
BUILD_L=50
ALPHA=1.2
PQ_CHUNKS=16
CONFIG_TAG="R${R}_L${BUILD_L}_A${ALPHA}_PQ${PQ_CHUNKS}"

QUERY_FILE="./data/$DATASET/${DATASET}_query.bin"
GT_FILE="./data/$DATASET/${DATASET}_groundtruth.bin"
INDEX_PREFIX="./index/$DATASET/index_${DATASET}_learn_${CONFIG_TAG}"
RESULT_PATH="./results/$DATASET/"
TOP_K=100
SEARCH_L_LIST="128 256 512 1024"

# Run the command
./build/apps/search_memory_index \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --index_path_prefix "$INDEX_PREFIX" \
  --query_file "$QUERY_FILE" \
  --gt_file "$GT_FILE" \
  -K "$TOP_K" \
  -L $SEARCH_L_LIST \
  --result_path "$RESULT_PATH"
