#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DIST_FN="l2"
R=64
L_BUILD=128
PQ_CHUNKS=0
ALPHA=1.2
CONFIG_TAG="R${R}_L${L_BUILD}_A${ALPHA}_PQ${PQ_CHUNKS}"

QUERY_FILE="./data/$DATASET/${DATASET}_query.bin"
GT_FILE="./data/$DATASET/${DATASET}_groundtruth.bin"
INDEX_PREFIX="./index/$DATASET/disk_index_${DATASET}_learn_${CONFIG_TAG}"
RESULT_PATH="./result/$DATASET/"
TOP_K=100
SEARCH_L_LIST="128 256 512 1024 2048 4096 8192"
NUM_NODES_TO_CACHE=10000

# Run the command
./build/apps/search_disk_index \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --index_path_prefix "$INDEX_PREFIX" \
  --query_file "$QUERY_FILE" \
  --gt_file "$GT_FILE" \
  -K "$TOP_K" \
  -L $SEARCH_L_LIST \
  --result_path "$RESULT_PATH" \
  --num_nodes_to_cache "$NUM_NODES_TO_CACHE"
