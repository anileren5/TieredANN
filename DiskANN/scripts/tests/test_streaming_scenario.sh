#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# ====== Parameters ======
DATASET="sift"
DATA_TYPE="float"
DIST_FN="l2"

DATA_PATH="data/$DATASET/${DATASET}_learn.bin"
QUERY_FILE="data/$DATASET/${DATASET}_query.bin"
INDEX_PREFIX="index/$DATASET/idx_learn_str"
RESULT_PATH="result/$DATASET/results"

# Index config
R=64
L=600
ALPHA=1.2
INS_THR=16
CONS_THR=16
INSERTS=100000
ACTIVE=20000
CONS_INT=10000

# Derived index & GT paths
INDEX="${INDEX_PREFIX}.after-streaming-act${ACTIVE}-cons${CONS_INT}-max${INSERTS}"
GT="data/$DATASET/$DATASET"

# Search parameters
SEARCH_K=10
SEARCH_L_LIST="20 40 60 80 100"
THR=64

# ====== Streaming build & search without filters ======
echo "[1] Running streaming scenario (no filters)..."
./build/apps/test_streaming_scenario \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --data_path "$DATA_PATH" \
  --index_path_prefix "$INDEX_PREFIX" \
  -R "$R" \
  -L "$L" \
  --alpha "$ALPHA" \
  --insert_threads "$INS_THR" \
  --consolidate_threads "$CONS_THR" \
  --max_points_to_insert "$INSERTS" \
  --active_window "$ACTIVE" \
  --consolidate_interval "$CONS_INT" \
  --start_point_norm 508

echo "[2] Computing ground truth (no filters)..."
./build/apps/utils/compute_groundtruth \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --base_file "${INDEX}.data" \
  --query_file "$QUERY_FILE" \
  --K 100 \
  --gt_file "$GT" \
  --tags_file "${INDEX}.tags"

echo "[3] Searching index (no filters)..."
./build/apps/search_memory_index \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --index_path_prefix "$INDEX" \
  --result_path "$RESULT_PATH" \
  --query_file "$QUERY_FILE" \
  --gt_file "$GT" \
  -K "$SEARCH_K" \
  -L $SEARCH_L_LIST \
  -T "$THR" \
  --dynamic true \
  --tags 1
