#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DIST_FN="l2"
DATA_PATH="data/$DATASET/${DATASET}_learn.bin"
QUERY_FILE="data/$DATASET/${DATASET}_query.bin"
INDEX_PREFIX="index/$DATASET/${DATASET}"
RESULT_PATH="result/$DATASET/${DATASET}"
DELETES=25000
INSERTS=75000
DELETES_AFTER=50000
PTS_PER_CHECKPOINT=10000
BEGIN=0
THR=64
R=64
L=300
ALPHA=1.2
TOP_K=100
SEARCH_K=10
SEARCH_L_LIST="20 40 60 80 100"

INDEX="${INDEX_PREFIX}.after-concurrent-delete-del${DELETES}-${INSERTS}"
GT_FILE="data/$DATASET/${DATASET}_groundtruth.bin"
TAGS_FILE="${INDEX}.tags"
INDEX_DATA_FILE="${INDEX}.data"

# Step 1: Insertions and deletions
./build/apps/test_insert_deletes_consolidate \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --data_path "$DATA_PATH" \
  --index_path_prefix "$INDEX_PREFIX" \
  -R "$R" \
  -L "$L" \
  --alpha "$ALPHA" \
  -T "$THR" \
  --points_to_skip 0 \
  --max_points_to_insert "$INSERTS" \
  --beginning_index_size "$BEGIN" \
  --points_per_checkpoint "$PTS_PER_CHECKPOINT" \
  --checkpoints_per_snapshot 0 \
  --points_to_delete_from_beginning "$DELETES" \
  --start_deletes_after "$DELETES_AFTER" \
  --do_concurrent true \
  --start_point_norm 508

# Step 2: Compute ground truth
./build/apps/utils/compute_groundtruth \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --base_file "$INDEX_DATA_FILE" \
  --query_file "$QUERY_FILE" \
  --K "$TOP_K" \
  --gt_file "$GT_FILE" \
  --tags_file "$TAGS_FILE"

# Step 3: Search dynamic index
./build/apps/search_memory_index \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --index_path_prefix "$INDEX" \
  --result_path "$RESULT_PATH" \
  --query_file "$QUERY_FILE" \
  --gt_file "$GT_FILE" \
  -K "$SEARCH_K" \
  -L $SEARCH_L_LIST \
  -T "$THR" \
  --dynamic true \
  --tags 1