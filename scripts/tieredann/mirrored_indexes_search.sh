#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/${DATASET}_groundtruth.bin"
R=32
L=128
K=100
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=32
BUILD_THREADS=8
INSERT_THREADS=8
CONSOLIDATE_THREADS=8
DISTANCE_METRIC="l2"
SINGLE_FILE_INDEX=0
NUM_NODES_TO_CACHE=500
BEAMWIDTH=2
RESULTS_PREFIX="./results/${DATASET}/${DATASET}"
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
TAGS_ENABLED=0
DISK_INDEX_ALREADY_BUILT=1
HIT_RATE=0.90

# Run the test with all parameters
./build/tests/mirrored_indexes_search \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --insert_threads "$INSERT_THREADS" \
  --consolidate_threads "$CONSOLIDATE_THREADS" \
  --distance_metric "$DISTANCE_METRIC" \
  --single_file_index "$SINGLE_FILE_INDEX" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --num_nodes_to_cache "$NUM_NODES_TO_CACHE" \
  --beamwidth "$BEAMWIDTH" \
  --hit_rate "$HIT_RATE" \
  --results_prefix "$RESULTS_PREFIX" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
  --tags_enabled "$TAGS_ENABLED"
