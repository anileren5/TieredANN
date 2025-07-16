#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/query/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/groundtruth/${DATASET}_groundtruth.bin"
R=32
L=128
K=100
B=8
M=8
SEARCH_THREADS=32
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2

# Run the test with only the required parameters
./build/tests/disk_index_search \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --beamwidth "$BEAMWIDTH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX"
