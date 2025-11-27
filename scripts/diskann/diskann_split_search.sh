#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/${DATASET}_groundtruth.bin"
R=64
L=128
K=100
ALPHA=1.2
SEARCH_THREADS=24
BUILD_THREADS=8
MAX_POINTS=1000000 # Maximum number of points the index can hold (should be >= number of points in base file to build full index)
N_ITERATION_PER_SPLIT=5 # Number of search iterations per split
N_SPLITS=30 # Number of splits for queries
N_ROUNDS=1 # Number of rounds to repeat all splits
METRIC="l2" # Distance metric: l2, cosine

# Run the test with all parameters
./build/tests/diskann_split_search \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --max_points "$MAX_POINTS" \
  --n_iteration_per_split "$N_ITERATION_PER_SPLIT" \
  --n_splits "$N_SPLITS" \
  --n_rounds "$N_ROUNDS" \
  --metric "$METRIC"

