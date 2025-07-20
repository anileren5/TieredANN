#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="spacev_1m"
DATA_TYPE="int8"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/${DATASET}_groundtruth.bin"
R=64
MEMORY_L=256
DISK_L=256
K=100
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=32
BUILD_THREADS=8
CONSOLIDATE_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
USE_RECONSTRUCTED_VECTORS=0
N_THETA_ESTIMATION_QUERIES=1000
P=0.90
DEVIATION_FACTOR=0.00
N_SEARCH_ITER=50
SECTOR_LEN=4096
EAGER_THETA_UPDATE=true

# Run the test with all parameters
./build/tests/tiered_index_search \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --R "$R" \
  --memory_L "$MEMORY_L" \
  --disk_L "$DISK_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --consolidate_threads "$CONSOLIDATE_THREADS" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --beamwidth "$BEAMWIDTH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
  --use_reconstructed_vectors "$USE_RECONSTRUCTED_VECTORS" \
  --p "$P" \
  --deviation_factor "$DEVIATION_FACTOR" \
  --n_theta_estimation_queries "$N_THETA_ESTIMATION_QUERIES" \
  --n_search_iter "$N_SEARCH_ITER" \
  --sector_len "$SECTOR_LEN" \
  --eager_theta_update "$EAGER_THETA_UPDATE"
