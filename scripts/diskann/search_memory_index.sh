#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
R=32
L=128
K=100
ALPHA=1.2
BUILD_THREADS=4
SEARCH_THREADS=32

./build/tests/search_memory_index \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --alpha "$ALPHA" \
  --build_threads "$BUILD_THREADS" \
  --search_threads "$SEARCH_THREADS"
