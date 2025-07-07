#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
CHUNKS_GROUNDTRUTH_PATH+="data/$DATASET/${DATASET}_chunk_groundtruths/${DATASET}_chunk"
R=32
L=128
K=100
ALPHA=1.2
CHUNK_SIZE=100000
BUILD_THREADS=4
INSERT_THREADS=4
CONSOLIDATE_THREADS=4
SEARCH_THREADS=4

# Step 1: Insertions and deletions
./build/apps/concurrent_search \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --chunks_groundtruth_path "$CHUNKS_GROUNDTRUTH_PATH" \
  --chunk_size "$CHUNK_SIZE" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --alpha "$ALPHA" \
  --build_threads "$BUILD_THREADS" \
  --insert_threads "$INSERT_THREADS" \
  --consolidate_threads "$CONSOLIDATE_THREADS" \
  --search_threads "$SEARCH_THREADS" \
