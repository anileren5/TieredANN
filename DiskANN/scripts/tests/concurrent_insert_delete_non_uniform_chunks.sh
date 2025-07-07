#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/clustered_${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
CHUNKS_GROUNDTRUTH_PATH+="data/$DATASET/clustered_${DATASET}_chunk_groundtruths/${DATASET}_chunk"
R=32
L=128
K=5
ALPHA=1.2
CHUNK_SIZE=100000
BUILD_THREADS=32
INSERT_THREADS=32
CONSOLIDATE_THREADS=32
SEARCH_THREADS=32

# Step 1: Insertions and deletions
./build/apps/concurrent_insert_delete_non_uniform_chunks \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --chunks_groundtruth_path "$CHUNKS_GROUNDTRUTH_PATH" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --alpha "$ALPHA" \
  --build_threads "$BUILD_THREADS" \
  --insert_threads "$INSERT_THREADS" \
  --consolidate_threads "$CONSOLIDATE_THREADS" \
  --search_threads "$SEARCH_THREADS" \
