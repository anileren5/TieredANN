#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
R=32
L=128
K=100
ALPHA=1.2
BUILD_THREADS=4
SEARCH_THREADS=32

# Define an array of query paths
QUERY_PATHS=(
  "data/$DATASET/${DATASET}_query.bin"
  "data/$DATASET/${DATASET}_clustered_query2.bin"
  "data/$DATASET/${DATASET}_clustered_query3.bin"
)

# Step 1: Run with different query paths
for QUERY_PATH in "${QUERY_PATHS[@]}"; do
  # Extract a base name for the output file
  BASENAME=$(basename "$QUERY_PATH" .bin)
  OUTPUT_FILE="results/${BASENAME}.out"

  # Make sure the output directory exists
  mkdir -p results

  echo "Running search with query path: $QUERY_PATH"
  ./build/apps/only_search \
    --data_type "$DATA_TYPE" \
    --data_path "$DATA_PATH" \
    --query_path "$QUERY_PATH" \
    --R "$R" \
    --L "$L" \
    --K "$K" \
    --alpha "$ALPHA" \
    --build_threads "$BUILD_THREADS" \
    --search_threads "$SEARCH_THREADS" \
    > "$OUTPUT_FILE" 2>&1

  echo "Output saved to $OUTPUT_FILE"
done
