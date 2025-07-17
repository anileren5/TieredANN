#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
R=32
L=128
B=8
M=8
SEARCH_THREADS=32
BUILD_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
N_SEARCH_ITER=50

RESULTS_DIR="results/query_pattern_experiments"
mkdir -p "$RESULTS_DIR"

K_LIST=(1 10 100)

run_for_folder_pair() {
  local QUERY_DIR="$1"
  local GT_DIR="$2"
  local SUBFOLDER_LABEL="$3"
  for QUERY_FILE in "$QUERY_DIR"/*.bin; do
    BASENAME=$(basename "$QUERY_FILE" .bin)
    # Compose groundtruth file name
    if [[ "$BASENAME" == *_no_interleave ]]; then
      if [[ -f "$GT_DIR/${BASENAME}_groundtruth.bin" ]]; then
        GT_FILE="$GT_DIR/${BASENAME}_groundtruth.bin"
      else
        GT_FILE="$GT_DIR/${BASENAME}.bin"
      fi
    else
      GT_FILE="$GT_DIR/${BASENAME}_groundtruth.bin"
    fi
    if [[ -f "$GT_FILE" ]]; then
      for K in "${K_LIST[@]}"; do
        OUT_FILE="$RESULTS_DIR/disk_index_search_${SUBFOLDER_LABEL}_${BASENAME}_K${K}.out"
        echo "Running disk_index_search for:"
        echo "  Query: $QUERY_FILE"
        echo "  Groundtruth: $GT_FILE"
        echo "  K: $K"
        echo "  Output: $OUT_FILE"
        ./build/tests/disk_index_search \
          --data_type "$DATA_TYPE" \
          --data_path "$DATA_PATH" \
          --query_path "$QUERY_FILE" \
          --groundtruth_path "$GT_FILE" \
          --R "$R" \
          --L "$L" \
          --K "$K" \
          --B "$B" \
          --M "$M" \
          --search_threads "$SEARCH_THREADS" \
          --build_threads "$BUILD_THREADS" \
          --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
          --beamwidth "$BEAMWIDTH" \
          --disk_index_prefix "$DISK_INDEX_PREFIX" \
          --n_search_iter "$N_SEARCH_ITER" \
          > "$OUT_FILE" 2>&1
      done
    else
      echo "WARNING: No groundtruth file for $QUERY_FILE (expected $GT_FILE)"
    fi
  done
}

# Run for anil and ioanna subfolders
run_for_folder_pair "data/$DATASET/query/anil" "data/$DATASET/groundtruth/anil" "anil"
run_for_folder_pair "data/$DATASET/query/ioanna" "data/$DATASET/groundtruth/ioanna" "ioanna"

# Run for top-level (non-anil, non-ioanna) files
for QUERY_FILE in data/$DATASET/query/*.bin; do
  [[ "$QUERY_FILE" == *"/anil/"* || "$QUERY_FILE" == *"/ioanna/"* ]] && continue
  BASENAME=$(basename "$QUERY_FILE" .bin)
  if [[ -f "data/$DATASET/groundtruth/${BASENAME}_groundtruth.bin" ]]; then
    GT_FILE="data/$DATASET/groundtruth/${BASENAME}_groundtruth.bin"
  else
    GT_FILE="data/$DATASET/groundtruth/${BASENAME}.bin"
  fi
  if [[ -f "$GT_FILE" ]]; then
    for K in "${K_LIST[@]}"; do
      OUT_FILE="$RESULTS_DIR/disk_index_search_top_${BASENAME}_K${K}.out"
      echo "Running disk_index_search for:"
      echo "  Query: $QUERY_FILE"
      echo "  Groundtruth: $GT_FILE"
      echo "  K: $K"
      echo "  Output: $OUT_FILE"
      ./build/tests/disk_index_search \
        --data_type "$DATA_TYPE" \
        --data_path "$DATA_PATH" \
        --query_path "$QUERY_FILE" \
        --groundtruth_path "$GT_FILE" \
        --R "$R" \
        --L "$L" \
        --K "$K" \
        --B "$B" \
        --M "$M" \
        --search_threads "$SEARCH_THREADS" \
        --build_threads "$BUILD_THREADS" \
        --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
        --beamwidth "$BEAMWIDTH" \
        --disk_index_prefix "$DISK_INDEX_PREFIX" \
        --n_search_iter "$N_SEARCH_ITER" \
        > "$OUT_FILE" 2>&1
    done
  else
    echo "WARNING: No groundtruth file for $QUERY_FILE (expected $GT_FILE)"
  fi
done
