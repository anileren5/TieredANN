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
ALPHA=1.2
SEARCH_THREADS=32
BUILD_THREADS=8
CONSOLIDATE_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
N_THETA_ESTIMATION_QUERIES=1000
N_SEARCH_ITER=50

RESULTS_DIR="results/parameter_experiments"
mkdir -p "$RESULTS_DIR"

K_LIST=(1 10 100)
DEVIATION_LIST=(-0.10 -0.05 0.00 0.05 0.10)
P_LIST=(0.70 0.80 0.90 0.95)
USE_RECONSTRUCTED_VECTORS_LIST=(0 1)

QUERY_DIR="data/$DATASET/query/anil"
GT_DIR="data/$DATASET/groundtruth/anil"

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
      for DEVIATION in "${DEVIATION_LIST[@]}"; do
        for P in "${P_LIST[@]}"; do
          for USE_RECONSTRUCTED_VECTORS in "${USE_RECONSTRUCTED_VECTORS_LIST[@]}"; do
            # Format deviation and p for filename (e.g., -0.10 -> m0p10, 0.05 -> 0p05, 0.90 -> 0p90)
            DEV_STR=$(printf "%.2f" "$DEVIATION" | sed 's/-/m/;s/\./p/')
            P_STR=$(printf "%.2f" "$P" | sed 's/\./p/')
            OUT_FILE="$RESULTS_DIR/tiered_index_search_anil_${BASENAME}_K${K}_DEV${DEV_STR}_P${P_STR}_URV${USE_RECONSTRUCTED_VECTORS}.out"
            echo "Running tiered_index_search for:"
            echo "  Query: $QUERY_FILE"
            echo "  Groundtruth: $GT_FILE"
            echo "  K: $K"
            echo "  DEVIATION_FACTOR: $DEVIATION"
            echo "  P: $P"
            echo "  USE_RECONSTRUCTED_VECTORS: $USE_RECONSTRUCTED_VECTORS"
            echo "  Output: $OUT_FILE"
            ./build/tests/tiered_index_search \
              --data_type "$DATA_TYPE" \
              --data_path "$DATA_PATH" \
              --query_path "$QUERY_FILE" \
              --groundtruth_path "$GT_FILE" \
              --R "$R" \
              --L "$L" \
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
              --deviation_factor "$DEVIATION" \
              --n_theta_estimation_queries "$N_THETA_ESTIMATION_QUERIES" \
              --n_search_iter "$N_SEARCH_ITER" \
              > "$OUT_FILE" 2>&1
          done
        done
      done
    done
  else
    echo "WARNING: No groundtruth file for $QUERY_FILE (expected $GT_FILE)"
  fi
done