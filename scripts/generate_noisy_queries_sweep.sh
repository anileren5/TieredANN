#!/bin/bash

# Script to generate noisy queries for all noise ratios from 0 to 1
# Increments by 0.025 at each step (41 total iterations)

# Don't exit on error - continue with next noise ratio
set +e

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Default values (can be overridden by command-line arguments)
DATASET="gist"
N_SPLIT="10"
N_SPLIT_REPEAT="2"
RANDOM_SEED="42"
DATA_DIR="data"
DATA_TYPE="float"
K="100"
METRIC="l2"

# Parse arguments (optional - if provided, override defaults)
if [ $# -ge 1 ]; then
    DATASET="$1"
fi
if [ $# -ge 2 ]; then
    N_SPLIT="$2"
fi
if [ $# -ge 3 ]; then
    N_SPLIT_REPEAT="$3"
fi
if [ $# -ge 4 ]; then
    RANDOM_SEED="$4"
fi
if [ $# -ge 5 ]; then
    DATA_DIR="$5"
fi
if [ $# -ge 6 ]; then
    DATA_TYPE="$6"
fi
if [ $# -ge 7 ]; then
    K="$7"
fi
if [ $# -ge 8 ]; then
    METRIC="$8"
fi

# Check if Python script exists
PYTHON_SCRIPT="scripts/generate_noisy_queries.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if query file exists
QUERY_FILE="$DATA_DIR/$DATASET/${DATASET}_query.bin"
if [ ! -f "$QUERY_FILE" ]; then
    echo "Error: Query file not found: $QUERY_FILE"
    exit 1
fi

# Check if compute_groundtruth binary exists
COMPUTE_GT_BIN="./build/tests/compute_groundtruth"
if [ ! -f "$COMPUTE_GT_BIN" ]; then
    echo "Warning: compute_groundtruth binary not found: $COMPUTE_GT_BIN"
    echo "Groundtruth computation will be skipped. Please build the project first."
    SKIP_GT=true
else
    SKIP_GT=false
fi

# Check if base file exists
DATASET_DIR="$DATA_DIR/$DATASET"
BASE_FILE="$DATASET_DIR/${DATASET}_base.bin"
if [ ! -f "$BASE_FILE" ]; then
    echo "Warning: Base file not found: $BASE_FILE"
    echo "Groundtruth computation will be skipped."
    SKIP_GT=true
fi

echo "=========================================="
echo "Generating Noisy Queries - Full Sweep"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Query file: $QUERY_FILE"
echo "n_split: $N_SPLIT"
echo "n_split_repeat: $N_SPLIT_REPEAT"
echo "random_seed: $RANDOM_SEED"
echo "Noise ratios: 0.000 to 1.000 (step: 0.025)"
echo "Total iterations: 41"
echo "=========================================="
echo ""

# Initialize counters
TOTAL=41
CURRENT=0
SUCCESS=0
FAILED=0

# Loop through noise ratios from 0.000 to 1.000 with 0.025 increments
# Using awk for floating point arithmetic
for i in $(seq 0 40); do
    CURRENT=$((CURRENT + 1))
    # Calculate noise ratio: i * 0.025
    NOISE_RATIO=$(awk "BEGIN {printf \"%.3f\", $i*0.025}")
    
    # Format noise_ratio for filename (remove trailing zeros after decimal)
    NOISE_STR=$(echo "$NOISE_RATIO" | sed -E 's/(\.[0-9]*[1-9])0+$/\1/;s/\.0+$//;s/\.$//')
    
    echo "[$CURRENT/$TOTAL] Processing noise_ratio=$NOISE_RATIO"
    
    # Run the Python script
    python3 "$PYTHON_SCRIPT" \
      --dataset "$DATASET" \
      --n_split "$N_SPLIT" \
      --n_split_repeat "$N_SPLIT_REPEAT" \
      --noise_ratio "$NOISE_RATIO" \
      --random_seed "$RANDOM_SEED" \
      --data_dir "$DATA_DIR" \
      --dtype "$DATA_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Failed to generate noisy queries for noise_ratio=$NOISE_RATIO"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Construct paths for compute_groundtruth
    GENERATED_QUERY_FILE="$DATASET_DIR/${DATASET}_query_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
    OUTPUT_GROUNDTRUTH_FILE=$(echo "$GENERATED_QUERY_FILE" | sed 's/_query_/_groundtruth_/')
    
    # Skip groundtruth if requested or if query file doesn't exist
    if [ "$SKIP_GT" = true ]; then
        echo "  ✓ Generated query file (groundtruth skipped)"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi
    
    if [ ! -f "$GENERATED_QUERY_FILE" ]; then
        echo "  ⚠ Generated query file not found: $GENERATED_QUERY_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Run compute_groundtruth
    "$COMPUTE_GT_BIN" "$BASE_FILE" "$GENERATED_QUERY_FILE" "$OUTPUT_GROUNDTRUTH_FILE" "$DATA_TYPE" "$K" "$METRIC" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success! Query and groundtruth generated for noise_ratio=$NOISE_RATIO"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed to compute groundtruth for noise_ratio=$NOISE_RATIO"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Sweep Complete"
echo "=========================================="
echo "Total iterations: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi

