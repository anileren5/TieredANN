#!/bin/bash

# Script to merge query splits from 10 splits to 5 splits
# This script wraps the Python script for easier command-line usage

set -e

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Default values (can be overridden by command-line arguments)
DATASET="deep10m"
N_SPLIT_REPEAT="5"
NOISE_RATIO="0.01"
DATA_DIR="data"
DATA_TYPE="float"

# Parse arguments (optional - if provided, override defaults)
if [ $# -ge 1 ]; then
    DATASET="$1"
fi
if [ $# -ge 2 ]; then
    N_SPLIT_REPEAT="$2"
fi
if [ $# -ge 3 ]; then
    NOISE_RATIO="$3"
fi
if [ $# -ge 4 ]; then
    DATA_DIR="$4"
fi
if [ $# -ge 5 ]; then
    DATA_TYPE="$5"
fi

# Check if Python script exists
PYTHON_SCRIPT="scripts/merge_splits.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if input files exist
# Format noise_ratio to match Python script (remove trailing zeros)
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
INPUT_QUERY_FILE="$DATA_DIR/$DATASET/${DATASET}_query_nsplit-10_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
INPUT_GROUNDTRUTH_FILE="$DATA_DIR/$DATASET/${DATASET}_groundtruth_nsplit-10_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"

if [ ! -f "$INPUT_QUERY_FILE" ]; then
    echo "Error: Input query file not found: $INPUT_QUERY_FILE"
    exit 1
fi

if [ ! -f "$INPUT_GROUNDTRUTH_FILE" ]; then
    echo "Error: Input groundtruth file not found: $INPUT_GROUNDTRUTH_FILE"
    exit 1
fi

echo "=========================================="
echo "Merging Splits: 10 -> 5"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Input query file: $INPUT_QUERY_FILE"
echo "Input groundtruth file: $INPUT_GROUNDTRUTH_FILE"
echo "n_split_repeat: $N_SPLIT_REPEAT"
echo "noise_ratio: $NOISE_RATIO"
echo "data_type: $DATA_TYPE"
echo "=========================================="
echo ""

# Run the Python script
python3 "$PYTHON_SCRIPT" \
  --dataset "$DATASET" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --noise_ratio "$NOISE_RATIO" \
  --data_dir "$DATA_DIR" \
  --dtype "$DATA_TYPE"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Error: Failed to merge splits"
    exit 1
fi

echo ""
echo "✓ Success! Merged 10 splits into 5 splits"

