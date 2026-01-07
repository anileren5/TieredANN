#!/bin/bash

# Script to analyze neighbor overlap across repetitions using analyze_neighbor_overlap.py
# This script wraps the Python script for easier command-line usage

set -e

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Check if Python script exists
PYTHON_SCRIPT="scripts/analyze_neighbor_overlap.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Default values (can be overridden by command-line arguments)
DATASET="gist"
N_SPLIT="10"
N_SPLIT_REPEAT="2"
NOISE_RATIO="0.01"
K="10"
DATA_DIR="data"
DATA_TYPE="float"

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
    NOISE_RATIO="$4"
fi
if [ $# -ge 5 ]; then
    K="$5"
fi
if [ $# -ge 6 ]; then
    DATA_DIR="$6"
fi
if [ $# -ge 7 ]; then
    DATA_TYPE="$7"
fi

# Check if groundtruth file exists
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
GT_FILE="$DATA_DIR/$DATASET/${DATASET}_groundtruth_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"

if [ ! -f "$GT_FILE" ]; then
    echo "Error: Groundtruth file not found: $GT_FILE"
    exit 1
fi

echo "=========================================="
echo "Analyzing Neighbor Overlap Across Repetitions"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Groundtruth file: $GT_FILE"
echo "n_split: $N_SPLIT"
echo "n_split_repeat: $N_SPLIT_REPEAT"
echo "noise_ratio: $NOISE_RATIO"
echo "k: $K"
echo "Data dir: $DATA_DIR"
echo "Data type: $DATA_TYPE"
echo "=========================================="
echo ""

# Run the Python script
python3 "$PYTHON_SCRIPT" \
  --dataset "$DATASET" \
  --n_split "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --noise_ratio "$NOISE_RATIO" \
  --k "$K" \
  --data_dir "$DATA_DIR" \
  --dtype "$DATA_TYPE"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Error: Failed to analyze neighbor overlap"
    exit 1
fi

echo ""
echo "✓ Success! Analysis complete"

