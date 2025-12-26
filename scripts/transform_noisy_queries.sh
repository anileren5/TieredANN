#!/bin/bash

# Script to transform noisy queries using transform_noisy_queries.py
# This script wraps the Python script for easier command-line usage

set -e

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Check if Python script exists
PYTHON_SCRIPT="scripts/transform_noisy_queries.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Default values (can be overridden by command-line arguments)
DATASET="glove"
SOURCE_N_SPLIT="10"
SOURCE_N_REPEAT="5"
SOURCE_NOISE="0.01"
TARGET_N_SPLIT="10"
TARGET_N_REPEAT="20"
TARGET_NOISE="0.01"
DATA_DIR="data"
DATA_TYPE="float"

# Parse arguments (optional - if provided, override defaults)
if [ $# -ge 1 ]; then
    DATASET="$1"
fi
if [ $# -ge 2 ]; then
    SOURCE_N_SPLIT="$2"
fi
if [ $# -ge 3 ]; then
    SOURCE_N_REPEAT="$3"
fi
if [ $# -ge 4 ]; then
    SOURCE_NOISE="$4"
fi
if [ $# -ge 5 ]; then
    TARGET_N_SPLIT="$5"
fi
if [ $# -ge 6 ]; then
    TARGET_N_REPEAT="$6"
fi
if [ $# -ge 7 ]; then
    TARGET_NOISE="$7"
fi
if [ $# -ge 8 ]; then
    DATA_DIR="$8"
fi
if [ $# -ge 9 ]; then
    DATA_TYPE="$9"
fi

# Check if source files exist
SOURCE_NOISE_STR=$(echo "$SOURCE_NOISE" | sed 's/\.0*$//;s/\.$//')
SOURCE_QUERY_FILE="$DATA_DIR/$DATASET/${DATASET}_query_nsplit-${SOURCE_N_SPLIT}_nrepeat-${SOURCE_N_REPEAT}_noise-${SOURCE_NOISE_STR}.bin"
SOURCE_GT_FILE="$DATA_DIR/$DATASET/${DATASET}_groundtruth_nsplit-${SOURCE_N_SPLIT}_nrepeat-${SOURCE_N_REPEAT}_noise-${SOURCE_NOISE_STR}.bin"

if [ ! -f "$SOURCE_QUERY_FILE" ]; then
    echo "Error: Source query file not found: $SOURCE_QUERY_FILE"
    exit 1
fi

if [ ! -f "$SOURCE_GT_FILE" ]; then
    echo "Error: Source groundtruth file not found: $SOURCE_GT_FILE"
    exit 1
fi

echo "=========================================="
echo "Transforming Noisy Queries"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Source: n_split=$SOURCE_N_SPLIT, n_repeat=$SOURCE_N_REPEAT, noise=$SOURCE_NOISE"
echo "Target: n_split=$TARGET_N_SPLIT, n_repeat=$TARGET_N_REPEAT, noise=$TARGET_NOISE"
echo "Data dir: $DATA_DIR"
echo "Data type: $DATA_TYPE"
echo "=========================================="
echo ""

# Run the Python script
python3 "$PYTHON_SCRIPT" \
  --dataset "$DATASET" \
  --source_n_split "$SOURCE_N_SPLIT" \
  --source_n_split_repeat "$SOURCE_N_REPEAT" \
  --source_noise_ratio "$SOURCE_NOISE" \
  --target_n_split "$TARGET_N_SPLIT" \
  --target_n_split_repeat "$TARGET_N_REPEAT" \
  --target_noise_ratio "$TARGET_NOISE" \
  --data_dir "$DATA_DIR" \
  --dtype "$DATA_TYPE"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Error: Failed to transform noisy queries"
    exit 1
fi

echo ""
echo "✓ Success! Transformed noisy queries and groundtruth"

