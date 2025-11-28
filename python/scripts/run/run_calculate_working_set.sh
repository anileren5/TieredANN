#!/bin/bash

# Change to project root (go up 3 levels from scripts/run/)
cd "$(dirname "$0")/../../.." || exit 1

# Dataset parameters
DATASET="glove"
GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth.bin"
K=100
N_SPLITS=10

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Run the working set calculation
python3 python/benchmarks/calculate_working_set.py \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --n_splits "$N_SPLITS" \
  --K "$K"

