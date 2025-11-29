#!/bin/bash

# Change to project root (go up 3 levels from scripts/run/)
cd "$(dirname "$0")/../../.." || exit 1

# Dataset parameters
DATASET="text_to_image_1m"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/${DATASET}_groundtruth.bin"
K=100
METRIC="inner_product"  # Distance metric: "l2", "cosine", or "inner_product" (use "cosine" for glove dataset)
PROGRESS_INTERVAL=1  # Print progress every N queries (use 1 to print for every query)

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Run the bruteforce backend only experiment
python3 python/benchmarks/bruteforce_backend_only.py \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --K "$K" \
  --metric "$METRIC" \
  --progress_interval "$PROGRESS_INTERVAL"

