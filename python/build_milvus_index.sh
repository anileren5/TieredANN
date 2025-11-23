#!/bin/bash

# Script to build Milvus index from binary data files

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Configuration
DATASET="sift"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
COLLECTION_NAME="vectors"  # Milvus collection name

# Detect if running inside Docker and set Milvus host/port accordingly
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    MILVUS_HOST="milvus"  # Docker internal network
    MILVUS_PORT=19530
else
    MILVUS_HOST="localhost"  # Local machine
    MILVUS_PORT=19530
fi

# Allow override via environment variables
if [ -n "$MILVUS_HOST_ENV" ]; then
    MILVUS_HOST="$MILVUS_HOST_ENV"
fi
if [ -n "$MILVUS_PORT_ENV" ]; then
    MILVUS_PORT="$MILVUS_PORT_ENV"
fi

# Options
RECREATE=true  # Set to true to recreate collection even if it exists

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Build Milvus index
echo "Building Milvus index..."
echo "Data file: $DATA_PATH"
echo "Collection name: $COLLECTION_NAME"
echo "Milvus host: $MILVUS_HOST"
echo "Milvus port: $MILVUS_PORT"

RECREATE_FLAG=""
if [ "$RECREATE" = true ]; then
    RECREATE_FLAG="--recreate"
    echo "Recreating collection..."
fi

python3 python/build_milvus_index.py \
    --data_path "$DATA_PATH" \
    --collection_name "$COLLECTION_NAME" \
    --milvus_host "$MILVUS_HOST" \
    --milvus_port "$MILVUS_PORT" \
    $RECREATE_FLAG

echo ""
echo "Milvus index build completed!"


