#!/bin/bash

# Script to run QVCache split search experiment with Pinecone backend

# Change to project root (go up 3 levels from scripts/run/)
cd "$(dirname "$0")/../../.." || exit 1

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/${DATASET}_groundtruth.bin"
K=100

# ============================================================================
# CLOUD PINECONE CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
# ============================================================================
# Your Pinecone API Key (starts with "pcsk_")
# Get this from your Pinecone dashboard: https://app.pinecone.io/
PINECONE_API_KEY="pcsk_5A9fQ3_9rJt2c14Ugs1yA99yRzARcNhPqbLxc99S18NVL863yerADFekcqwHf8RxXbtkcW"

# Your Pinecone region/environment (e.g., "us-east-1", "us-west-2", "eu-west-1")
# Check your Pinecone dashboard for the correct region
PINECONE_ENVIRONMENT="us-east-1"

# Your Pinecone index name (check your Pinecone dashboard)
INDEX_NAME="pinecone-sift-index"

# Not needed for cloud Pinecone (only used for local Docker setup)
PINECONE_HOST=""

# Validate API key is set
if [ "$PINECONE_API_KEY" = "YOUR_PINECONE_API_KEY_HERE" ] || [ -z "$PINECONE_API_KEY" ]; then
    echo "ERROR: Pinecone API key is not set!"
    echo "Please update PINECONE_API_KEY in this script with your actual API key from https://app.pinecone.io/"
    exit 1
fi

# Experiment parameters
N_ITERATION_PER_SPLIT=5 # Number of search iterations per split
N_SPLITS=30 # Number of splits for queries
N_ROUNDS=1 # Number of rounds to repeat all splits

# Tiered index parameters
PCA_PREFIX="./index/${DATASET}/${DATASET}"
R=64
MEMORY_L=128  
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=24
BUILD_THREADS=8
BEAMWIDTH=2
USE_RECONSTRUCTED_VECTORS=0
P=0.90
DEVIATION_FACTOR=0.025
USE_REGIONAL_THETA=True # Set to False to use global theta instead of regional theta
PCA_DIM=16 # Set to desired PCA dimension (e.g., 16)
BUCKETS_PER_DIM=8 # Set to desired number of buckets per PCA dimension (e.g., 4)
MEMORY_INDEX_MAX_POINTS=200000 # Set to desired max points for memory index
N_ASYNC_INSERT_THREADS=16 # Number of async insert threads
LAZY_THETA_UPDATES=True # Set to True to enable lazy theta updates, False for immediate updates
NUMBER_OF_MINI_INDEXES=4 # Number of mini indexes for shadow cycling
SEARCH_MINI_INDEXES_IN_PARALLEL=False # Set to True to search mini indexes in parallel
MAX_SEARCH_THREADS=32 # Maximum threads for parallel search
SEARCH_STRATEGY="SEQUENTIAL_LRU_ADAPTIVE" # Search strategy: SEQUENTIAL_LRU_STOP_FIRST_HIT, SEQUENTIAL_LRU_ADAPTIVE, SEQUENTIAL_ALL, PARALLEL

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Check if required files exist
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$QUERY_PATH" ]; then
    echo "Error: Query file not found: $QUERY_PATH"
    exit 1
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    exit 1
fi

# Run the Python test with all parameters
echo "Running QVCache split search experiment with Pinecone backend..."
echo "Pinecone index: $INDEX_NAME"
echo "API key: ${PINECONE_API_KEY:0:10}..."  # Show first 10 chars only

ENV_FLAG=""
if [ -n "$PINECONE_ENVIRONMENT" ]; then
    ENV_FLAG="--environment $PINECONE_ENVIRONMENT"
    echo "Environment: $PINECONE_ENVIRONMENT"
fi

HOST_FLAG=""
if [ -n "$PINECONE_HOST" ]; then
    HOST_FLAG="--host $PINECONE_HOST"
    echo "Host: $PINECONE_HOST"
fi

echo ""

python3 python/benchmarks/qvcache_split_search_pinecone_backend.py \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --pca_prefix "$PCA_PREFIX" \
  --index_name "$INDEX_NAME" \
  --api_key "$PINECONE_API_KEY" \
  $ENV_FLAG \
  $HOST_FLAG \
  --R "$R" \
  --memory_L "$MEMORY_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --beamwidth "$BEAMWIDTH" \
  --use_reconstructed_vectors "$USE_RECONSTRUCTED_VECTORS" \
  --p "$P" \
  --deviation_factor "$DEVIATION_FACTOR" \
  --n_iteration_per_split "$N_ITERATION_PER_SPLIT" \
  --use_regional_theta "$USE_REGIONAL_THETA" \
  --pca_dim "$PCA_DIM" \
  --buckets_per_dim "$BUCKETS_PER_DIM" \
  --memory_index_max_points "$MEMORY_INDEX_MAX_POINTS" \
  --n_splits "$N_SPLITS" \
  --n_rounds "$N_ROUNDS" \
  --n_async_insert_threads "$N_ASYNC_INSERT_THREADS" \
  --lazy_theta_updates "$LAZY_THETA_UPDATES" \
  --number_of_mini_indexes "$NUMBER_OF_MINI_INDEXES" \
  --search_mini_indexes_in_parallel "$SEARCH_MINI_INDEXES_IN_PARALLEL" \
  --max_search_threads "$MAX_SEARCH_THREADS" \
  --search_strategy "$SEARCH_STRATEGY"

