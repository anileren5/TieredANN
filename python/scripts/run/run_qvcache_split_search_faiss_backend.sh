#!/bin/bash

# Script to run QVCache split search experiment with FAISS backend

# Change to project root (go up 3 levels from scripts/run/)
cd "$(dirname "$0")/../../.." || exit 1

# Dataset parameters
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
GROUNDTRUTH_PATH="./data/$DATASET/${DATASET}_groundtruth.bin"
K=100

# FAISS parameters
INDEX_PATH="./faiss_index.bin"  # Path to FAISS index

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

if [ ! -f "$INDEX_PATH" ]; then
    echo "Error: FAISS index not found: $INDEX_PATH"
    echo "Please build the index first using: ./python/scripts/build_index/build_faiss_index.sh"
    exit 1
fi

# Run the Python test with all parameters
echo "Running QVCache split search experiment with FAISS backend..."
echo "FAISS index: $INDEX_PATH"
echo ""

python3 python/benchmarks/qvcache_split_search_faiss_backend.py \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --pca_prefix "$PCA_PREFIX" \
  --index_path "$INDEX_PATH" \
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

