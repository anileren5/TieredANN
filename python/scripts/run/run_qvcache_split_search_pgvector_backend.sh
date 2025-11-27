#!/bin/bash

# Script to run QVCache split search experiment with pgvector backend

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
# POSTGRESQL CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
# ============================================================================
# PostgreSQL connection settings
# For Docker: Use service name "postgres" and default credentials
# For local: Use "localhost" and your PostgreSQL credentials
DB_HOST="${DB_HOST:-localhost}"  # Default: localhost (use "postgres" for Docker)
DB_PORT="${DB_PORT:-5432}"       # Default PostgreSQL port
DB_NAME="${DB_NAME:-postgres}"   # Default database name
DB_USER="${DB_USER:-postgres}"   # Default user
DB_PASSWORD="${DB_PASSWORD:-postgres}"  # Default password

# Table name is derived from dataset name
TABLE_NAME="$DATASET"

# Detect if running inside Docker and set PostgreSQL host accordingly
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    DB_HOST="postgres"  # Docker internal network
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
NUMBER_OF_MINI_INDEXES=4 # Number of mini indexes
SEARCH_MINI_INDEXES_IN_PARALLEL=True # Set to True to search mini indexes in parallel
MAX_SEARCH_THREADS=32 # Maximum number of search threads
SEARCH_STRATEGY="SEQUENTIAL_LRU_ADAPTIVE" # Search strategy: SEQUENTIAL_LRU_STOP_FIRST_HIT, SEQUENTIAL_LRU_ADAPTIVE, SEQUENTIAL_ALL, PARALLEL

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Run experiment
echo "Running QVCache split search experiment with pgvector backend..."
echo "PostgreSQL table: $TABLE_NAME"
echo "PostgreSQL host: $DB_HOST"
echo "PostgreSQL port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""

python3 python/benchmarks/qvcache_split_search_pgvector_backend.py \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --pca_prefix "$PCA_PREFIX" \
  --table_name "$TABLE_NAME" \
  --db_host "$DB_HOST" \
  --db_port "$DB_PORT" \
  --db_name "$DB_NAME" \
  --db_user "$DB_USER" \
  --db_password "$DB_PASSWORD" \
  --R $R \
  --memory_L $MEMORY_L \
  --K $K \
  --B $B \
  --M $M \
  --alpha $ALPHA \
  --build_threads $BUILD_THREADS \
  --search_threads $SEARCH_THREADS \
  --beamwidth $BEAMWIDTH \
  --use_reconstructed_vectors $USE_RECONSTRUCTED_VECTORS \
  --p $P \
  --deviation_factor $DEVIATION_FACTOR \
  --n_iteration_per_split $N_ITERATION_PER_SPLIT \
  --memory_index_max_points $MEMORY_INDEX_MAX_POINTS \
  --use_regional_theta $USE_REGIONAL_THETA \
  --pca_dim $PCA_DIM \
  --buckets_per_dim $BUCKETS_PER_DIM \
  --n_splits $N_SPLITS \
  --n_rounds $N_ROUNDS \
  --n_async_insert_threads $N_ASYNC_INSERT_THREADS \
  --lazy_theta_updates $LAZY_THETA_UPDATES \
  --number_of_mini_indexes $NUMBER_OF_MINI_INDEXES \
  --search_mini_indexes_in_parallel $SEARCH_MINI_INDEXES_IN_PARALLEL \
  --max_search_threads $MAX_SEARCH_THREADS \
  --search_strategy "$SEARCH_STRATEGY" \
  --data_type "$DATA_TYPE"

