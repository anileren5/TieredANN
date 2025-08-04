#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="sift"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"
QUERY_PATH="data/$DATASET/noisy_queries_groundtruths/${DATASET}_query.noisy_nrounds-5_noiseratio-0.05.bin" # Concatenated base+noisy queries
GROUNDTRUTH_PATH="./data/$DATASET/noisy_queries_groundtruths/${DATASET}_groundtruth.noisy_nrounds-5_noiseratio-0.05.bin" # Concatenated base+noisy groundtruth
R=64
MEMORY_L=128
DISK_L=128
K=100
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=32
BUILD_THREADS=8
CONSOLIDATE_THREADS=8
DISK_INDEX_PREFIX="./index/${DATASET}/${DATASET}"
DISK_INDEX_ALREADY_BUILT=1
BEAMWIDTH=2
USE_RECONSTRUCTED_VECTORS=0
N_THETA_ESTIMATION_QUERIES=1000
P=0.90
DEVIATION_FACTOR=0.025
N_ITERATION_PER_SPLIT=60 # Number of search iterations per split
N_SPLITS=10 # Number of splits for queries
N_ROUNDS=5 # Number of rounds (base + noisy sets)
SECTOR_LEN=4096
USE_REGIONAL_THETA=1 # Set to 0 to use global theta instead of regional theta
PCA_DIM=16 # Set to desired PCA dimension (e.g., 16)
BUCKETS_PER_DIM=8 # Set to desired number of buckets per PCA dimension (e.g., 4)
MEMORY_INDEX_MAX_POINTS=1000000 # Set to desired max points for memory index
N_ASYNC_INSERT_THREADS=16 # Number of async insert threads
LAZY_THETA_UPDATES=1 # Set to 1 to enable lazy theta updates, 0 for immediate updates
CONSOLIDATION_RATIO=0.25 # Fraction of memory index to evict during consolidation (0.0-1.0)
LRU_ASYNC_THREADS=4 # Number of threads for LRU async operations

# Run the test with all parameters
./build/tests/tiered_index_split_search_noisy \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --R "$R" \
  --memory_L "$MEMORY_L" \
  --disk_L "$DISK_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --build_threads "$BUILD_THREADS" \
  --consolidate_threads "$CONSOLIDATE_THREADS" \
  --disk_index_already_built "$DISK_INDEX_ALREADY_BUILT" \
  --beamwidth "$BEAMWIDTH" \
  --disk_index_prefix "$DISK_INDEX_PREFIX" \
  --use_reconstructed_vectors "$USE_RECONSTRUCTED_VECTORS" \
  --p "$P" \
  --deviation_factor "$DEVIATION_FACTOR" \
  --n_theta_estimation_queries "$N_THETA_ESTIMATION_QUERIES" \
  --n_iteration_per_split "$N_ITERATION_PER_SPLIT" \
  --sector_len "$SECTOR_LEN" \
  --use_regional_theta "$USE_REGIONAL_THETA" \
  --pca_dim "$PCA_DIM" \
  --buckets_per_dim "$BUCKETS_PER_DIM" \
  --memory_index_max_points "$MEMORY_INDEX_MAX_POINTS" \
  --n_splits "$N_SPLITS" \
  --n_rounds "$N_ROUNDS" \
  --n_async_insert_threads "$N_ASYNC_INSERT_THREADS" \
  --lazy_theta_updates "$LAZY_THETA_UPDATES" \
  --consolidation_ratio "$CONSOLIDATION_RATIO" \
  --lru_async_threads "$LRU_ASYNC_THREADS" 