#!/bin/bash

# Script to run QVCache benchmark experiment with SPTAG (SPANN) backend

set -e

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# Define variables
DATASET="siftsmall"
DATA_TYPE="float"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"

# Noisy query parameters
N_SPLIT=10
N_SPLIT_REPEAT=5
NOISE_RATIO=0.01

# Construct query and groundtruth paths based on noisy query parameters
NOISE_STR=$(echo "$NOISE_RATIO" | sed 's/\.0*$//;s/\.$//')
QUERY_PATH="data/$DATASET/${DATASET}_query_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"

# QVCache parameters (memory-only, no disk index needed with SPTAG backend)
MEMORY_L=32
K=10
B=8
M=8
ALPHA=1.2
SEARCH_THREADS=24
USE_RECONSTRUCTED_VECTORS=0
P=0.90
DEVIATION_FACTOR=0.075
SECTOR_LEN=4096
USE_REGIONAL_THETA=1
PCA_DIM=16
BUCKETS_PER_DIM=8
MEMORY_INDEX_MAX_POINTS=30000
N_ASYNC_INSERT_THREADS=16
LAZY_THETA_UPDATES=1
NUMBER_OF_MINI_INDEXES=4
SEARCH_MINI_INDEXES_IN_PARALLEL=false
MAX_SEARCH_THREADS=32
SEARCH_STRATEGY="SEQUENTIAL_LRU_ADAPTIVE"
METRIC="l2"

# SPTAG server configuration
SPTAG_SERVER_ADDR="${SPTAG_SERVER_ADDR:-sptag}"
SPTAG_SERVER_PORT="${SPTAG_SERVER_PORT:-8000}"
VECTOR_DIM=128

# Check if query file exists (fallback to regular query file if noisy query doesn't exist)
if [ ! -f "$QUERY_PATH" ]; then
    echo "Warning: Noisy query file not found: $QUERY_PATH"
    echo "Falling back to regular query file..."
    QUERY_PATH="data/$DATASET/${DATASET}_query.bin"
    GROUNDTRUTH_PATH="data/$DATASET/${DATASET}_groundtruth.bin"
fi

if [ ! -f "$QUERY_PATH" ]; then
    echo "Error: Query file not found: $QUERY_PATH"
    exit 1
fi

if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "Error: Groundtruth file not found: $GROUNDTRUTH_PATH"
    exit 1
fi

echo "=========================================="
echo "QVCache Benchmark - SPTAG Backend"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Query file: $QUERY_PATH"
echo "Groundtruth file: $GROUNDTRUTH_PATH"
echo "SPTAG Server: $SPTAG_SERVER_ADDR:$SPTAG_SERVER_PORT"
echo "=========================================="
echo ""

# Run the benchmark with all parameters and measure memory usage
# Monitor memory using /proc/<pid>/status VmHWM (High Water Mark)

# Start benchmark in background
./build/tests/qvcache_benchmark_sptag \
  --data_type "$DATA_TYPE" \
  --data_path "$DATA_PATH" \
  --query_path "$QUERY_PATH" \
  --groundtruth_path "$GROUNDTRUTH_PATH" \
  --memory_L "$MEMORY_L" \
  --K "$K" \
  --B "$B" \
  --M "$M" \
  --alpha "$ALPHA" \
  --search_threads "$SEARCH_THREADS" \
  --use_reconstructed_vectors "$USE_RECONSTRUCTED_VECTORS" \
  --p "$P" \
  --deviation_factor "$DEVIATION_FACTOR" \
  --sector_len "$SECTOR_LEN" \
  --use_regional_theta "$USE_REGIONAL_THETA" \
  --pca_dim "$PCA_DIM" \
  --buckets_per_dim "$BUCKETS_PER_DIM" \
  --memory_index_max_points "$MEMORY_INDEX_MAX_POINTS" \
  --n_splits "$N_SPLIT" \
  --n_split_repeat "$N_SPLIT_REPEAT" \
  --n_async_insert_threads "$N_ASYNC_INSERT_THREADS" \
  --lazy_theta_updates "$LAZY_THETA_UPDATES" \
  --number_of_mini_indexes "$NUMBER_OF_MINI_INDEXES" \
  --search_mini_indexes_in_parallel "$SEARCH_MINI_INDEXES_IN_PARALLEL" \
  --max_search_threads "$MAX_SEARCH_THREADS" \
  --search_strategy "$SEARCH_STRATEGY" \
  --metric "$METRIC" \
  --sptag_server_addr "$SPTAG_SERVER_ADDR" \
  --sptag_server_port "$SPTAG_SERVER_PORT" \
  --vector_dim "$VECTOR_DIM" &

BENCHMARK_PID=$!

# Monitor VmHWM (High Water Mark - peak RSS) while process is running
MAX_RSS=0
while kill -0 "$BENCHMARK_PID" 2>/dev/null; do
    if [ -f "/proc/$BENCHMARK_PID/status" ]; then
        # VmHWM is the peak RSS maintained by kernel
        current_rss=$(grep "^VmHWM:" "/proc/$BENCHMARK_PID/status" 2>/dev/null | awk '{print $2}')
        if [ -n "$current_rss" ] && [ "$current_rss" -gt "$MAX_RSS" ]; then
            MAX_RSS=$current_rss
        fi
    fi
    sleep 0.01
done

# Wait for benchmark to complete and get exit code
wait $BENCHMARK_PID
BENCHMARK_EXIT_CODE=$?

# Get final VmHWM (read quickly before /proc entry is cleaned up)
if [ -f "/proc/$BENCHMARK_PID/status" ]; then
    final_rss=$(grep "^VmHWM:" "/proc/$BENCHMARK_PID/status" 2>/dev/null | awk '{print $2}')
    if [ -n "$final_rss" ] && [ "$final_rss" -gt "$MAX_RSS" ]; then
        MAX_RSS=$final_rss
    fi
fi

if [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    exit $BENCHMARK_EXIT_CODE
fi

# Display memory statistics
echo ""
echo "=========================================="
echo "Memory Usage Statistics"
echo "=========================================="
if [ -n "$MAX_RSS" ] && [ "$MAX_RSS" -gt 0 ]; then
    MAX_RSS_MB=$((MAX_RSS / 1024))
    echo "Maximum resident set size (RSS): ${MAX_RSS} KB (${MAX_RSS_MB} MB)"
else
    echo "Warning: Could not determine maximum memory usage"
fi

