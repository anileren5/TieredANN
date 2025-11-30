#!/bin/bash

# Script to run generate_noisy_queries.sh for multiple datasets in parallel
# Each job runs in the background with nohup so it survives SSH disconnection
# Jobs will continue running even after SSH disconnection

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Configuration: dataset, data_type, metric
declare -a CONFIGS=(
    "bigann:uint8:l2"
    "deep1m:float:l2"
    "deep10m:float:l2"
    "gist:float:l2"
    "glove:float:cosine"
    "sift:float:l2"
    "siftsmall:float:l2"
    "spacev_1m:int8:l2"
    "text_to_image_1m:float:l2"
)

# Parameters
N_SPLIT=10
N_SPLIT_REPEAT=5
NOISE_RATIO=0.01
RANDOM_SEED=42
DATA_DIR="data"
K=100

# Create logs directory
LOGS_DIR="logs/noisy_queries"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "Running Noisy Query Generation for All Datasets"
echo "=========================================="
echo "n_split: $N_SPLIT"
echo "n_split_repeat: $N_SPLIT_REPEAT"
echo "noise_ratio: $NOISE_RATIO"
echo "random_seed: $RANDOM_SEED"
echo "K: $K"
echo "Total datasets: ${#CONFIGS[@]}"
echo "=========================================="
echo ""

# Array to store PIDs
declare -a PIDS=()

# Run each configuration in parallel
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r dataset data_type metric <<< "$config"
    
    echo "Starting job for: $dataset (dtype: $data_type, metric: $metric)"
    
    # Create log file for this dataset
    LOG_FILE="$LOGS_DIR/${dataset}_noisy_queries.log"
    
    # Run in background with nohup, redirecting output to log file
    nohup bash scripts/generate_noisy_queries.sh \
        "$dataset" \
        "$N_SPLIT" \
        "$N_SPLIT_REPEAT" \
        "$NOISE_RATIO" \
        "$RANDOM_SEED" \
        "$DATA_DIR" \
        "$data_type" \
        "$K" \
        "$metric" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    
    echo "  PID: $PID"
    echo "  Log: $LOG_FILE"
    echo ""
done

# Save PIDs to file for later reference
PID_FILE="$LOGS_DIR/job_pids.txt"
echo "# PIDs for noisy query generation jobs started on $(date)" > "$PID_FILE"
for i in "${!CONFIGS[@]}"; do
    IFS=':' read -r dataset data_type metric <<< "${CONFIGS[$i]}"
    echo "${PIDS[$i]} $dataset $data_type $metric" >> "$PID_FILE"
done

echo "=========================================="
echo "All jobs started!"
echo "=========================================="
echo "Total jobs: ${#PIDS[@]}"
echo "PID file: $PID_FILE"
echo "Logs directory: $LOGS_DIR"
echo ""
echo "To check job status:"
echo "  ps -p ${PIDS[*]}"
echo ""
echo "To monitor logs:"
echo "  tail -f $LOGS_DIR/*.log"
echo ""
echo "To check if jobs are still running:"
echo "  cat $PID_FILE"
echo ""
echo "Jobs will continue running even after SSH disconnection."
echo "You can safely disconnect now."

