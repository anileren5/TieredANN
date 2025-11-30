#!/bin/bash

# Script to check the status of noisy query generation jobs

# Change to project root
cd "$(dirname "$0")/.." || exit 1

LOGS_DIR="logs/noisy_queries"
PID_FILE="$LOGS_DIR/job_pids.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Jobs may not have been started."
    exit 1
fi

echo "=========================================="
echo "Noisy Query Generation Job Status"
echo "=========================================="
echo ""

# Read PID file and check each job
while IFS=' ' read -r pid dataset data_type metric rest; do
    # Skip comments and empty lines
    [[ "$pid" =~ ^#.*$ ]] && continue
    [[ -z "$pid" ]] && continue
    
    # Check if process is still running
    if ps -p "$pid" > /dev/null 2>&1; then
        status="RUNNING"
        runtime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo "N/A")
    else
        status="COMPLETED/STOPPED"
        runtime="N/A"
    fi
    
    log_file="$LOGS_DIR/${dataset}_noisy_queries.log"
    if [ -f "$log_file" ]; then
        log_size=$(du -h "$log_file" | cut -f1)
        last_line=$(tail -n 1 "$log_file" 2>/dev/null | head -c 80)
    else
        log_size="N/A"
        last_line="Log file not found"
    fi
    
    echo "Dataset: $dataset (dtype: $data_type, metric: $metric)"
    echo "  PID: $pid"
    echo "  Status: $status"
    echo "  Runtime: $runtime"
    echo "  Log size: $log_size"
    echo "  Last log line: $last_line"
    echo ""
done < "$PID_FILE"

echo "=========================================="
echo "To view a specific log:"
echo "  tail -f $LOGS_DIR/<dataset>_noisy_queries.log"
echo "=========================================="

