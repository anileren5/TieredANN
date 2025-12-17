#!/bin/bash

# Script to visualize all backend and QVCache log files
# Usage: ./visualize_all_logs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_SCRIPT="${SCRIPT_DIR}/visualize_logs.py"
OUTPUT_BASE="${SCRIPT_DIR}/plots"

# Function to process log files in a directory
process_directory() {
    local dir="$1"
    local dir_name=$(basename "$dir")
    local backend_log_pattern="${dir}/backend_only_*.log"
    
    echo "Processing directory: ${dir_name}"
    echo "=========================================="
    
    # Find all backend_only log files
    for backend_log in ${backend_log_pattern}; do
        # Check if file exists (glob might not match)
        if [ ! -f "$backend_log" ]; then
            continue
        fi
        
        # Extract the experiment name (e.g., "faiss" from "backend_only_faiss.log")
        local basename=$(basename "$backend_log")
        local experiment_name="${basename#backend_only_}"
        experiment_name="${experiment_name%.log}"
        
        # Construct the matching QVCache log file path
        local qvcache_log="${dir}/qvcache_${experiment_name}.log"
        
        # Check if QVCache log exists
        if [ ! -f "$qvcache_log" ]; then
            echo "Warning: QVCache log not found for ${experiment_name}: ${qvcache_log}"
            echo "Skipping ${experiment_name}..."
            echo ""
            continue
        fi
        
        # Create output directory
        local output_dir="${OUTPUT_BASE}/${dir_name}/${experiment_name}"
        mkdir -p "$output_dir"
        
        echo "Processing: ${experiment_name}"
        echo "  Backend log: ${backend_log}"
        echo "  QVCache log: ${qvcache_log}"
        echo "  Output dir: ${output_dir}"
        
        # Run visualization script
        python3 "$VISUALIZE_SCRIPT" \
            --backend_log "$backend_log" \
            --qvcache_log "$qvcache_log" \
            --output "$output_dir" \
            --n_repeat 5
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully generated plots for ${experiment_name}"
        else
            echo "  ✗ Failed to generate plots for ${experiment_name}"
        fi
        echo ""
    done
}

# Main execution
echo "Starting visualization of all log files..."
echo "=========================================="
echo ""

# Process backend_experiments directory
if [ -d "${SCRIPT_DIR}/backend_experiments" ]; then
    process_directory "${SCRIPT_DIR}/backend_experiments"
else
    echo "Warning: backend_experiments directory not found"
fi

# Process dataset_experiments directory
if [ -d "${SCRIPT_DIR}/dataset_experiments" ]; then
    process_directory "${SCRIPT_DIR}/dataset_experiments"
else
    echo "Warning: dataset_experiments directory not found"
fi

echo "=========================================="
echo "Visualization complete!"
echo "All plots saved to: ${OUTPUT_BASE}"

