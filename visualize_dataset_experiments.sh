#!/bin/bash

# Script to visualize all dataset experiment log files
# Usage: ./visualize_dataset_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_SCRIPT="${SCRIPT_DIR}/visualize_logs.py"
DATASET_EXPERIMENTS_DIR="${SCRIPT_DIR}/experiments/dataset_experiments"
OUTPUT_BASE="${SCRIPT_DIR}/plots/dataset_experiments"

# Check if dataset_experiments directory exists
if [ ! -d "$DATASET_EXPERIMENTS_DIR" ]; then
    echo "Error: dataset_experiments directory not found: ${DATASET_EXPERIMENTS_DIR}"
    echo "Please ensure the dataset_experiments directory exists with log files."
    exit 1
fi

echo "Starting visualization of dataset experiment log files..."
echo "=========================================="
echo "Dataset experiments directory: ${DATASET_EXPERIMENTS_DIR}"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# Process each dataset subdirectory
for dataset_dir in "${DATASET_EXPERIMENTS_DIR}"/*; do
    # Check if it's a directory
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi
    
    dataset_name=$(basename "$dataset_dir")
    echo "Processing dataset: ${dataset_name}"
    echo "----------------------------------------"
    
    # Look for backend and QVCache log files
    # Try different naming patterns
    backend_log=""
    qvcache_log=""
    
    # Pattern 1: backend_only_*.log and qvcache_*.log
    backend_log_pattern="${dataset_dir}/backend_only_*.log"
    for log_file in ${backend_log_pattern}; do
        if [ -f "$log_file" ]; then
            backend_log="$log_file"
            # Extract experiment name and find matching QVCache log
            basename=$(basename "$log_file")
            experiment_name="${basename#backend_only_}"
            experiment_name="${experiment_name%.log}"
            qvcache_log="${dataset_dir}/qvcache_${experiment_name}.log"
            break
        fi
    done
    
    # Pattern 2: backend_*.log and qvcache_*.log (without "only")
    if [ -z "$backend_log" ]; then
        backend_log_pattern="${dataset_dir}/backend_*.log"
        for log_file in ${backend_log_pattern}; do
            if [ -f "$log_file" ]; then
                backend_log="$log_file"
                # Extract experiment name and find matching QVCache log
                basename=$(basename "$log_file")
                experiment_name="${basename#backend_}"
                experiment_name="${experiment_name%.log}"
                qvcache_log="${dataset_dir}/qvcache_${experiment_name}.log"
                break
            fi
        done
    fi
    
    # Pattern 3: Just look for any qvcache_*.log and try to find matching backend
    if [ -z "$qvcache_log" ] || [ ! -f "$qvcache_log" ]; then
        qvcache_log_pattern="${dataset_dir}/qvcache_*.log"
        for log_file in ${qvcache_log_pattern}; do
            if [ -f "$log_file" ]; then
                qvcache_log="$log_file"
                # Extract experiment name and find matching backend log
                basename=$(basename "$log_file")
                experiment_name="${basename#qvcache_}"
                experiment_name="${experiment_name%.log}"
                # Try both patterns
                if [ -f "${dataset_dir}/backend_only_${experiment_name}.log" ]; then
                    backend_log="${dataset_dir}/backend_only_${experiment_name}.log"
                elif [ -f "${dataset_dir}/backend_${experiment_name}.log" ]; then
                    backend_log="${dataset_dir}/backend_${experiment_name}.log"
                fi
                break
            fi
        done
    fi
    
    # Check if we found the required log files
    if [ -z "$qvcache_log" ] || [ ! -f "$qvcache_log" ]; then
        echo "  Warning: QVCache log not found for ${dataset_name}"
        echo "  Skipping ${dataset_name}..."
        echo ""
        continue
    fi
    
    # Create output directory
    output_dir="${OUTPUT_BASE}/${dataset_name}"
    mkdir -p "$output_dir"
    
    echo "  QVCache log: ${qvcache_log}"
    if [ -n "$backend_log" ] && [ -f "$backend_log" ]; then
        echo "  Backend log: ${backend_log}"
    else
        echo "  Backend log: (not found, will generate QVCache-only plots)"
        backend_log=""
    fi
    echo "  Output dir: ${output_dir}"
    
    # Run visualization script
    if [ -n "$backend_log" ] && [ -f "$backend_log" ]; then
        # Both logs available
        python3 "$VISUALIZE_SCRIPT" \
            --backend_log "$backend_log" \
            --qvcache_log "$qvcache_log" \
            --output "$output_dir"
    else
        # Only QVCache log available
        python3 "$VISUALIZE_SCRIPT" \
            --qvcache_log "$qvcache_log" \
            --output "$output_dir"
    fi
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully generated plots for ${dataset_name}"
    else
        echo "  ✗ Failed to generate plots for ${dataset_name}"
    fi
    echo ""
done

echo "=========================================="
echo "Visualization complete!"
echo "All plots saved to: ${OUTPUT_BASE}"

