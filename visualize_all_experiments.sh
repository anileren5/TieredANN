#!/bin/bash

# Unified script to visualize all experiment log files
# Handles: backend_experiments, dataset_experiments, granularity_experiments, spatial_threshold_experiments, deviation_factor_experiments, and pca_experiments
# Usage: ./visualize_all_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_SCRIPT="${SCRIPT_DIR}/visualize_logs.py"
VISUALIZE_MULTI_SCRIPT="${SCRIPT_DIR}/visualize_logs_multi.py"
EXPERIMENTS_DIR="${SCRIPT_DIR}/experiments"

echo "Starting visualization of all experiment log files..."
echo "=========================================="
echo "Experiments directory: ${EXPERIMENTS_DIR}"
echo ""

# ============================================================================
# Process Backend Experiments
# ============================================================================
BACKEND_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}/backend_experiments"
OUTPUT_BACKEND="${SCRIPT_DIR}/plots/backend_experiments"

if [ -d "$BACKEND_EXPERIMENTS_DIR" ]; then
    echo "Processing Backend Experiments..."
    echo "=========================================="
    echo "Backend experiments directory: ${BACKEND_EXPERIMENTS_DIR}"
    echo "Output directory: ${OUTPUT_BACKEND}"
    echo ""
    
    for backend_dir in "${BACKEND_EXPERIMENTS_DIR}"/*; do
        # Check if it's a directory
        if [ ! -d "$backend_dir" ]; then
            continue
        fi
        
        backend_name=$(basename "$backend_dir")
        echo "Processing backend: ${backend_name}"
        echo "----------------------------------------"
        
        # Look for backend and QVCache log files
        backend_log="${backend_dir}/backend.log"
        qvcache_log="${backend_dir}/qvcache.log"
        
        # Check if we found the required log files
        if [ ! -f "$qvcache_log" ]; then
            echo "  Warning: QVCache log not found for ${backend_name}"
            echo "  Skipping ${backend_name}..."
            echo ""
            continue
        fi
        
        # Create output directory
        output_dir="${OUTPUT_BACKEND}/${backend_name}"
        mkdir -p "$output_dir"
        
        echo "  QVCache log: ${qvcache_log}"
        if [ -f "$backend_log" ]; then
            echo "  Backend log: ${backend_log}"
        else
            echo "  Backend log: (not found, will generate QVCache-only plots)"
            backend_log=""
        fi
        echo "  Output dir: ${output_dir}"
        
        # Run visualization script
        if [ -f "$backend_log" ]; then
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
            echo "  ✓ Successfully generated plots for ${backend_name}"
        else
            echo "  ✗ Failed to generate plots for ${backend_name}"
        fi
        echo ""
    done
    
    echo "Backend experiments visualization complete!"
    echo "Plots saved to: ${OUTPUT_BACKEND}"
    echo ""
fi

# ============================================================================
# Process Dataset Experiments
# ============================================================================
DATASET_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}/dataset_experiments"
OUTPUT_DATASET="${SCRIPT_DIR}/plots/dataset_experiments"

if [ -d "$DATASET_EXPERIMENTS_DIR" ]; then
    echo "Processing Dataset Experiments..."
    echo "=========================================="
    echo "Dataset experiments directory: ${DATASET_EXPERIMENTS_DIR}"
    echo "Output directory: ${OUTPUT_DATASET}"
    echo ""
    
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
        output_dir="${OUTPUT_DATASET}/${dataset_name}"
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
    
    echo "Dataset experiments visualization complete!"
    echo "Plots saved to: ${OUTPUT_DATASET}"
    echo ""
fi

# ============================================================================
# Process Granularity Experiments
# ============================================================================
GRANULARITY_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}/granularity_experiments"
OUTPUT_GRANULARITY="${SCRIPT_DIR}/plots/granularity_experiments"

if [ -d "$GRANULARITY_EXPERIMENTS_DIR" ]; then
    echo "Processing Granularity Experiments..."
    echo "=========================================="
    echo "Granularity experiments directory: ${GRANULARITY_EXPERIMENTS_DIR}"
    echo "Output directory: ${OUTPUT_GRANULARITY}"
    echo ""
    
    # Find all .log files in the granularity_experiments directory
    log_files=()
    for log_file in "${GRANULARITY_EXPERIMENTS_DIR}"/*.log; do
        if [ -f "$log_file" ]; then
            log_files+=("$log_file")
        fi
    done
    
    if [ ${#log_files[@]} -eq 0 ]; then
        echo "  Warning: No log files found in ${GRANULARITY_EXPERIMENTS_DIR}"
        echo "  Skipping granularity experiments..."
        echo ""
    else
        # Sort log files numerically by their basename (e.g., 1.log, 2.log, 4.log, etc.)
        # Create temporary array with numeric prefix for sorting
        declare -A log_map
        for log_file in "${log_files[@]}"; do
            basename=$(basename "$log_file")
            numeric="${basename%.log}"
            # Store full path with numeric key
            log_map["$numeric"]="$log_file"
        done
        
        # Extract and sort numeric keys
        sorted_keys=($(printf '%s\n' "${!log_map[@]}" | sort -n))
        
        # Build sorted arrays
        legends=()
        log_paths=()
        for key in "${sorted_keys[@]}"; do
            log_path="${log_map[$key]}"
            legends+=("$key")
            log_paths+=("$log_path")
        done
        
        if [ ${#log_paths[@]} -gt 0 ]; then
            echo "  Found ${#log_paths[@]} log files:"
            for i in "${!log_paths[@]}"; do
                echo "    ${legends[$i]}: ${log_paths[$i]}"
            done
            echo ""
            
            # Create output directory
            mkdir -p "$OUTPUT_GRANULARITY"
            
            # Run visualization script with all logs
            echo "  Generating plots with visualize_logs_multi.py..."
            python3 "$VISUALIZE_MULTI_SCRIPT" \
                --logs "${log_paths[@]}" \
                --legends "${legends[@]}" \
                --output "$OUTPUT_GRANULARITY"
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Successfully generated plots for granularity experiments"
            else
                echo "  ✗ Failed to generate plots for granularity experiments"
            fi
        else
            echo "  Warning: No valid log files found after sorting"
        fi
        echo ""
        
        echo "Granularity experiments visualization complete!"
        echo "Plots saved to: ${OUTPUT_GRANULARITY}"
        echo ""
    fi
fi

# ============================================================================
# Process Spatial Threshold Experiments
# ============================================================================
SPATIAL_THRESHOLD_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}/spatial_threshold_experiments"
OUTPUT_SPATIAL_THRESHOLD="${SCRIPT_DIR}/plots/spatial_threshold_experiments"

if [ -d "$SPATIAL_THRESHOLD_EXPERIMENTS_DIR" ]; then
    echo "Processing Spatial Threshold Experiments..."
    echo "=========================================="
    echo "Spatial threshold experiments directory: ${SPATIAL_THRESHOLD_EXPERIMENTS_DIR}"
    echo "Output directory: ${OUTPUT_SPATIAL_THRESHOLD}"
    echo ""
    
    # Define log files and their corresponding legends
    log_files=(
        "${SPATIAL_THRESHOLD_EXPERIMENTS_DIR}/backend_only.log"
        "${SPATIAL_THRESHOLD_EXPERIMENTS_DIR}/spatial.log"
        "${SPATIAL_THRESHOLD_EXPERIMENTS_DIR}/global.log"
    )
    legends=(
        "Backend only"
        "Spatial Thresholds"
        "Global Threshold"
    )
    
    # Check if all log files exist
    missing_files=()
    for log_file in "${log_files[@]}"; do
        if [ ! -f "$log_file" ]; then
            missing_files+=("$log_file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "  Warning: Some log files are missing:"
        for missing_file in "${missing_files[@]}"; do
            echo "    ${missing_file}"
        done
        echo "  Skipping spatial threshold experiments..."
        echo ""
    else
        echo "  Found ${#log_files[@]} log files:"
        for i in "${!log_files[@]}"; do
            echo "    ${legends[$i]}: ${log_files[$i]}"
        done
        echo ""
        
        # Create output directory
        mkdir -p "$OUTPUT_SPATIAL_THRESHOLD"
        
        # Run visualization script with all logs
        echo "  Generating plots with visualize_logs_multi.py..."
        python3 "$VISUALIZE_MULTI_SCRIPT" \
            --logs "${log_files[@]}" \
            --legends "${legends[@]}" \
            --output "$OUTPUT_SPATIAL_THRESHOLD"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully generated plots for spatial threshold experiments"
        else
            echo "  ✗ Failed to generate plots for spatial threshold experiments"
        fi
        echo ""
        
        echo "Spatial threshold experiments visualization complete!"
        echo "Plots saved to: ${OUTPUT_SPATIAL_THRESHOLD}"
        echo ""
    fi
fi

# ============================================================================
# Process Deviation Factor Experiments
# ============================================================================
DEVIATION_FACTOR_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}/deviation_factor_experiments"
OUTPUT_DEVIATION_FACTOR="${SCRIPT_DIR}/plots/deviation_factor_experiments"

if [ -d "$DEVIATION_FACTOR_EXPERIMENTS_DIR" ]; then
    echo "Processing Deviation Factor Experiments..."
    echo "=========================================="
    echo "Deviation factor experiments directory: ${DEVIATION_FACTOR_EXPERIMENTS_DIR}"
    echo "Output directory: ${OUTPUT_DEVIATION_FACTOR}"
    echo ""
    
    # Define log files and their corresponding legends
    log_files=(
        "${DEVIATION_FACTOR_EXPERIMENTS_DIR}/0_05.log"
        "${DEVIATION_FACTOR_EXPERIMENTS_DIR}/0_1.log"
        "${DEVIATION_FACTOR_EXPERIMENTS_DIR}/0_25.log"
        "${DEVIATION_FACTOR_EXPERIMENTS_DIR}/0_5.log"
        "${DEVIATION_FACTOR_EXPERIMENTS_DIR}/backend_only.log"
    )
    legends=(
        "0.05"
        "0.1"
        "0.25"
        "0.5"
        "Backend Only"
    )
    
    # Check if all log files exist
    missing_files=()
    for log_file in "${log_files[@]}"; do
        if [ ! -f "$log_file" ]; then
            missing_files+=("$log_file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "  Warning: Some log files are missing:"
        for missing_file in "${missing_files[@]}"; do
            echo "    ${missing_file}"
        done
        echo "  Skipping deviation factor experiments..."
        echo ""
    else
        echo "  Found ${#log_files[@]} log files:"
        for i in "${!log_files[@]}"; do
            echo "    ${legends[$i]}: ${log_files[$i]}"
        done
        echo ""
        
        # Create output directory
        mkdir -p "$OUTPUT_DEVIATION_FACTOR"
        
        # Run visualization script with all logs
        echo "  Generating plots with visualize_logs_multi.py..."
        python3 "$VISUALIZE_MULTI_SCRIPT" \
            --logs "${log_files[@]}" \
            --legends "${legends[@]}" \
            --output "$OUTPUT_DEVIATION_FACTOR"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully generated plots for deviation factor experiments"
        else
            echo "  ✗ Failed to generate plots for deviation factor experiments"
        fi
        echo ""
        
        echo "Deviation factor experiments visualization complete!"
        echo "Plots saved to: ${OUTPUT_DEVIATION_FACTOR}"
        echo ""
    fi
fi

# ============================================================================
# Process PCA Experiments
# ============================================================================
PCA_EXPERIMENTS_DIR="${EXPERIMENTS_DIR}/pca_experiments"
OUTPUT_PCA="${SCRIPT_DIR}/plots/pca_experiments"

if [ -d "$PCA_EXPERIMENTS_DIR" ]; then
    echo "Processing PCA Experiments..."
    echo "=========================================="
    echo "PCA experiments directory: ${PCA_EXPERIMENTS_DIR}"
    echo "Output directory: ${OUTPUT_PCA}"
    echo ""
    
    # Process d_reduced experiments
    D_REDUCED_DIR="${PCA_EXPERIMENTS_DIR}/d_reduced"
    if [ -d "$D_REDUCED_DIR" ]; then
        echo "Processing d_reduced experiments..."
        echo "----------------------------------------"
        
        # Define log files and their corresponding legends
        log_files=(
            "${D_REDUCED_DIR}/16.log"
            "${D_REDUCED_DIR}/64.log"
            "${D_REDUCED_DIR}/128.log"
            "${D_REDUCED_DIR}/256.log"
            "${D_REDUCED_DIR}/512.log"
            "${D_REDUCED_DIR}/backend.log"
        )
        legends=(
            "16"
            "64"
            "128"
            "256"
            "512"
            "Backend Only"
        )
        
        # Check if all log files exist
        missing_files=()
        for log_file in "${log_files[@]}"; do
            if [ ! -f "$log_file" ]; then
                missing_files+=("$log_file")
            fi
        done
        
        if [ ${#missing_files[@]} -gt 0 ]; then
            echo "  Warning: Some log files are missing:"
            for missing_file in "${missing_files[@]}"; do
                echo "    ${missing_file}"
            done
            echo "  Skipping d_reduced experiments..."
            echo ""
        else
            echo "  Found ${#log_files[@]} log files"
            echo ""
            
            # Create output directory
            output_dir="${OUTPUT_PCA}/d_reduced"
            mkdir -p "$output_dir"
            
            # Run visualization script with all logs
            echo "  Generating plots with visualize_logs_multi.py..."
            python3 "$VISUALIZE_MULTI_SCRIPT" \
                --logs "${log_files[@]}" \
                --legends "${legends[@]}" \
                --format-type pca_dim \
                --output "$output_dir" \
                --tick-interval 5
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Successfully generated plots for d_reduced experiments"
            else
                echo "  ✗ Failed to generate plots for d_reduced experiments"
            fi
            echo ""
        fi
    fi
    
    # Process n_buckets experiments
    N_BUCKETS_DIR="${PCA_EXPERIMENTS_DIR}/n_buckets"
    if [ -d "$N_BUCKETS_DIR" ]; then
        echo "Processing n_buckets experiments..."
        echo "----------------------------------------"
        
        # Define log files and their corresponding legends
        log_files=(
            "${N_BUCKETS_DIR}/8.log"
            "${N_BUCKETS_DIR}/16.log"
            "${N_BUCKETS_DIR}/32.log"
            "${N_BUCKETS_DIR}/64.log"
            "${N_BUCKETS_DIR}/128.log"
            "${N_BUCKETS_DIR}/backend.log"
        )
        legends=(
            "8"
            "16"
            "32"
            "64"
            "128"
            "Backend Only"
        )
        
        # Check if all log files exist
        missing_files=()
        for log_file in "${log_files[@]}"; do
            if [ ! -f "$log_file" ]; then
                missing_files+=("$log_file")
            fi
        done
        
        if [ ${#missing_files[@]} -gt 0 ]; then
            echo "  Warning: Some log files are missing:"
            for missing_file in "${missing_files[@]}"; do
                echo "    ${missing_file}"
            done
            echo "  Skipping n_buckets experiments..."
            echo ""
        else
            echo "  Found ${#log_files[@]} log files"
            echo ""
            
            # Create output directory
            output_dir="${OUTPUT_PCA}/n_buckets"
            mkdir -p "$output_dir"
            
            # Run visualization script with all logs
            echo "  Generating plots with visualize_logs_multi.py..."
            python3 "$VISUALIZE_MULTI_SCRIPT" \
                --logs "${log_files[@]}" \
                --legends "${legends[@]}" \
                --format-type buckets_per_dim \
                --output "$output_dir" \
                --tick-interval 5
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Successfully generated plots for n_buckets experiments"
            else
                echo "  ✗ Failed to generate plots for n_buckets experiments"
            fi
            echo ""
        fi
    fi
    
    echo "PCA experiments visualization complete!"
    echo "Plots saved to: ${OUTPUT_PCA}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "All visualizations complete!"
echo ""
echo "Summary:"
if [ -d "$OUTPUT_BACKEND" ]; then
    echo "  Backend experiments: ${OUTPUT_BACKEND}"
fi
if [ -d "$OUTPUT_DATASET" ]; then
    echo "  Dataset experiments: ${OUTPUT_DATASET}"
fi
if [ -d "$OUTPUT_GRANULARITY" ]; then
    echo "  Granularity experiments: ${OUTPUT_GRANULARITY}"
fi
if [ -d "$OUTPUT_SPATIAL_THRESHOLD" ]; then
    echo "  Spatial threshold experiments: ${OUTPUT_SPATIAL_THRESHOLD}"
fi
if [ -d "$OUTPUT_DEVIATION_FACTOR" ]; then
    echo "  Deviation factor experiments: ${OUTPUT_DEVIATION_FACTOR}"
fi
if [ -d "$OUTPUT_PCA" ]; then
    echo "  PCA experiments: ${OUTPUT_PCA}"
fi
echo ""

