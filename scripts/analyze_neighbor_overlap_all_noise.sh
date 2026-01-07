#!/bin/bash

# Script to analyze neighbor overlap across repetitions for multiple noise ratios
# Processes all groundtruth files with nsplit=1, nrepeat=2 and different noise ratios

set -e

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Check if Python script exists
PYTHON_SCRIPT="scripts/analyze_neighbor_overlap.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Default values (can be overridden by command-line arguments)
DATASET="gist"
N_SPLIT="10"
N_SPLIT_REPEAT="2"
K="10"
DATA_DIR="data"
DATA_TYPE="float"

# Noise ratios to process (0 to 1 with 0.025 increments)
NOISE_RATIOS=(
    "0.000"
    "0.025"
    "0.050"
    "0.075"
    "0.100"
    "0.125"
    "0.150"
    "0.175"
    "0.200"
    "0.225"
    "0.250"
    "0.275"
    "0.300"
    "0.325"
    "0.350"
    "0.375"
    "0.400"
    "0.425"
    "0.450"
    "0.475"
    "0.500"
    "0.525"
    "0.550"
    "0.575"
    "0.600"
    "0.625"
    "0.650"
    "0.675"
    "0.700"
    "0.725"
    "0.750"
    "0.775"
    "0.800"
    "0.825"
    "0.850"
    "0.875"
    "0.900"
    "0.925"
    "0.950"
    "0.975"
    "1.000"
)

# Parse optional arguments
if [ $# -ge 1 ]; then
    DATASET="$1"
fi
if [ $# -ge 2 ]; then
    K="$2"
fi
if [ $# -ge 3 ]; then
    DATA_DIR="$3"
fi
if [ $# -ge 4 ]; then
    DATA_TYPE="$4"
fi

echo "=========================================="
echo "Analyzing Neighbor Overlap for Multiple Noise Ratios"
echo "=========================================="
echo "Dataset: $DATASET"
echo "n_split: $N_SPLIT"
echo "n_split_repeat: $N_SPLIT_REPEAT"
echo "k: $K"
echo "Data dir: $DATA_DIR"
echo "Data type: $DATA_TYPE"
echo "Noise ratios: ${NOISE_RATIOS[*]}"
echo "=========================================="
echo ""

# Arrays to store results
declare -a results_avg
declare -a results_std
declare -a results_min
declare -a results_max
declare -a noise_values

# Process each noise ratio
for NOISE_RATIO in "${NOISE_RATIOS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing noise ratio: $NOISE_RATIO"
    echo "=========================================="
    
    # Check if groundtruth file exists
    # Format noise_ratio to match file naming (remove trailing zeros after decimal)
    NOISE_STR=$(echo "$NOISE_RATIO" | sed -E 's/(\.[0-9]*[1-9])0+$/\1/;s/\.0+$//;s/\.$//')
    GT_FILE="$DATA_DIR/$DATASET/${DATASET}_groundtruth_nsplit-${N_SPLIT}_nrepeat-${N_SPLIT_REPEAT}_noise-${NOISE_STR}.bin"
    
    if [ ! -f "$GT_FILE" ]; then
        echo "  ⚠ Warning: Groundtruth file not found: $GT_FILE"
        echo "  Skipping noise ratio $NOISE_RATIO..."
        continue
    fi
    
    echo "  Groundtruth file: $GT_FILE"
    echo ""
    
    # Run the Python script and capture output
    OUTPUT=$(python3 "$PYTHON_SCRIPT" \
        --dataset "$DATASET" \
        --n_split "$N_SPLIT" \
        --n_split_repeat "$N_SPLIT_REPEAT" \
        --noise_ratio "$NOISE_RATIO" \
        --k "$K" \
        --data_dir "$DATA_DIR" \
        --dtype "$DATA_TYPE" 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "  ✗ Error: Failed to analyze noise ratio $NOISE_RATIO"
        echo "$OUTPUT"
        continue
    fi
    
    # Extract statistics from output
    # Look for lines like:
    # "Average neighbor overlap across all queries: 0.8523 (85.23%)"
    # "Standard deviation: 0.1234 (12.34%)"
    # "Min overlap: 0.5000 (50.00%)"
    # "Max overlap: 1.0000 (100.00%)"
    AVG_OVERLAP=$(echo "$OUTPUT" | grep -oP "Average neighbor overlap across all queries: \K[0-9]+\.[0-9]+" || echo "N/A")
    STD_OVERLAP=$(echo "$OUTPUT" | grep -oP "Standard deviation: \K[0-9]+\.[0-9]+" || echo "N/A")
    MIN_OVERLAP=$(echo "$OUTPUT" | grep -oP "Min overlap: \K[0-9]+\.[0-9]+" || echo "N/A")
    MAX_OVERLAP=$(echo "$OUTPUT" | grep -oP "Max overlap: \K[0-9]+\.[0-9]+" || echo "N/A")
    
    if [ "$AVG_OVERLAP" != "N/A" ]; then
        results_avg+=("$AVG_OVERLAP")
        results_std+=("$STD_OVERLAP")
        results_min+=("$MIN_OVERLAP")
        results_max+=("$MAX_OVERLAP")
        noise_values+=("$NOISE_RATIO")
        echo "  ✓ Success!"
        echo "    Average: $AVG_OVERLAP | Std Dev: $STD_OVERLAP | Min: $MIN_OVERLAP | Max: $MAX_OVERLAP"
    else
        echo "  ⚠ Warning: Could not extract statistics from output"
    fi
done

# Print summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
printf "%-11s | %12s | %12s | %12s | %12s\n" "Noise Ratio" "Avg Overlap" "Std Dev" "Min Overlap" "Max Overlap"
echo "------------|-------------|-------------|-------------|-------------"

if [ ${#results_avg[@]} -eq 0 ]; then
    echo "No results to display"
else
    for i in "${!results_avg[@]}"; do
        printf "%-11s | %12s | %12s | %12s | %12s\n" \
            "${noise_values[$i]}" \
            "${results_avg[$i]}" \
            "${results_std[$i]}" \
            "${results_min[$i]}" \
            "${results_max[$i]}"
    done
fi

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="

