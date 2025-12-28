#!/bin/bash

# Exit on error
set -e

# Change to project root (one level above this script's location)
cd "$(dirname "$0")/../../.." || exit 1

# Configuration
data_type="float"
dataset="glove"
single_file_index=0
tags_enabled=0
num_nodes_to_cache=500
num_threads=32
beamwidth=2
query_file="./data/${dataset}/${dataset}_query.bin"
truth_file="./data/${dataset}/${dataset}_groundtruth.bin"
K=100
results_prefix="./results/${dataset}/${dataset}"
similarity="l2"  # Distance metric: "l2", "cosine", or "inner_product"
Ls=(128 256 512 1024)
sector_len=4096

# Create results directory if it doesn't exist
mkdir -p "$(dirname "$results_prefix")"

# Run the search
./build/tests/search_disk_index "$data_type" \
  --index_prefix_path "./index/${dataset}/${dataset}" \
  --single_file_index "$single_file_index" \
  --tags "$tags_enabled" \
  --num_nodes_to_cache "$num_nodes_to_cache" \
  --num_threads "$num_threads" \
  --beamwidth "$beamwidth" \
  --query_bin "$query_file" \
  --truthset_bin "$truth_file" \
  --recall_at "$K" \
  --result_output_prefix "$results_prefix" \
  --dist_metric "$similarity" \
  --sector_len "$sector_len" \
  --L_values "${Ls[@]}"
