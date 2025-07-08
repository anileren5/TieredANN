#!/bin/bash

# Exit on error
set -e

# Change to project root (one level above this script's location)
cd "$(dirname "$0")/.." || exit 1

# Configuration
data_type="float"
dataset="sift"
single_file_index=0
tags_enabled=0
num_nodes_to_cache=500
num_threads=16
beamwidth=2
query_file="./data/${dataset}/${dataset}_query.bin"
truth_file="./data/${dataset}/${dataset}_groundtruth.bin"
K=100
results_prefix="./results/${dataset}/${dataset}"
similarity="l2"
Ls=(128 256 512 1024)

# Create results directory if it doesn't exist
mkdir -p "$(dirname "$results_prefix")"

# Run the search
./build/tests/search_disk_index \
  "$data_type" \
  "./index/${dataset}/${dataset}" \
  "$single_file_index" \
  "$tags_enabled" \
  "$num_nodes_to_cache" \
  "$num_threads" \
  "$beamwidth" \
  "$query_file" \
  "$truth_file" \
  "$K" \
  "$results_prefix" \
  "$similarity" \
  "${Ls[@]}"
