#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change to the project root directory (one level above the script's location)
cd "$(dirname "$0")/.." || exit 1

# Configuration parameters
dataset="sift"
data_type="float"
R=64
L=128
B=8
M=8
T=16
similarity="l2"
single_file_index=0

# Input and output paths
base_file="./data/${dataset}/${dataset}_base.bin"
index_dir="./index/${dataset}"
index_prefix="${index_dir}/${dataset}"

# Create index directory if it doesn't exist
mkdir -p "$index_dir"

# Run the build_disk_index command with all parameters
./build/tests/build_disk_index "$data_type" "$base_file" "$index_prefix" "$R" "$L" "$B" "$M" "$T" "$similarity" "$single_file_index"
