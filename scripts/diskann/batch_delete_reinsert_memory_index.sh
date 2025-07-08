#!/bin/bash

# Builds an index from all dataset vectors. Each cycle picks a subset (size set by batch_size_in_percentage),
# deletes these vectors from the index, consolidates deletions, then reinserts them.
# Repeats for n_iterations cycles. Mainly used to inspect stable recall, but also logs performance metrics
# (QPS, latency, etc.). Useful for reproducing experiments like Figures 2 & 3 in FlashDiskANN.
# Example plots are in the plots folder; example logs are in the experiments folder.

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

# ====== Parameters ======
dataset="sift"
data_type="float"
dist_fn="l2"

data_path="data/$dataset/${dataset}_base.bin"
query_path="data/$dataset/${dataset}_query.bin"
groundtruth_path="data/$dataset/${dataset}_groundtruth.bin"

# Index config
R=32
K=100
L=128
alpha=1.2
ins_thr=32
cons_thr=32
build_thr=32
search_thr=32
batch_size_in_percentage=10
n_iterations=20

./build/tests/batch_delete_reinsert_memory_index \
  --data_type "$data_type" \
  --data_path "$data_path" \
  --query_path "$query_path" \
  --groundtruth_path "$groundtruth_path" \
  --R "$R" \
  --L "$L" \
  --K "$K" \
  --alpha "$alpha" \
  --insert_threads "$ins_thr" \
  --consolidate_threads "$cons_thr" \
  --build_threads "$build_thr" \
  --search_threads "$search_thr" \
  --batch_size_in_percentage "$batch_size_in_percentage" \
  --iterations "$n_iterations"