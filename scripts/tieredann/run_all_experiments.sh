#!/bin/bash

# Change to project root
cd "$(dirname "$0")/../.." || exit 1

echo "=== Running tiered_index_search_all.sh ==="
scripts/tieredann/tiered_index_search_all.sh

echo "=== Running disk_index_search_all.sh ==="
scripts/tieredann/disk_index_search_all.sh

echo "=== Running parameter_experiments_tiered_index_search.sh ==="
scripts/tieredann/parameter_experiments_tiered_index_search.sh

echo "=== All experiments completed ==="