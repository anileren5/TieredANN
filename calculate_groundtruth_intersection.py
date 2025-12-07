#!/usr/bin/env python3
"""
Calculate the intersection percentage of neighbors across n_repeat copies
for each query in the groundtruth file.

For each query, we get the neighbor sets from all n_repeat copies,
calculate the intersection, and compute the percentage.
"""

import argparse
import os
import struct
import numpy as np
from typing import List, Set


def read_groundtruth(input_file: str):
    """
    Read groundtruth from binary file (DiskANN format).
    
    Format:
    - First 4 bytes: num_queries (int32)
    - Next 4 bytes: K (int32)
    - Next: num_queries * K * 4 bytes of IDs (uint32_t)
    - Optionally: num_queries * K * 4 bytes of distances (float32)
    
    Returns:
        groundtruth_ids, groundtruth_dists (or None), num_queries, K
    """
    file_size = os.path.getsize(input_file)
    
    with open(input_file, "rb") as f:
        num_queries = int(struct.unpack("i", f.read(4))[0])
        K = int(struct.unpack("i", f.read(4))[0])
        
        # Read IDs
        ids_size = num_queries * K * 4
        groundtruth_ids = np.frombuffer(f.read(ids_size), dtype=np.uint32)
        groundtruth_ids = groundtruth_ids.reshape(num_queries, K)
        
        # Check if distances are present by comparing file size
        expected_size_with_dists = 2 * 4 + 2 * num_queries * K * 4  # 2 ints (4 bytes each) + IDs + dists
        expected_size_no_dists = 2 * 4 + num_queries * K * 4  # 2 ints (4 bytes each) + IDs only
        
        groundtruth_dists = None
        if file_size == expected_size_with_dists:
            # Read distances
            dists_size = num_queries * K * 4
            groundtruth_dists = np.frombuffer(f.read(dists_size), dtype=np.float32)
            groundtruth_dists = groundtruth_dists.reshape(num_queries, K)
        elif file_size != expected_size_no_dists:
            raise ValueError(f"Unexpected file size: {file_size}. Expected {expected_size_with_dists} (with dists) or {expected_size_no_dists} (no dists)")
    
    return groundtruth_ids, groundtruth_dists, num_queries, K


def calculate_intersection_percentage(groundtruth_ids: np.ndarray, n_splits: int, n_repeat: int, k: int) -> float:
    """
    Calculate the average intersection percentage across all queries.
    
    For each query in each split:
    1. Get neighbor sets from all n_repeat copies (using first k neighbors)
    2. Calculate intersection of these sets
    3. Calculate percentage: intersection_size / k
    4. Average across all queries
    
    Args:
        groundtruth_ids: Array of shape (num_queries, K_file) with neighbor IDs
        n_splits: Number of splits
        n_repeat: Number of repeats per split
        k: Number of neighbors to use for intersection calculation (must be <= K_file)
        
    Returns:
        Average intersection percentage (0.0 to 1.0)
    """
    num_queries, K_file = groundtruth_ids.shape
    
    if k > K_file:
        raise ValueError(f"k ({k}) cannot be larger than K in file ({K_file})")
    
    queries_per_split = num_queries // (n_splits * n_repeat)
    
    if num_queries % (n_splits * n_repeat) != 0:
        raise ValueError(f"Number of queries ({num_queries}) must be divisible by n_splits * n_repeat ({n_splits * n_repeat})")
    
    intersection_percentages = []
    
    # Process each split
    for split_idx in range(n_splits):
        # Process each query in this split
        for query_idx in range(queries_per_split):
            # Get neighbor sets for all n_repeat copies of this query
            neighbor_sets = []
            
            for copy_idx in range(n_repeat):
                # Calculate position in groundtruth array
                # Structure: split 0 (all copies), split 1 (all copies), ...
                # Within each split: copy 0, copy 1, ..., copy (n_repeat-1)
                split_offset = split_idx * n_repeat * queries_per_split
                copy_offset = copy_idx * queries_per_split
                position = split_offset + copy_offset + query_idx
                
                # Get first k neighbor IDs for this copy
                neighbor_ids = set(groundtruth_ids[position, :k].tolist())
                neighbor_sets.append(neighbor_ids)
            
            # Calculate intersection of all neighbor sets
            if len(neighbor_sets) == 0:
                continue
            
            intersection = neighbor_sets[0]
            for neighbor_set in neighbor_sets[1:]:
                intersection = intersection & neighbor_set
            
            # Calculate percentage: intersection size / k
            intersection_size = len(intersection)
            percentage = intersection_size / k
            intersection_percentages.append(percentage)
    
    # Calculate average
    if len(intersection_percentages) == 0:
        return 0.0
    
    average_percentage = np.mean(intersection_percentages)
    return average_percentage


def main():
    parser = argparse.ArgumentParser(
        description='Calculate intersection percentage of neighbors across n_repeat copies'
    )
    parser.add_argument('--groundtruth_path', type=str, required=True,
                        help='Path to groundtruth file (DiskANN binary format)')
    parser.add_argument('--n_splits', type=int, required=True,
                        help='Number of splits')
    parser.add_argument('--n_repeat', type=int, required=True,
                        help='Number of repeats per split')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of neighbors to use for intersection calculation (default: use K from file)')
    
    args = parser.parse_args()
    
    print(f"Loading groundtruth from: {args.groundtruth_path}")
    groundtruth_ids, groundtruth_dists, num_queries, K_file = read_groundtruth(args.groundtruth_path)
    print(f"Loaded {num_queries} queries with K={K_file} in file")
    
    # Determine which K to use for calculation
    if args.k is not None:
        k = args.k
        if k > K_file:
            print(f"Error: --k ({k}) cannot be larger than K in file ({K_file})")
            return 1
        print(f"Using k={k} for intersection calculation (first {k} neighbors)")
    else:
        k = K_file
        print(f"Using k={k} for intersection calculation (all neighbors from file)")
    
    print(f"\nCalculating intersection percentage...")
    print(f"  n_splits: {args.n_splits}")
    print(f"  n_repeat: {args.n_repeat}")
    print(f"  k: {k}")
    
    queries_per_split = num_queries // (args.n_splits * args.n_repeat)
    print(f"  queries_per_split: {queries_per_split}")
    
    if num_queries % (args.n_splits * args.n_repeat) != 0:
        print(f"Warning: {num_queries} queries is not divisible by {args.n_splits * args.n_repeat}")
        print(f"  Using {args.n_splits * args.n_repeat * queries_per_split} queries")
    
    average_percentage = calculate_intersection_percentage(
        groundtruth_ids, args.n_splits, args.n_repeat, k
    )
    
    print(f"\nAverage intersection percentage: {average_percentage:.4f} ({average_percentage * 100:.2f}%)")
    print(f"  This means on average, {average_percentage * k:.2f} out of {k} neighbors")
    print(f"  are common across all {args.n_repeat} copies of each query.")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

