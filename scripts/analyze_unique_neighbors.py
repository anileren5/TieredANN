#!/usr/bin/env python3
"""
Analyze unique neighbors in groundtruth files.

For each split and its repetitions, calculate the number of unique neighbors
within k, accumulating across repetitions within each split.
When switching to a new split, reset the accumulator to 0.
"""

import argparse
import numpy as np
import struct
import os
import sys


def read_bin_vectors(input_file, dtype="float"):
    """Read vectors from a .bin file (DiskANN format)."""
    with open(input_file, "rb") as f:
        num_vectors = struct.unpack("I", f.read(4))[0]
        dim = struct.unpack("I", f.read(4))[0]
        total_values = num_vectors * dim
        
        if dtype == "float":
            vector_data = struct.unpack(f"{total_values}f", f.read(total_values * 4))
            vectors = np.array(vector_data, dtype=np.float32).reshape(num_vectors, dim)
        elif dtype == "int8":
            vector_data = struct.unpack(f"{total_values}b", f.read(total_values * 1))
            vectors = np.array(vector_data, dtype=np.int8).reshape(num_vectors, dim)
        elif dtype == "uint8":
            vector_data = struct.unpack(f"{total_values}B", f.read(total_values * 1))
            vectors = np.array(vector_data, dtype=np.uint8).reshape(num_vectors, dim)
        else:
            raise ValueError(f"Unsupported data type: {dtype}. Must be float, int8, or uint8.")
        
        return vectors, num_vectors, dim


def read_groundtruth(input_file):
    """
    Read groundtruth from binary file (DiskANN format).
    
    Format:
    - First 4 bytes: num_queries (int32)
    - Next 4 bytes: K (int32)
    - Next: num_queries * K * 4 bytes of IDs (uint32_t)
    - Next: num_queries * K * 4 bytes of distances (float32)
    """
    with open(input_file, "rb") as f:
        num_queries = int(struct.unpack("i", f.read(4))[0])
        K = int(struct.unpack("i", f.read(4))[0])
        
        # Read IDs
        ids_size = num_queries * K * 4
        groundtruth_ids = np.frombuffer(f.read(ids_size), dtype=np.uint32)
        groundtruth_ids = groundtruth_ids.reshape(num_queries, K)
        
        # Read distances
        dists_size = num_queries * K * 4
        groundtruth_dists = np.frombuffer(f.read(dists_size), dtype=np.float32)
        groundtruth_dists = groundtruth_dists.reshape(num_queries, K)
    
    return groundtruth_ids, groundtruth_dists, num_queries, K


def analyze_unique_neighbors(
    dataset_name,
    n_split,
    n_split_repeat,
    noise_ratio,
    k,
    data_dir="data",
    dtype="float"
):
    """
    Analyze unique neighbors in groundtruth file.
    
    Args:
        dataset_name: Name of the dataset
        n_split: Number of splits
        n_split_repeat: Number of copies per split
        noise_ratio: Noise ratio used in generation
        k: Number of neighbors to consider
        data_dir: Base directory for data files
        dtype: Data type - float, int8, or uint8 (default: float)
    """
    if dtype not in ["float", "int8", "uint8"]:
        raise ValueError(f"dtype must be float, int8, or uint8, got {dtype}")
    
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    
    # Construct paths
    dataset_dir = os.path.join(data_dir, dataset_name)
    
    # Format noise_ratio to match filename format
    noise_str = f"{noise_ratio:.10f}".rstrip('0').rstrip('.')
    groundtruth_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_groundtruth_nsplit-{n_split}_nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    
    # Check if groundtruth file exists
    if not os.path.exists(groundtruth_file):
        raise FileNotFoundError(f"Groundtruth file not found: {groundtruth_file}")
    
    # Read original query file to determine split sizes
    original_query_file = os.path.join(dataset_dir, f"{dataset_name}_query.bin")
    if not os.path.exists(original_query_file):
        raise FileNotFoundError(f"Original query file not found: {original_query_file}")
    
    print("=" * 60)
    print("Analyzing Unique Neighbors")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Groundtruth file: {groundtruth_file}")
    print(f"n_split: {n_split}")
    print(f"n_split_repeat: {n_split_repeat}")
    print(f"noise_ratio: {noise_ratio}")
    print(f"k: {k}")
    print(f"dtype: {dtype}")
    print()
    
    # Read original queries to determine split sizes
    print("Reading original query file to determine split sizes...")
    original_queries, num_original_queries, original_dim = read_bin_vectors(original_query_file, dtype)
    print(f"Original queries: {num_original_queries} queries")
    
    # Determine how queries were originally split
    original_split_arrays = np.array_split(original_queries, n_split, axis=0)
    original_split_sizes = [arr.shape[0] for arr in original_split_arrays]
    
    print(f"Original split sizes: {original_split_sizes}")
    print()
    
    # Read groundtruth file
    print("Reading groundtruth file...")
    gt_ids, gt_dists, num_gt_queries, K_gt = read_groundtruth(groundtruth_file)
    print(f"Loaded groundtruth: {num_gt_queries} queries with K={K_gt}")
    
    if k > K_gt:
        raise ValueError(f"k={k} is greater than K={K_gt} in groundtruth file")
    
    # Verify: total queries in groundtruth should be sum(original_split_sizes) * n_split_repeat
    expected_total = sum(original_split_sizes) * n_split_repeat
    if num_gt_queries != expected_total:
        raise ValueError(f"Query count mismatch: expected {expected_total}, got {num_gt_queries}")
    
    print()
    print("=" * 60)
    print("Unique Neighbors Analysis (accumulated per split)")
    print("=" * 60)
    print(f"Format: Split <split_idx>, Repetition <rep_idx>: <unique_count> unique neighbors")
    print()
    
    # Process each split
    # Structure: For each split, all copies are concatenated:
    # Split 0: [copy0_all_queries, copy1_all_queries, ..., copy(n_split_repeat-1)_all_queries]
    # Split 1: [copy0_all_queries, copy1_all_queries, ..., copy(n_split_repeat-1)_all_queries]
    # etc.
    split_start_idx = 0
    for split_idx in range(n_split):
        split_size = original_split_sizes[split_idx]
        
        # Initialize unique neighbors set for this split
        unique_neighbors = set()
        
        # Process each repetition in this split
        # For repetition rep_idx, queries are at indices:
        # split_start_idx + rep_idx * split_size to split_start_idx + (rep_idx + 1) * split_size - 1
        for rep_idx in range(n_split_repeat):
            # Extract queries for this repetition
            rep_start_idx = split_start_idx + rep_idx * split_size
            rep_end_idx = split_start_idx + (rep_idx + 1) * split_size
            repetition_gt_ids = gt_ids[rep_start_idx:rep_end_idx]
            
            # Get top k neighbors for each query in this repetition
            for query_gt_ids in repetition_gt_ids:
                # Add the top k neighbors to the unique set
                top_k_neighbors = query_gt_ids[:k]
                unique_neighbors.update(top_k_neighbors)
            
            # Print the count
            print(f"Split {split_idx}, Repetition {rep_idx}: {len(unique_neighbors)} unique neighbors")
        
        # Move to next split
        split_start_idx += split_size * n_split_repeat
        print()  # Empty line between splits
    
    print("=" * 60)
    print("Analysis complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze unique neighbors in groundtruth files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--n_split",
        type=int,
        required=True,
        help="Number of splits"
    )
    parser.add_argument(
        "--n_split_repeat",
        type=int,
        required=True,
        help="Number of copies per split"
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        required=True,
        help="Noise ratio used in generation"
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of neighbors to consider"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for data files (default: data)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float",
        choices=["float", "int8", "uint8"],
        help="Data type - float, int8, or uint8 (default: float)"
    )
    
    args = parser.parse_args()
    
    try:
        analyze_unique_neighbors(
            dataset_name=args.dataset,
            n_split=args.n_split,
            n_split_repeat=args.n_split_repeat,
            noise_ratio=args.noise_ratio,
            k=args.k,
            data_dir=args.data_dir,
            dtype=args.dtype
        )
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

