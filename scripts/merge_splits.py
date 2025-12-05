#!/usr/bin/env python3
"""
Merge query splits from 10 splits to 5 splits by merging pairs of splits.

This script:
1. Reads query file with 10 splits (n_split=10)
2. Merges pairs: (0,1), (2,3), (4,5), (6,7), (8,9) into 5 new splits
3. Writes new query file with 5 splits
4. Does the same for corresponding groundtruth file
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


def write_bin_vectors(vectors, output_file, dtype="float"):
    """Write vectors to a .bin file (DiskANN format)."""
    num_vectors, dim = vectors.shape
    with open(output_file, "wb") as f:
        f.write(struct.pack("I", num_vectors))
        f.write(struct.pack("I", dim))
        
        if dtype == "float":
            vectors_flat = vectors.astype(np.float32).flatten()
        elif dtype == "int8":
            vectors_flat = vectors.astype(np.int8).flatten()
        elif dtype == "uint8":
            vectors_flat = vectors.astype(np.uint8).flatten()
        else:
            raise ValueError(f"Unsupported data type: {dtype}. Must be float, int8, or uint8.")
        
        # Use tofile for efficient writing of large arrays
        vectors_flat.tofile(f)


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


def write_groundtruth(ids, dists, output_file):
    """
    Write groundtruth to binary file (DiskANN format).
    
    Args:
        ids: numpy array of shape (num_queries, K) with uint32 IDs
        dists: numpy array of shape (num_queries, K) with float32 distances
        output_file: output file path
    """
    num_queries, K = ids.shape
    
    with open(output_file, "wb") as f:
        # Write metadata
        f.write(struct.pack("i", num_queries))
        f.write(struct.pack("i", K))
        
        # Write IDs
        ids_flat = ids.astype(np.uint32).flatten()
        ids_flat.tofile(f)
        
        # Write distances
        dists_flat = dists.astype(np.float32).flatten()
        dists_flat.tofile(f)


def merge_splits(
    dataset_name,
    n_split_repeat,
    noise_ratio,
    data_dir="data",
    dtype="float"
):
    """
    Merge query splits from 10 splits to 5 splits.
    
    Args:
        dataset_name: Name of the dataset
        n_split_repeat: Number of copies per split (from original generation)
        noise_ratio: Noise ratio used in original generation
        data_dir: Base directory for data files
        dtype: Data type - float, int8, or uint8 (default: float)
    
    Returns:
        Tuple of (output_query_file, output_groundtruth_file)
    """
    if dtype not in ["float", "int8", "uint8"]:
        raise ValueError(f"dtype must be float, int8, or uint8, got {dtype}")
    
    # Construct input paths (10 splits)
    dataset_dir = os.path.join(data_dir, dataset_name)
    
    # Format noise_ratio to match filename format
    noise_str = f"{noise_ratio:.10f}".rstrip('0').rstrip('.')
    input_query_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_query_nsplit-10_nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    input_groundtruth_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_groundtruth_nsplit-10_nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    
    # Check if input files exist
    if not os.path.exists(input_query_file):
        raise FileNotFoundError(f"Input query file not found: {input_query_file}")
    if not os.path.exists(input_groundtruth_file):
        raise FileNotFoundError(f"Input groundtruth file not found: {input_groundtruth_file}")
    
    print("=" * 60)
    print("Merging Splits: 10 -> 5")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Input query file: {input_query_file}")
    print(f"Input groundtruth file: {input_groundtruth_file}")
    print(f"n_split_repeat: {n_split_repeat}")
    print(f"noise_ratio: {noise_ratio}")
    print(f"dtype: {dtype}")
    print()
    
    # Read input query file
    print("Reading input query file...")
    queries, num_queries, dim = read_bin_vectors(input_query_file, dtype)
    print(f"Loaded {num_queries} queries of dimension {dim}")
    
    # Read input groundtruth file
    print("\nReading input groundtruth file...")
    gt_ids, gt_dists, num_gt_queries, K = read_groundtruth(input_groundtruth_file)
    print(f"Loaded groundtruth: {num_gt_queries} queries with K={K}")
    
    if num_queries != num_gt_queries:
        raise ValueError(f"Mismatch: {num_queries} queries but {num_gt_queries} groundtruth entries")
    
    # Read original query file to determine original split sizes
    original_query_file = os.path.join(dataset_dir, f"{dataset_name}_query.bin")
    if not os.path.exists(original_query_file):
        raise FileNotFoundError(f"Original query file not found: {original_query_file}")
    
    print("\nReading original query file to determine split sizes...")
    original_queries, num_original_queries, original_dim = read_bin_vectors(original_query_file, dtype)
    print(f"Original queries: {num_original_queries} queries")
    
    if original_dim != dim:
        raise ValueError(f"Dimension mismatch: original dim={original_dim}, split file dim={dim}")
    
    # Use np.array_split to determine how queries were originally split (matching generate_noisy_queries.py)
    # This gives us the size of each original split
    original_split_arrays = np.array_split(original_queries, 10, axis=0)
    original_split_sizes = [arr.shape[0] for arr in original_split_arrays]
    
    print(f"\nOriginal split sizes: {original_split_sizes}")
    print(f"Total original queries: {sum(original_split_sizes)}")
    
    # Verify: total queries in split file should be sum(original_split_sizes) * n_split_repeat
    expected_total = sum(original_split_sizes) * n_split_repeat
    if num_queries != expected_total:
        raise ValueError(f"Query count mismatch: expected {expected_total}, got {num_queries}")
    
    # Extract 10 splits from the file
    # Structure: Split 0 (all copies), Split 1 (all copies), ..., Split 9 (all copies)
    print("\nExtracting 10 original splits from file...")
    split_queries = []
    split_gt_ids = []
    split_gt_dists = []
    
    current_idx = 0
    for split_idx in range(10):
        split_size = original_split_sizes[split_idx]
        # Each split appears n_split_repeat times consecutively
        num_queries_in_split_block = split_size * n_split_repeat
        
        end_idx = current_idx + num_queries_in_split_block
        split_queries.append(queries[current_idx:end_idx])
        split_gt_ids.append(gt_ids[current_idx:end_idx])
        split_gt_dists.append(gt_dists[current_idx:end_idx])
        
        print(f"  Split {split_idx}: {split_queries[split_idx].shape[0]} queries "
              f"({split_size} original queries × {n_split_repeat} copies)")
        current_idx = end_idx
    
    # Merge pairs: (0,1), (2,3), (4,5), (6,7), (8,9) -> 5 new splits
    print("\nMerging split pairs into 5 new splits...")
    merged_queries = []
    merged_gt_ids = []
    merged_gt_dists = []
    
    for new_split_idx in range(5):
        old_split_idx_1 = new_split_idx * 2
        old_split_idx_2 = new_split_idx * 2 + 1
        
        # Concatenate the two splits
        merged_query = np.vstack([split_queries[old_split_idx_1], split_queries[old_split_idx_2]])
        merged_gt_id = np.vstack([split_gt_ids[old_split_idx_1], split_gt_ids[old_split_idx_2]])
        merged_gt_dist = np.vstack([split_gt_dists[old_split_idx_1], split_gt_dists[old_split_idx_2]])
        
        merged_queries.append(merged_query)
        merged_gt_ids.append(merged_gt_id)
        merged_gt_dists.append(merged_gt_dist)
        
        print(f"  New split {new_split_idx}: merged splits {old_split_idx_1} and {old_split_idx_2} "
              f"({merged_query.shape[0]} queries)")
    
    # Concatenate all merged splits
    print("\nConcatenating all merged splits...")
    final_queries = np.vstack(merged_queries)
    final_gt_ids = np.vstack(merged_gt_ids)
    final_gt_dists = np.vstack(merged_gt_dists)
    
    print(f"Final queries: {final_queries.shape[0]} queries")
    print(f"Final groundtruth: {final_gt_ids.shape[0]} queries")
    
    # Generate output filenames (5 splits)
    output_query_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_query_nsplit-5_nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    output_groundtruth_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_groundtruth_nsplit-5_nrepeat-{n_split_repeat}_noise-{noise_str}.bin"
    )
    
    # Write output query file
    print(f"\nWriting merged queries to {output_query_file}...")
    write_bin_vectors(final_queries, output_query_file, dtype)
    query_file_size = os.path.getsize(output_query_file)
    print(f"Query file written successfully ({query_file_size / (1024**2):.2f} MB)")
    
    # Write output groundtruth file
    print(f"\nWriting merged groundtruth to {output_groundtruth_file}...")
    write_groundtruth(final_gt_ids, final_gt_dists, output_groundtruth_file)
    gt_file_size = os.path.getsize(output_groundtruth_file)
    print(f"Groundtruth file written successfully ({gt_file_size / (1024**2):.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✓ Success! Merged 10 splits into 5 splits")
    print("=" * 60)
    print(f"Output query file: {output_query_file}")
    print(f"Output groundtruth file: {output_groundtruth_file}")
    
    return output_query_file, output_groundtruth_file


def main():
    parser = argparse.ArgumentParser(
        description="Merge query splits from 10 splits to 5 splits"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--n_split_repeat",
        type=int,
        required=True,
        help="Number of copies per split (from original generation)"
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        required=True,
        help="Noise ratio used in original generation"
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
        output_query_file, output_groundtruth_file = merge_splits(
            dataset_name=args.dataset,
            n_split_repeat=args.n_split_repeat,
            noise_ratio=args.noise_ratio,
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

