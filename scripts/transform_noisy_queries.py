#!/usr/bin/env python3
"""
Transform noisy queries and groundtruth files from source parameters to target parameters.

This script:
1. Loads source noisy queries and groundtruth files (identified by source parameters)
2. Transforms them to match target parameters:
   - n_split_repeat: round-robin if target > source, drop if target < source
   - n_split: split further if target > source, merge if target < source
3. Writes transformed files with target parameters in filename
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


def transform_noisy_queries(
    dataset_name,
    source_n_split,
    source_n_split_repeat,
    source_noise_ratio,
    target_n_split,
    target_n_split_repeat,
    target_noise_ratio,
    data_dir="data",
    dtype="float"
):
    """
    Transform noisy queries and groundtruth from source to target parameters.
    
    Args:
        dataset_name: Name of the dataset
        source_n_split: Source number of splits
        source_n_split_repeat: Source number of copies per split
        source_noise_ratio: Source noise ratio
        target_n_split: Target number of splits
        target_n_split_repeat: Target number of copies per split
        target_noise_ratio: Target noise ratio (used in filename only)
        data_dir: Base directory for data files
        dtype: Data type - float, int8, or uint8 (default: float)
    
    Returns:
        Tuple of (output_query_file, output_groundtruth_file)
    """
    if dtype not in ["float", "int8", "uint8"]:
        raise ValueError(f"dtype must be float, int8, or uint8, got {dtype}")
    
    # Construct input paths
    dataset_dir = os.path.join(data_dir, dataset_name)
    
    # Format noise_ratio to match filename format
    source_noise_str = f"{source_noise_ratio:.10f}".rstrip('0').rstrip('.')
    target_noise_str = f"{target_noise_ratio:.10f}".rstrip('0').rstrip('.')
    
    source_query_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_query_nsplit-{source_n_split}_nrepeat-{source_n_split_repeat}_noise-{source_noise_str}.bin"
    )
    source_groundtruth_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_groundtruth_nsplit-{source_n_split}_nrepeat-{source_n_split_repeat}_noise-{source_noise_str}.bin"
    )
    
    # Check if source files exist
    if not os.path.exists(source_query_file):
        raise FileNotFoundError(f"Source query file not found: {source_query_file}")
    if not os.path.exists(source_groundtruth_file):
        raise FileNotFoundError(f"Source groundtruth file not found: {source_groundtruth_file}")
    
    print("=" * 60)
    print("Transforming Noisy Queries")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Source: n_split={source_n_split}, n_repeat={source_n_split_repeat}, noise={source_noise_ratio}")
    print(f"Target: n_split={target_n_split}, n_repeat={target_n_split_repeat}, noise={target_noise_ratio}")
    print(f"dtype: {dtype}")
    print()
    
    # Read source query file
    print("Reading source query file...")
    source_queries, num_source_queries, dim = read_bin_vectors(source_query_file, dtype)
    print(f"Loaded {num_source_queries} queries of dimension {dim}")
    
    # Read source groundtruth file
    print("\nReading source groundtruth file...")
    source_gt_ids, source_gt_dists, num_source_gt_queries, K = read_groundtruth(source_groundtruth_file)
    print(f"Loaded groundtruth: {num_source_gt_queries} queries with K={K}")
    
    if num_source_queries != num_source_gt_queries:
        raise ValueError(f"Mismatch: {num_source_queries} queries but {num_source_gt_queries} groundtruth entries")
    
    # Read original query file to determine original split sizes
    original_query_file = os.path.join(dataset_dir, f"{dataset_name}_query.bin")
    if not os.path.exists(original_query_file):
        raise FileNotFoundError(f"Original query file not found: {original_query_file}")
    
    print("\nReading original query file to determine split sizes...")
    original_queries, num_original_queries, original_dim = read_bin_vectors(original_query_file, dtype)
    print(f"Original queries: {num_original_queries} queries")
    
    if original_dim != dim:
        raise ValueError(f"Dimension mismatch: original dim={original_dim}, source file dim={dim}")
    
    # Determine how queries were originally split (matching generate_noisy_queries.py)
    original_split_arrays = np.array_split(original_queries, source_n_split, axis=0)
    original_split_sizes = [arr.shape[0] for arr in original_split_arrays]
    
    print(f"\nOriginal split sizes (source): {original_split_sizes}")
    print(f"Total original queries: {sum(original_split_sizes)}")
    
    # Verify: total queries in source file should be sum(original_split_sizes) * source_n_split_repeat
    expected_total = sum(original_split_sizes) * source_n_split_repeat
    if num_source_queries != expected_total:
        raise ValueError(f"Query count mismatch: expected {expected_total}, got {num_source_queries}")
    
    # Extract source splits from the file
    # Structure: Split 0 (all copies), Split 1 (all copies), ..., Split N-1 (all copies)
    print("\nExtracting source splits from file...")
    source_split_queries = []
    source_split_gt_ids = []
    source_split_gt_dists = []
    
    current_idx = 0
    for split_idx in range(source_n_split):
        split_size = original_split_sizes[split_idx]
        # Each split appears source_n_split_repeat times consecutively
        num_queries_in_split_block = split_size * source_n_split_repeat
        
        end_idx = current_idx + num_queries_in_split_block
        source_split_queries.append(source_queries[current_idx:end_idx])
        source_split_gt_ids.append(source_gt_ids[current_idx:end_idx])
        source_split_gt_dists.append(source_gt_dists[current_idx:end_idx])
        
        print(f"  Split {split_idx}: {source_split_queries[split_idx].shape[0]} queries "
              f"({split_size} original queries × {source_n_split_repeat} copies)")
        current_idx = end_idx
    
    # Step 1: Transform n_split_repeat for each split
    print(f"\nStep 1: Transforming n_split_repeat ({source_n_split_repeat} -> {target_n_split_repeat})...")
    transformed_split_queries = []
    transformed_split_gt_ids = []
    transformed_split_gt_dists = []
    
    for split_idx, split_queries in enumerate(source_split_queries):
        split_size = original_split_sizes[split_idx]
        
        # Reshape to (source_n_split_repeat, split_size, dim) for queries
        # and (source_n_split_repeat, split_size, K) for groundtruth
        split_queries_reshaped = split_queries.reshape(source_n_split_repeat, split_size, dim)
        split_gt_ids_reshaped = source_split_gt_ids[split_idx].reshape(source_n_split_repeat, split_size, K)
        split_gt_dists_reshaped = source_split_gt_dists[split_idx].reshape(source_n_split_repeat, split_size, K)
        
        if target_n_split_repeat > source_n_split_repeat:
            # Round-robin: repeat copies in a round-robin fashion
            print(f"  Split {split_idx}: Round-robin repeating {source_n_split_repeat} copies to {target_n_split_repeat}")
            indices = np.arange(target_n_split_repeat) % source_n_split_repeat
            transformed_queries = split_queries_reshaped[indices]
            transformed_gt_ids = split_gt_ids_reshaped[indices]
            transformed_gt_dists = split_gt_dists_reshaped[indices]
        elif target_n_split_repeat < source_n_split_repeat:
            # Drop: take first target_n_split_repeat copies
            print(f"  Split {split_idx}: Dropping last {source_n_split_repeat - target_n_split_repeat} copies")
            transformed_queries = split_queries_reshaped[:target_n_split_repeat]
            transformed_gt_ids = split_gt_ids_reshaped[:target_n_split_repeat]
            transformed_gt_dists = split_gt_dists_reshaped[:target_n_split_repeat]
        else:
            # Same: no change needed
            print(f"  Split {split_idx}: No change needed")
            transformed_queries = split_queries_reshaped
            transformed_gt_ids = split_gt_ids_reshaped
            transformed_gt_dists = split_gt_dists_reshaped
        
        # Reshape back to (target_n_split_repeat * split_size, ...)
        transformed_split_queries.append(transformed_queries.reshape(-1, dim))
        transformed_split_gt_ids.append(transformed_gt_ids.reshape(-1, K))
        transformed_split_gt_dists.append(transformed_gt_dists.reshape(-1, K))
    
    # Step 2: Transform n_split
    print(f"\nStep 2: Transforming n_split ({source_n_split} -> {target_n_split})...")
    
    if target_n_split > source_n_split:
        # Split further: need to split each source split into multiple target splits
        print(f"  Splitting {source_n_split} splits into {target_n_split} splits...")
        
        # First, we need to determine how to distribute target splits across source splits
        # We'll use the original split sizes to determine the distribution
        total_original_queries = sum(original_split_sizes)
        
        # Calculate target split sizes based on original queries
        target_original_split_arrays = np.array_split(original_queries, target_n_split, axis=0)
        target_original_split_sizes = [arr.shape[0] for arr in target_original_split_arrays]
        
        # Now map source splits to target splits
        target_split_queries = []
        target_split_gt_ids = []
        target_split_gt_dists = []
        
        current_source_split = 0
        current_source_offset = 0  # Offset within current source split (in original queries)
        
        for target_idx in range(target_n_split):
            target_size = target_original_split_sizes[target_idx]
            target_queries_list = []
            target_gt_ids_list = []
            target_gt_dists_list = []
            
            remaining = target_size
            while remaining > 0 and current_source_split < source_n_split:
                source_split_size = original_split_sizes[current_source_split]
                available = source_split_size - current_source_offset
                take = min(remaining, available)
                
                # Extract the queries from current source split
                # Current structure: [Q0_copy0, Q0_copy1, ..., Q0_copyN, Q1_copy0, ..., Q1_copyN, ...]
                # We need to reshape to (source_split_size, target_n_split_repeat, ...) to group by original query
                source_queries_reshaped = transformed_split_queries[current_source_split].reshape(
                    source_split_size, target_n_split_repeat, dim
                )
                source_gt_ids_reshaped = transformed_split_gt_ids[current_source_split].reshape(
                    source_split_size, target_n_split_repeat, K
                )
                source_gt_dists_reshaped = transformed_split_gt_dists[current_source_split].reshape(
                    source_split_size, target_n_split_repeat, K
                )
                
                # Extract the slice: (take, target_n_split_repeat, ...)
                # This gives us take original queries, each with all target_n_split_repeat copies
                slice_queries = source_queries_reshaped[current_source_offset:current_source_offset + take, :, :]
                slice_gt_ids = source_gt_ids_reshaped[current_source_offset:current_source_offset + take, :, :]
                slice_gt_dists = source_gt_dists_reshaped[current_source_offset:current_source_offset + take, :, :]
                
                # Reshape to (take * target_n_split_repeat, ...) to get back to flat structure
                # [Q0_all_copies, Q1_all_copies, ..., Q(take-1)_all_copies]
                target_queries_list.append(slice_queries.reshape(-1, dim))
                target_gt_ids_list.append(slice_gt_ids.reshape(-1, K))
                target_gt_dists_list.append(slice_gt_dists.reshape(-1, K))
                
                remaining -= take
                current_source_offset += take
                
                # Move to next source split if we've consumed all of current one
                if current_source_offset >= source_split_size:
                    current_source_split += 1
                    current_source_offset = 0
            
            # Concatenate all parts for this target split
            if target_queries_list:
                target_split_queries.append(np.vstack(target_queries_list))
                target_split_gt_ids.append(np.vstack(target_gt_ids_list))
                target_split_gt_dists.append(np.vstack(target_gt_dists_list))
                print(f"    Target split {target_idx}: {target_split_queries[target_idx].shape[0]} queries "
                      f"({target_size} original queries × {target_n_split_repeat} copies)")
            else:
                raise ValueError(f"Failed to create target split {target_idx}")
        
    elif target_n_split < source_n_split:
        # Merge: combine multiple source splits into fewer target splits
        print(f"  Merging {source_n_split} splits into {target_n_split} splits...")
        
        # Calculate how many source splits each target split should contain
        source_splits_per_target = source_n_split // target_n_split
        remainder = source_n_split % target_n_split
        
        target_split_queries = []
        target_split_gt_ids = []
        target_split_gt_dists = []
        
        current_source_idx = 0
        for target_idx in range(target_n_split):
            # Determine how many source splits to merge for this target split
            num_source_splits = source_splits_per_target + (1 if target_idx < remainder else 0)
            
            # Merge source splits
            merged_queries = []
            merged_gt_ids = []
            merged_gt_dists = []
            
            for _ in range(num_source_splits):
                merged_queries.append(transformed_split_queries[current_source_idx])
                merged_gt_ids.append(transformed_split_gt_ids[current_source_idx])
                merged_gt_dists.append(transformed_split_gt_dists[current_source_idx])
                current_source_idx += 1
            
            # Concatenate merged splits
            target_split_queries.append(np.vstack(merged_queries))
            target_split_gt_ids.append(np.vstack(merged_gt_ids))
            target_split_gt_dists.append(np.vstack(merged_gt_dists))
            
            print(f"    Target split {target_idx}: merged {num_source_splits} source splits "
                  f"({target_split_queries[target_idx].shape[0]} queries)")
    else:
        # Same: no change needed
        print(f"  No change needed")
        target_split_queries = transformed_split_queries
        target_split_gt_ids = transformed_split_gt_ids
        target_split_gt_dists = transformed_split_gt_dists
    
    # Concatenate all target splits
    print("\nConcatenating all target splits...")
    final_queries = np.vstack(target_split_queries)
    final_gt_ids = np.vstack(target_split_gt_ids)
    final_gt_dists = np.vstack(target_split_gt_dists)
    
    print(f"Final queries: {final_queries.shape[0]} queries")
    print(f"Final groundtruth: {final_gt_ids.shape[0]} queries")
    
    # Generate output filenames
    output_query_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_query_nsplit-{target_n_split}_nrepeat-{target_n_split_repeat}_noise-{target_noise_str}.bin"
    )
    output_groundtruth_file = os.path.join(
        dataset_dir,
        f"{dataset_name}_groundtruth_nsplit-{target_n_split}_nrepeat-{target_n_split_repeat}_noise-{target_noise_str}.bin"
    )
    
    # Write output query file
    print(f"\nWriting transformed queries to {output_query_file}...")
    write_bin_vectors(final_queries, output_query_file, dtype)
    query_file_size = os.path.getsize(output_query_file)
    print(f"Query file written successfully ({query_file_size / (1024**2):.2f} MB)")
    
    # Write output groundtruth file
    print(f"\nWriting transformed groundtruth to {output_groundtruth_file}...")
    write_groundtruth(final_gt_ids, final_gt_dists, output_groundtruth_file)
    gt_file_size = os.path.getsize(output_groundtruth_file)
    print(f"Groundtruth file written successfully ({gt_file_size / (1024**2):.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✓ Success! Transformed noisy queries and groundtruth")
    print("=" * 60)
    print(f"Output query file: {output_query_file}")
    print(f"Output groundtruth file: {output_groundtruth_file}")
    
    return output_query_file, output_groundtruth_file


def main():
    parser = argparse.ArgumentParser(
        description="Transform noisy queries and groundtruth from source to target parameters"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--source_n_split",
        type=int,
        required=True,
        help="Source number of splits"
    )
    parser.add_argument(
        "--source_n_split_repeat",
        type=int,
        required=True,
        help="Source number of copies per split"
    )
    parser.add_argument(
        "--source_noise_ratio",
        type=float,
        required=True,
        help="Source noise ratio"
    )
    parser.add_argument(
        "--target_n_split",
        type=int,
        required=True,
        help="Target number of splits"
    )
    parser.add_argument(
        "--target_n_split_repeat",
        type=int,
        required=True,
        help="Target number of copies per split"
    )
    parser.add_argument(
        "--target_noise_ratio",
        type=float,
        required=True,
        help="Target noise ratio (used in filename only)"
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
        output_query_file, output_groundtruth_file = transform_noisy_queries(
            dataset_name=args.dataset,
            source_n_split=args.source_n_split,
            source_n_split_repeat=args.source_n_split_repeat,
            source_noise_ratio=args.source_noise_ratio,
            target_n_split=args.target_n_split,
            target_n_split_repeat=args.target_n_split_repeat,
            target_noise_ratio=args.target_noise_ratio,
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

