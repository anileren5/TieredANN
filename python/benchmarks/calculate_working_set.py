#!/usr/bin/env python3
"""
Calculate working set size for each query split using groundtruth data.

This script splits queries into N splits and reports the number of distinct tags
that appear in the top-K groundtruth results for all queries within each split.
"""

import argparse
import numpy as np
import json
import sys
from typing import Dict


def load_groundtruth(groundtruth_path: str) -> tuple:
    """
    Load groundtruth from binary file (DiskANN format).
    
    Format (int32 for metadata):
    - First 4 bytes: num_queries (int32)
    - Next 4 bytes: K (int32)
    - Rest: num_queries * K * sizeof(uint32_t) bytes of groundtruth IDs
    - Optionally: num_queries * K * sizeof(float) bytes of distances
    """
    with open(groundtruth_path, 'rb') as f:
        # Read metadata (int32 format, not uint32)
        num_queries = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        K = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        
        # Read IDs
        ids_size = num_queries * K * 4
        groundtruth_ids = np.frombuffer(f.read(ids_size), dtype=np.uint32)
        groundtruth_ids = groundtruth_ids.reshape(num_queries, K)
    
    return groundtruth_ids, K


def calculate_working_set(
    groundtruth_path: str,
    n_splits: int,
    K: int
) -> Dict:
    """
    Calculate working set size for each query split using groundtruth.
    
    Args:
        groundtruth_path: Path to groundtruth binary file
        n_splits: Number of splits to divide queries into
        K: Number of nearest neighbors to use (from groundtruth)
    
    Returns:
        Dictionary with working set statistics
    """
    # Print experiment parameters
    params = {
        "event": "params",
        "script": "calculate_working_set",
        "groundtruth_path": groundtruth_path,
        "n_splits": n_splits,
        "K": K
    }
    print(json.dumps(params))
    
    # Load groundtruth
    print(f"Loading groundtruth from {groundtruth_path}...", file=sys.stderr)
    groundtruth_ids, groundtruth_K = load_groundtruth(groundtruth_path)
    query_num = groundtruth_ids.shape[0]
    print(f"Loaded groundtruth: {query_num} queries with {groundtruth_K} neighbors per query", file=sys.stderr)
    
    # Use the minimum of requested K and groundtruth K
    if groundtruth_K < K:
        print(f"Warning: Groundtruth has {groundtruth_K} neighbors but K={K}. Using K={groundtruth_K}", file=sys.stderr)
        K = groundtruth_K
    
    # Calculate split size
    split_size = (query_num + n_splits - 1) // n_splits
    
    # Working set statistics
    working_set_sizes = []
    working_set_details = []
    
    # Track cumulative unique tags across all splits
    cumulative_tags = set()
    cumulative_unique_per_split = []
    
    print(f"\nProcessing {n_splits} splits (split_size={split_size})...", file=sys.stderr)
    
    # Process each split
    for split_idx in range(n_splits):
        start = split_idx * split_size
        end = min(start + split_size, query_num)
        
        if start >= end:
            # Empty split
            working_set_sizes.append(0)
            cumulative_unique_per_split.append(len(cumulative_tags))
            working_set_details.append({
                "split": split_idx,
                "start_query": start,
                "end_query": end,
                "num_queries": 0,
                "distinct_tags": 0,
                "total_results": 0,
                "cumulative_unique_tags": len(cumulative_tags)
            })
            continue
        
        this_split_size = end - start
        
        print(f"  Processing split {split_idx + 1}/{n_splits} (queries {start}-{end-1})...", 
              file=sys.stderr, flush=True)
        
        # Collect all tags from all queries in this split using groundtruth
        split_tags = set()
        total_results = 0
        
        # Get groundtruth tags for queries in this split
        split_groundtruth = groundtruth_ids[start:end, :K]
        
        for query_idx in range(this_split_size):
            # Get tags from groundtruth for this query
            tags = split_groundtruth[query_idx]
            
            for tag in tags:
                tag_int = int(tag)
                split_tags.add(tag_int)
                cumulative_tags.add(tag_int)  # Track cumulative across all splits
            
            total_results += len(tags)
        
        distinct_tags_count = len(split_tags)
        cumulative_unique_count = len(cumulative_tags)
        
        working_set_sizes.append(distinct_tags_count)
        cumulative_unique_per_split.append(cumulative_unique_count)
        
        working_set_details.append({
            "split": split_idx,
            "start_query": start,
            "end_query": end,
            "num_queries": this_split_size,
            "distinct_tags": distinct_tags_count,
            "total_results": total_results,
            "avg_tags_per_query": total_results / this_split_size if this_split_size > 0 else 0,
            "cumulative_unique_tags": cumulative_unique_count
        })
        
        print(f"  Split {split_idx + 1} completed: {distinct_tags_count} distinct tags "
              f"from {total_results} total results across {this_split_size} queries "
              f"(cumulative: {cumulative_unique_count})", 
              file=sys.stderr)
    
    # Calculate summary statistics
    non_zero_sizes = [s for s in working_set_sizes if s > 0]
    
    summary = {
        "min_working_set": int(min(working_set_sizes)) if working_set_sizes else 0,
        "max_working_set": int(max(working_set_sizes)) if working_set_sizes else 0,
        "avg_working_set": float(np.mean(working_set_sizes)) if working_set_sizes else 0.0,
        "median_working_set": float(np.median(working_set_sizes)) if working_set_sizes else 0.0,
        "std_working_set": float(np.std(working_set_sizes)) if len(working_set_sizes) > 1 else 0.0,
        "min_working_set_non_zero": int(min(non_zero_sizes)) if non_zero_sizes else 0,
        "max_working_set_non_zero": int(max(non_zero_sizes)) if non_zero_sizes else 0,
        "avg_working_set_non_zero": float(np.mean(non_zero_sizes)) if non_zero_sizes else 0.0,
        "total_unique_tags_all_splits": len(cumulative_tags),
        "cumulative_unique_per_split": cumulative_unique_per_split
    }
    
    # Prepare results
    results = {
        "event": "results",
        "working_set_sizes": working_set_sizes,
        "working_set_details": working_set_details,
        "summary": summary
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate working set size (distinct tags) for each query split using groundtruth"
    )
    parser.add_argument("--groundtruth_path", type=str, required=True,
                       help="Path to groundtruth binary file")
    parser.add_argument("--n_splits", type=int, required=True,
                       help="Number of splits to divide queries into")
    parser.add_argument("--K", type=int, required=True,
                       help="Number of nearest neighbors to use from groundtruth")
    
    args = parser.parse_args()
    
    results = calculate_working_set(
        args.groundtruth_path,
        args.n_splits,
        args.K
    )
    
    # Output results as JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

