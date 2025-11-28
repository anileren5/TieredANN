#!/usr/bin/env python3
"""
Simple bruteforce backend benchmark without QVCache.

This script loads queries once, performs searches once for all queries,
and logs metrics (latency, recall, QPS) in JSON format.
"""

import argparse
import time
import numpy as np
import json
import sys

from backends.bruteforce_backend import BruteforceBackend


def load_binary_queries(query_path: str) -> np.ndarray:
    """
    Load queries from binary file (DiskANN format).
    
    Format:
    - First 4 bytes: num_queries (uint32_t)
    - Next 4 bytes: dimension (uint32_t)
    - Rest: num_queries * dimension * sizeof(float) bytes of query vectors
    """
    with open(query_path, 'rb') as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        
        queries = np.frombuffer(f.read(num_queries * dim * 4), dtype=np.float32)
        queries = queries.reshape(num_queries, dim)
    
    return queries


def load_groundtruth(groundtruth_path: str) -> tuple:
    """
    Load groundtruth from binary file (DiskANN format).
    
    Format (int32 for metadata):
    - First 4 bytes: num_queries (int32)
    - Next 4 bytes: K (int32)
    - Rest: num_queries * K * sizeof(uint32_t) bytes of groundtruth IDs
    - Optionally: num_queries * K * sizeof(float) bytes of distances
    """
    import os
    
    file_size = os.path.getsize(groundtruth_path)
    
    with open(groundtruth_path, 'rb') as f:
        # Read metadata (int32 format, not uint32)
        num_queries = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        K = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        
        # Calculate expected file sizes
        expected_size_ids_only = 2 * 4 + num_queries * K * 4
        expected_size_with_dists = 2 * 4 + num_queries * K * 4 + num_queries * K * 4
        
        # Read IDs
        ids_size = num_queries * K * 4
        groundtruth_ids = np.frombuffer(f.read(ids_size), dtype=np.uint32)
        groundtruth_ids = groundtruth_ids.reshape(num_queries, K)
        
        # Skip distances if present (we don't need them for recall calculation)
    
    return groundtruth_ids, K


def experiment_bruteforce_only(
    data_path: str,
    query_path: str,
    groundtruth_path: str,
    K: int,
    metric: str = "l2",
    progress_interval: int = 100
):
    """
    Run bruteforce backend experiment without QVCache.
    
    Args:
        data_path: Path to data binary file
        query_path: Path to query binary file
        groundtruth_path: Path to groundtruth binary file
        K: Number of nearest neighbors to return
        metric: Distance metric ("l2" or "cosine")
    """
    # Print experiment parameters
    params = {
        "event": "params",
        "backend": "bruteforce_only",
        "data_path": data_path,
        "query_path": query_path,
        "groundtruth_path": groundtruth_path,
        "K": K,
        "metric": metric
    }
    print(json.dumps(params))
    
    # Initialize bruteforce backend
    print(f"\nInitializing BruteforceBackend...")
    backend = BruteforceBackend(data_path, metric=metric)
    
    # Load queries
    print(f"Loading queries from {query_path}...")
    queries = load_binary_queries(query_path)
    query_num, query_dim = queries.shape
    print(f"Loaded {query_num} queries of dimension {query_dim}")
    
    # Load groundtruth
    print(f"Loading groundtruth from {groundtruth_path}...")
    groundtruth_ids, groundtruth_K = load_groundtruth(groundtruth_path)
    print(f"Loaded groundtruth with {groundtruth_K} neighbors per query")
    
    if groundtruth_K < K:
        print(f"Warning: Groundtruth has {groundtruth_K} neighbors but K={K}. Using K={groundtruth_K}")
        K = groundtruth_K
    
    # Prepare result arrays
    query_result_tags = np.zeros((query_num, K), dtype=np.uint32)
    query_result_dists = np.zeros((query_num, K), dtype=np.float32)
    latencies_ms = np.zeros(query_num, dtype=np.float64)
    
    # Perform searches for all queries
    print(f"\nPerforming searches for {query_num} queries...")
    global_start = time.time()
    
    INVALID_ID = np.iinfo(np.uint32).max
    query_recalls = []
    
    for i in range(query_num):
        query_start = time.time()
        
        # Search
        tags, dists = backend.search(queries[i], K)
        
        # Store results
        result_len = min(len(tags), K)
        query_result_tags[i, :result_len] = tags[:result_len]
        query_result_dists[i, :result_len] = dists[:result_len]
        
        query_end = time.time()
        latencies_ms[i] = (query_end - query_start) * 1000.0  # Convert to ms
        
        # Calculate recall for this query
        groundtruth_set = set(groundtruth_ids[i, :K])
        valid_tags = query_result_tags[i, :K]
        valid_mask = valid_tags != INVALID_ID
        valid_tags_filtered = valid_tags[valid_mask]
        
        if len(valid_tags_filtered) > 0:
            # Bruteforce backend returns 0-based tags (array indices), same as groundtruth
            calculated_set = set(valid_tags_filtered)
            matching = len(groundtruth_set & calculated_set)
            recall = matching / K
        else:
            recall = 0.0
        
        query_recalls.append(recall)
        
        # Print progress
        if (i + 1) % progress_interval == 0 or (i + 1) == query_num:
            elapsed = time.time() - global_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            avg_recall_so_far = np.mean(query_recalls)
            current_recall = recall
            print(f"Progress: {i + 1}/{query_num} queries completed ({100.0 * (i + 1) / query_num:.1f}%) - "
                  f"Rate: {rate:.1f} q/s - Current recall: {current_recall:.4f} - Avg recall: {avg_recall_so_far:.4f}", 
                  flush=True)
    
    global_end = time.time()
    
    # Calculate metrics
    total_time_ms = (global_end - global_start) * 1000.0
    total_time_sec = total_time_ms / 1000.0
    avg_latency_ms = np.mean(latencies_ms)
    qps = query_num / total_time_sec if total_time_sec > 0 else 0.0
    
    # Calculate percentiles
    sorted_latencies = np.sort(latencies_ms)
    p50 = np.percentile(sorted_latencies, 50)
    p90 = np.percentile(sorted_latencies, 90)
    p95 = np.percentile(sorted_latencies, 95)
    p99 = np.percentile(sorted_latencies, 99)
    
    # Calculate final recall statistics (recalls already calculated during search loop)
    avg_recall = np.mean(query_recalls)
    low_recall_count = sum(1 for r in query_recalls if r < 0.5)
    very_low_recall_count = sum(1 for r in query_recalls if r < 0.1)
    
    recall_metrics = {
        "recall_all": float(avg_recall),
        "low_recall_queries": low_recall_count,
        "very_low_recall_queries": very_low_recall_count
    }
    
    # Log results
    results = {
        "event": "results",
        "backend": "bruteforce_only",
        "total_queries": int(query_num),
        "K": int(K),
        "metric": metric,
        "total_time_ms": total_time_ms,
        "total_time_sec": total_time_sec,
        "avg_latency_ms": float(avg_latency_ms),
        "qps": float(qps),
        "tail_latency_ms": {
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99)
        },
        **recall_metrics
    }
    
    print(json.dumps(results))
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total queries: {query_num}")
    print(f"Total time: {total_time_sec:.2f}s")
    print(f"Average latency: {avg_latency_ms:.3f}ms")
    print(f"QPS: {qps:.2f}")
    print(f"Recall: {recall_metrics.get('recall_all', 0.0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Bruteforce backend benchmark without QVCache")
    
    parser.add_argument("--data_path", required=True, help="Path to data binary file")
    parser.add_argument("--query_path", required=True, help="Path to query binary file")
    parser.add_argument("--groundtruth_path", required=True, help="Path to groundtruth binary file")
    parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbors")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"],
                       help="Distance metric (default: l2)")
    parser.add_argument("--progress_interval", type=int, default=100,
                       help="Progress update interval (print every N queries, default: 100)")
    
    args = parser.parse_args()
    
    experiment_bruteforce_only(
        args.data_path,
        args.query_path,
        args.groundtruth_path,
        args.K,
        args.metric,
        args.progress_interval
    )


if __name__ == "__main__":
    main()

