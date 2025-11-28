#!/usr/bin/env python3
"""
Python version of qvcache_split_search_bruteforce_backend.cpp

This script allows running the same test as the C++ version but using
a Python-implemented bruteforce backend.
"""

import argparse
import time
import numpy as np
import json
import sys
from typing import List, Tuple

# Import the compiled qvcache module
try:
    import qvcache as qvc
except ImportError:
    print("Error: qvcache module not found. Please build the Python bindings first.")
    print("Run: cd build && cmake .. && make")
    sys.exit(1)

from backends.bruteforce_backend import BruteforceBackend
from benchmarks.utils import hybrid_search, calculate_recall, calculate_hit_recall, log_split_metrics


def experiment_split(
    data_path: str,
    query_path: str,
    groundtruth_path: str,
    pca_prefix: str,
    R: int,
    memory_L: int,
    K: int,
    B: int,
    M: int,
    alpha: float,
    build_threads: int,
    search_threads: int,
    beamwidth: int,
    use_reconstructed_vectors: int,
    p: float,
    deviation_factor: float,
    n_iteration_per_split: int,
    memory_index_max_points: int,
    use_regional_theta: bool,
    pca_dim: int,
    buckets_per_dim: int,
    n_splits: int,
    n_rounds: int,
    n_async_insert_threads: int,
    lazy_theta_updates: bool,
    number_of_mini_indexes: int,
    search_mini_indexes_in_parallel: bool,
    max_search_threads: int,
    search_strategy: str,
    backend: BruteforceBackend,
    metric: str = "l2"
):
    """Run the split search experiment."""
    # Convert metric string to enum
    if metric.lower() == "cosine":
        metric_enum = qvc.Metric.COSINE
    elif metric.lower() == "l2":
        metric_enum = qvc.Metric.L2
    else:
        raise ValueError(f"Unsupported metric: {metric}. Supported: l2, cosine")
    
    # Create QVCache with Python backend
    qvcache = qvc.QVCache(
        data_path=data_path,
        pca_prefix=pca_prefix,
        R=R,
        memory_L=memory_L,
        B=B,
        M=M,
        alpha=alpha,
        build_threads=build_threads,
        search_threads=search_threads,
        use_reconstructed_vectors=bool(use_reconstructed_vectors),
        p=p,
        deviation_factor=deviation_factor,
        memory_index_max_points=memory_index_max_points,
        beamwidth=beamwidth,
        use_regional_theta=use_regional_theta,
        pca_dim=pca_dim,
        buckets_per_dim=buckets_per_dim,
        n_async_insert_threads=n_async_insert_threads,
        lazy_theta_updates=lazy_theta_updates,
        number_of_mini_indexes=number_of_mini_indexes,
        search_mini_indexes_in_parallel=search_mini_indexes_in_parallel,
        max_search_threads=max_search_threads,
        metric=metric_enum,
        backend=backend
    )
    
    # Set search strategy
    if search_strategy == "SEQUENTIAL_LRU_STOP_FIRST_HIT":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_STOP_FIRST_HIT)
    elif search_strategy == "SEQUENTIAL_LRU_ADAPTIVE":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_LRU_ADAPTIVE)
        qvcache.enable_adaptive_strategy(True)
        qvcache.set_hit_ratio_window_size(100)
        qvcache.set_hit_ratio_threshold(0.90)
    elif search_strategy == "SEQUENTIAL_ALL":
        qvcache.set_search_strategy(qvc.SearchStrategy.SEQUENTIAL_ALL)
    elif search_strategy == "PARALLEL":
        qvcache.set_search_strategy(qvc.SearchStrategy.PARALLEL)
    else:
        print(f"Unknown search strategy: {search_strategy}", file=sys.stderr)
        print("Available strategies: SEQUENTIAL_LRU_STOP_FIRST_HIT, SEQUENTIAL_LRU_ADAPTIVE, SEQUENTIAL_ALL, PARALLEL", file=sys.stderr)
        sys.exit(1)
    
    # Load ground truth
    groundtruth_ids, groundtruth_dists = qvc.load_ground_truth_data(groundtruth_path)
    n_groundtruth, groundtruth_dim = groundtruth_ids.shape
    
    # Load queries
    queries, query_dim, query_aligned_dim = qvc.load_aligned_binary_data(query_path)
    query_num = queries.shape[0]
    
    # Ensure queries are float32
    queries = queries.astype(np.float32)
    
    # Split queries
    split_size = (query_num + n_splits - 1) // n_splits
    
    for round in range(n_rounds):
        for split in range(0, n_splits, 2):
            # Process splits in pattern: 1, 2, 2, 1, 3, 4, 4, 3, 5, 6, 6, 5, etc.
            
            # First split in the pair
            start = split * split_size
            end = min(start + split_size, query_num)
            if start < end:
                this_split_size = end - start
                split_queries = queries[start:end]
                
                for iter in range(n_iteration_per_split):
                    hit_results, _, query_result_tags, metrics = hybrid_search(
                        qvcache,
                        split_queries,
                        K,
                        search_threads,
                        data_path
                    )
                    recall_all = calculate_recall(K, groundtruth_ids[start:end], query_result_tags,
                                                 this_split_size, groundtruth_dim)
                    recall_hits = calculate_hit_recall(K, groundtruth_ids[start:end], query_result_tags,
                                                      hit_results, this_split_size, groundtruth_dim)
                    log_split_metrics(metrics, recall_all, recall_hits)
            
            # Second split in the pair
            start2 = (split + 1) * split_size
            end2 = min(start2 + split_size, query_num)
            if start2 < end2:
                this_split_size2 = end2 - start2
                split_queries2 = queries[start2:end2]
                
                for iter in range(n_iteration_per_split):
                    hit_results2, _, query_result_tags2, metrics2 = hybrid_search(
                        qvcache,
                        split_queries2,
                        K,
                        search_threads,
                        data_path
                    )
                    recall_all2 = calculate_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                                 this_split_size2, groundtruth_dim)
                    recall_hits2 = calculate_hit_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                                       hit_results2, this_split_size2, groundtruth_dim)
                    log_split_metrics(metrics2, recall_all2, recall_hits2)
            
            # Second split again (reverse order)
            if start2 < end2:
                this_split_size2 = end2 - start2
                split_queries2 = queries[start2:end2]
                
                for iter in range(n_iteration_per_split):
                    hit_results2, _, query_result_tags2, metrics2 = hybrid_search(
                        qvcache,
                        split_queries2,
                        K,
                        search_threads,
                        data_path
                    )
                    recall_all2 = calculate_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                                 this_split_size2, groundtruth_dim)
                    recall_hits2 = calculate_hit_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                                       hit_results2, this_split_size2, groundtruth_dim)
                    log_split_metrics(metrics2, recall_all2, recall_hits2)
            
            # First split again (reverse order)
            if start < end:
                this_split_size = end - start
                split_queries = queries[start:end]
                
                for iter in range(n_iteration_per_split):
                    hit_results, _, query_result_tags, metrics = hybrid_search(
                        qvcache,
                        split_queries,
                        K,
                        search_threads,
                        data_path
                    )
                    recall_all = calculate_recall(K, groundtruth_ids[start:end], query_result_tags,
                                                 this_split_size, groundtruth_dim)
                    recall_hits = calculate_hit_recall(K, groundtruth_ids[start:end], query_result_tags,
                                                      hit_results, this_split_size, groundtruth_dim)
                    log_split_metrics(metrics, recall_all, recall_hits)


def main():
    parser = argparse.ArgumentParser(description="QVCache split search with Python backend")
    
    parser.add_argument("--data_type", required=True, choices=["float", "int8", "uint8"],
                       help="Type of data")
    parser.add_argument("--data_path", required=True, help="Path to data")
    parser.add_argument("--query_path", required=True, help="Path to query")
    parser.add_argument("--groundtruth_path", required=True, help="Path to groundtruth")
    parser.add_argument("--pca_prefix", required=True, help="Prefix for PCA files")
    parser.add_argument("--R", type=int, required=True, help="Value of R")
    parser.add_argument("--memory_L", type=int, required=True, help="Value of memory L")
    parser.add_argument("--K", type=int, required=True, help="Value of K")
    parser.add_argument("--B", type=int, default=8, help="Value of B")
    parser.add_argument("--M", type=int, default=8, help="Value of M")
    parser.add_argument("--build_threads", type=int, required=True, help="Threads for building")
    parser.add_argument("--search_threads", type=int, required=True, help="Threads for searching")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha parameter")
    parser.add_argument("--use_reconstructed_vectors", type=int, default=0,
                       help="Use reconstructed vectors for insertion to memory index")
    parser.add_argument("--beamwidth", type=int, default=2, help="Beamwidth")
    parser.add_argument("--p", type=float, default=0.75, help="Value of p")
    parser.add_argument("--deviation_factor", type=float, default=0.05, help="Value of deviation factor")
    parser.add_argument("--n_iteration_per_split", type=int, required=True,
                       help="Number of search iterations per split")
    parser.add_argument("--use_regional_theta", type=lambda x: (str(x).lower() == 'true'), default=True,
                       help="Use regional theta (True/False)")
    parser.add_argument("--pca_dim", type=int, required=True, help="Value of PCA dimension")
    parser.add_argument("--buckets_per_dim", type=int, required=True, help="Value of buckets per dimension")
    parser.add_argument("--memory_index_max_points", type=int, required=True,
                       help="Max points for memory index")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits for queries")
    parser.add_argument("--n_rounds", type=int, default=1, help="Number of rounds to repeat all splits")
    parser.add_argument("--n_async_insert_threads", type=int, default=4, help="Number of async insert threads")
    parser.add_argument("--lazy_theta_updates", type=lambda x: (str(x).lower() == 'true'), default=True,
                       help="Enable lazy theta updates (True/False)")
    parser.add_argument("--number_of_mini_indexes", type=int, default=2,
                       help="Number of mini indexes for shadow cycling")
    parser.add_argument("--search_mini_indexes_in_parallel", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="Search mini indexes in parallel (True/False)")
    parser.add_argument("--max_search_threads", type=int, default=32,
                       help="Maximum threads for parallel search")
    parser.add_argument("--search_strategy", type=str, default="SEQUENTIAL_LRU_STOP_FIRST_HIT",
                       choices=["SEQUENTIAL_LRU_STOP_FIRST_HIT", "SEQUENTIAL_LRU_ADAPTIVE",
                               "SEQUENTIAL_ALL", "PARALLEL"],
                       help="Search strategy")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"],
                       help="Distance metric (default: l2)")
    
    args = parser.parse_args()
    
    # Print parameters
    params = {
        "event": "params",
        "data_type": args.data_type,
        "data_path": args.data_path,
        "query_path": args.query_path,
        "groundtruth_path": args.groundtruth_path,
        "pca_prefix": args.pca_prefix,
        "R": args.R,
        "memory_L": args.memory_L,
        "K": args.K,
        "B": args.B,
        "M": args.M,
        "build_threads": args.build_threads,
        "search_threads": args.search_threads,
        "alpha": args.alpha,
        "use_reconstructed_vectors": args.use_reconstructed_vectors,
        "beamwidth": args.beamwidth,
        "p": args.p,
        "deviation_factor": args.deviation_factor,
        "n_iteration_per_split": args.n_iteration_per_split,
        "use_regional_theta": args.use_regional_theta,
        "pca_dim": args.pca_dim,
        "buckets_per_dim": args.buckets_per_dim,
        "memory_index_max_points": args.memory_index_max_points,
        "n_splits": args.n_splits,
        "n_rounds": args.n_rounds,
        "n_async_insert_threads": args.n_async_insert_threads,
        "lazy_theta_updates": args.lazy_theta_updates,
        "number_of_mini_indexes": args.number_of_mini_indexes,
        "search_mini_indexes_in_parallel": args.search_mini_indexes_in_parallel,
        "max_search_threads": args.max_search_threads,
        "search_strategy": args.search_strategy,
        "metric": args.metric
    }
    print(json.dumps(params))
    
    # Create Python backend
    if args.data_type == "float":
        backend = BruteforceBackend(args.data_path, metric=args.metric)
    else:
        print(f"Unsupported data type for Python backend: {args.data_type}", file=sys.stderr)
        print("Note: Python backend currently only supports float32", file=sys.stderr)
        sys.exit(1)
    
    # Run experiment
    experiment_split(
        args.data_path, args.query_path, args.groundtruth_path, args.pca_prefix,
        args.R, args.memory_L, args.K, args.B, args.M, args.alpha,
        args.build_threads, args.search_threads, args.beamwidth,
        args.use_reconstructed_vectors, args.p, args.deviation_factor,
        args.n_iteration_per_split, args.memory_index_max_points,
        args.use_regional_theta, args.pca_dim, args.buckets_per_dim,
        args.n_splits, args.n_rounds, args.n_async_insert_threads,
        args.lazy_theta_updates, args.number_of_mini_indexes,
        args.search_mini_indexes_in_parallel, args.max_search_threads,
        args.search_strategy, backend, args.metric
    )


if __name__ == "__main__":
    main()

