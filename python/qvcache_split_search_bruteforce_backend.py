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

from bruteforce_backend import BruteforceBackend


def hybrid_search(
    qvcache: qvc.QVCache,
    queries: np.ndarray,
    K: int,
    search_threads: int,
    data_path: str
) -> Tuple[List[bool], List[float], np.ndarray]:
    """
    Perform hybrid search on queries.
    
    Args:
        qvcache: QVCache instance (QVCache object)
        queries: Query vectors as numpy array (query_num, dim)
        K: Number of nearest neighbors
        search_threads: Number of search threads (for logging)
        data_path: Path to data (for logging)
        
    Returns:
        Tuple of (hit_results, latencies_ms, query_result_tags)
    """
    query_num = queries.shape[0]
    query_dim = queries.shape[1]
    
    query_result_tags = np.zeros((query_num, K), dtype=np.uint32)
    query_result_dists = np.zeros((query_num, K), dtype=np.float32)
    hit_results = [False] * query_num
    latencies_ms = [0.0] * query_num
    
    global_start = time.time()
    hit_count = 0
    
    # Search each query
    for i in range(query_num):
        start = time.time()
        query = queries[i].astype(np.float32)
        
        hit, tags, dists = qvcache.search(query, K)
        
        hit_results[i] = hit
        if hit:
            hit_count += 1
        
        # Copy results (handle case where tags/dists might be shorter than K)
        result_len = min(len(tags), K)
        query_result_tags[i, :result_len] = tags[:result_len]
        query_result_dists[i, :result_len] = dists[:result_len]
        
        end = time.time()
        latencies_ms[i] = (end - start) * 1000.0  # Convert to ms
    
    # Calculate statistics
    total_hit_latency_ms = sum(lat for i, lat in enumerate(latencies_ms) if hit_results[i])
    actual_hit_count = sum(hit_results)
    avg_hit_latency_ms = total_hit_latency_ms / actual_hit_count if actual_hit_count > 0 else 0.0
    
    final_ratio = hit_count / query_num if query_num > 0 else 0.0
    print(json.dumps({
        "event": "hit_ratio",
        "hit_ratio": final_ratio,
        "hits": hit_count,
        "total": query_num
    }))
    
    global_end = time.time()
    total_time_ms = (global_end - global_start) * 1000.0
    total_time_sec = total_time_ms / 1000.0
    avg_latency_ms = sum(latencies_ms) / query_num if query_num > 0 else 0.0
    qps = query_num / total_time_sec if total_time_sec > 0 else 0.0
    qps_per_thread = qps / search_threads if search_threads > 0 else 0.0
    
    sorted_latencies = sorted(latencies_ms)
    get_percentile = lambda p: sorted_latencies[int(np.ceil(p * query_num)) - 1] if query_num > 0 else 0.0
    p90 = get_percentile(0.90) if query_num > 0 else 0.0
    p95 = get_percentile(0.95) if query_num > 0 else 0.0
    p99 = get_percentile(0.99) if query_num > 0 else 0.0
    
    # Build mini index vector counts
    num_mini_indexes = qvcache.get_number_of_mini_indexes()
    mini_index_counts = {}
    for i in range(num_mini_indexes):
        mini_index_counts[f"index_{i}_vectors"] = qvcache.get_index_vector_count(i)
    
    print(json.dumps({
        "event": "latency",
        "threads": search_threads,
        "avg_latency_ms": avg_latency_ms,
        "avg_hit_latency_ms": avg_hit_latency_ms,
        "qps": qps,
        "qps_per_thread": qps_per_thread,
        "memory_active_vectors": qvcache.get_number_of_vectors_in_memory_index(),
        "memory_max_points": qvcache.get_number_of_max_points_in_memory_index(),
        "pca_active_regions": qvcache.get_number_of_active_pca_regions(),
        **mini_index_counts,
        "tail_latency_ms": {
            "p90": p90,
            "p95": p95,
            "p99": p99
        }
    }))
    
    return hit_results, latencies_ms, query_result_tags


def calculate_recall(K: int, groundtruth_ids: np.ndarray, query_result_tags: np.ndarray,
                     query_num: int, groundtruth_dim: int):
    """Calculate recall metric."""
    total_recall = 0.0
    for i in range(query_num):
        groundtruth_set = set(groundtruth_ids[i, :K])
        # C++ version subtracts 1 from calculated tags to match groundtruth format
        calculated_set = set(query_result_tags[i, :K] - 1)
        matching = len(groundtruth_set & calculated_set)
        recall = matching / K
        total_recall += recall
    
    avg_recall = total_recall / query_num if query_num > 0 else 0.0
    print(json.dumps({
        "event": "recall",
        "K": K,
        "recall": avg_recall,
        "type": "all"
    }))


def calculate_hit_recall(K: int, groundtruth_ids: np.ndarray, query_result_tags: np.ndarray,
                        hit_results: List[bool], query_num: int, groundtruth_dim: int):
    """Calculate recall for cache hits only."""
    total_recall = 0.0
    hit_count = 0
    for i in range(query_num):
        if hit_results[i]:
            groundtruth_set = set(groundtruth_ids[i, :K])
            # C++ version subtracts 1 from calculated tags to match groundtruth format
            calculated_set = set(query_result_tags[i, :K] - 1)
            matching = len(groundtruth_set & calculated_set)
            recall = matching / K
            total_recall += recall
            hit_count += 1
    
    if hit_count > 0:
        avg_recall = total_recall / hit_count
        print(json.dumps({
            "event": "recall",
            "K": K,
            "recall": avg_recall,
            "type": "cache_hits",
            "hit_count": hit_count
        }))
    else:
        print(json.dumps({
            "event": "recall",
            "K": K,
            "recall": None,
            "type": "cache_hits",
            "hit_count": 0
        }))


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
    backend: BruteforceBackend
):
    """Run the split search experiment."""
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
                    hit_results, _, query_result_tags = hybrid_search(
                        qvcache,
                        split_queries,
                        K,
                        search_threads,
                        data_path
                    )
                    calculate_recall(K, groundtruth_ids[start:end], query_result_tags,
                                   this_split_size, groundtruth_dim)
                    calculate_hit_recall(K, groundtruth_ids[start:end], query_result_tags,
                                        hit_results, this_split_size, groundtruth_dim)
            
            # Second split in the pair
            start2 = (split + 1) * split_size
            end2 = min(start2 + split_size, query_num)
            if start2 < end2:
                this_split_size2 = end2 - start2
                split_queries2 = queries[start2:end2]
                
                for iter in range(n_iteration_per_split):
                    hit_results2, _, query_result_tags2 = hybrid_search(
                        qvcache,
                        split_queries2,
                        K,
                        search_threads,
                        data_path
                    )
                    calculate_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                   this_split_size2, groundtruth_dim)
                    calculate_hit_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                        hit_results2, this_split_size2, groundtruth_dim)
            
            # Second split again (reverse order)
            if start2 < end2:
                this_split_size2 = end2 - start2
                split_queries2 = queries[start2:end2]
                
                for iter in range(n_iteration_per_split):
                    hit_results2, _, query_result_tags2 = hybrid_search(
                        qvcache,
                        split_queries2,
                        K,
                        search_threads,
                        data_path
                    )
                    calculate_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                   this_split_size2, groundtruth_dim)
                    calculate_hit_recall(K, groundtruth_ids[start2:end2], query_result_tags2,
                                        hit_results2, this_split_size2, groundtruth_dim)
            
            # First split again (reverse order)
            if start < end:
                this_split_size = end - start
                split_queries = queries[start:end]
                
                for iter in range(n_iteration_per_split):
                    hit_results, _, query_result_tags = hybrid_search(
                        qvcache,
                        split_queries,
                        K,
                        search_threads,
                        data_path
                    )
                    calculate_recall(K, groundtruth_ids[start:end], query_result_tags,
                                   this_split_size, groundtruth_dim)
                    calculate_hit_recall(K, groundtruth_ids[start:end], query_result_tags,
                                      hit_results, this_split_size, groundtruth_dim)


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
        "search_strategy": args.search_strategy
    }
    print(json.dumps(params))
    
    # Create Python backend
    if args.data_type == "float":
        backend = BruteforceBackend(args.data_path)
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
        args.search_strategy, backend
    )


if __name__ == "__main__":
    main()

