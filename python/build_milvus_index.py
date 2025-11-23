#!/usr/bin/env python3
"""
Helper script to build a Milvus index from binary data files.

This script loads vectors from a binary file (DiskANN format) and uploads them to Milvus.
"""

import argparse
import numpy as np
from milvus_backend import MilvusBackend
from pymilvus import utility
import sys


def main():
    parser = argparse.ArgumentParser(description="Build Milvus index from binary data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to binary data file (DiskANN format)")
    parser.add_argument("--collection_name", type=str, default="vectors",
                       help="Name of the Milvus collection (default: vectors)")
    parser.add_argument("--milvus_host", type=str, default="localhost",
                       help="Host of the Milvus service (default: localhost)")
    parser.add_argument("--milvus_port", type=int, default=19530,
                       help="Port of the Milvus service (default: 19530)")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate collection even if it exists")
    parser.add_argument("--dimension", type=int, default=None,
                       help="Vector dimension (will be read from file if not provided)")
    
    args = parser.parse_args()
    
    # Read metadata to get dimension
    if args.dimension is None:
        with open(args.data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dimension = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])  # Convert to Python int
        print(f"Detected {num_vectors} vectors of dimension {dimension} in {args.data_path}")
    else:
        dimension = int(args.dimension)  # Ensure it's a Python int
    
    # Initialize Milvus backend (this will create collection and load data)
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}...")
    backend = MilvusBackend(
        collection_name=args.collection_name,
        dimension=dimension,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        data_path=args.data_path,
        recreate_collection=args.recreate
    )
    
    # Verify the collection
    collection = backend.collection
    num_entities = collection.num_entities
    print(f"\nIndex built successfully!")
    print(f"Collection '{args.collection_name}' contains {num_entities} vectors")
    print(f"Vector dimension: {dimension}")


if __name__ == "__main__":
    main()

