#!/usr/bin/env python3
"""
Helper script to build a Pinecone index from binary data files.

This script loads vectors from a binary file (DiskANN format) and uploads them to Pinecone.
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from backends.pinecone_backend import PineconeBackend


def main():
    parser = argparse.ArgumentParser(description="Build Pinecone index from binary data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to binary data file (DiskANN format)")
    parser.add_argument("--index_name", type=str, default="vectors",
                       help="Name of the Pinecone index (default: vectors). "
                            "For cloud: Check your Pinecone dashboard for the exact index name")
    parser.add_argument("--api_key", type=str, default=None,
                       help="Pinecone API key (default: from PINECONE_API_KEY env var). "
                            "For cloud: Your API key (starts with 'pcsk_'). "
                            "For local: Use 'pclocal' or 'local'")
    parser.add_argument("--environment", type=str, default=None,
                       help="Pinecone environment/region. "
                            "For cloud: REQUIRED! (e.g., 'us-east-1', 'us-west-2', 'eu-west-1'). "
                            "For local: Not needed")
    parser.add_argument("--host", type=str, default=None,
                       help="Pinecone host (for local Docker, use service name like 'pinecone')")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate index even if it exists")
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
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("PINECONE_API_KEY", "local")  # "local" will be converted to "pclocal" in backend
    
    # Initialize Pinecone backend (this will create index and load data)
    print(f"Connecting to Pinecone...")
    if args.environment:
        print(f"Using environment: {args.environment}")
    backend = PineconeBackend(
        index_name=args.index_name,
        dimension=dimension,
        api_key=api_key,
        environment=args.environment,
        host=args.host,
        data_path=args.data_path,
        recreate_index=args.recreate
    )
    
    # Verify the index
    try:
        stats = backend.index.describe_index_stats()
        num_entities = stats.get('total_vector_count', 0)
        print(f"\nIndex built successfully!")
        print(f"Index '{args.index_name}' contains {num_entities} vectors")
        print(f"Vector dimension: {dimension}")
    except Exception as e:
        print(f"\nIndex created, but could not verify stats: {e}")


if __name__ == "__main__":
    main()

