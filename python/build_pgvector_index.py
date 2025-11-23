#!/usr/bin/env python3
"""
Helper script to build a pgvector index from binary data files.

This script loads vectors from a binary file (DiskANN format) and uploads them to PostgreSQL with pgvector.
"""

import argparse
import numpy as np
from pgvector_backend import PgVectorBackend
import sys


def main():
    parser = argparse.ArgumentParser(description="Build pgvector index from binary data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to binary data file (DiskANN format)")
    parser.add_argument("--table_name", type=str, default="vectors",
                       help="Name of the PostgreSQL table (default: vectors)")
    parser.add_argument("--db_host", type=str, default="localhost",
                       help="Host of the PostgreSQL service (default: localhost)")
    parser.add_argument("--db_port", type=int, default=5432,
                       help="Port of the PostgreSQL service (default: 5432)")
    parser.add_argument("--db_name", type=str, default="postgres",
                       help="Database name (default: postgres)")
    parser.add_argument("--db_user", type=str, default="postgres",
                       help="Database user (default: postgres)")
    parser.add_argument("--db_password", type=str, default="postgres",
                       help="Database password (default: postgres)")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate table even if it exists")
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
    
    # Initialize pgvector backend (this will create table and load data)
    print(f"Connecting to PostgreSQL at {args.db_host}:{args.db_port}...")
    backend = PgVectorBackend(
        table_name=args.table_name,
        dimension=dimension,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        data_path=args.data_path,
        recreate_table=args.recreate
    )
    
    # Verify the table
    with backend.conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {args.table_name}")
        num_entities = cur.fetchone()[0]
    print(f"\nIndex built successfully!")
    print(f"Table '{args.table_name}' contains {num_entities} vectors")
    print(f"Vector dimension: {dimension}")


if __name__ == "__main__":
    main()

