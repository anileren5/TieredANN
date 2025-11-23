"""
Python implementation of a Milvus backend for QVCache.

This backend implements the required interface:
- search(query: numpy array, K: int) -> tuple of (tags: numpy array, distances: numpy array)
- fetch_vectors_by_ids(ids: list) -> list of numpy arrays
"""

import numpy as np
from typing import List, Tuple
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
import os


class MilvusBackend:
    """
    A Milvus backend that stores vectors in Milvus and performs approximate nearest neighbor search.
    """
    
    def __init__(self, collection_name: str, dimension: int, 
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530,
                 data_path: str = None,
                 recreate_collection: bool = False):
        """
        Initialize the Milvus backend.
        
        Args:
            collection_name: Name of the Milvus collection
            dimension: Dimension of the vectors
            milvus_host: Host of the Milvus service (default: localhost)
            milvus_port: Port of the Milvus service (default: 19530)
            data_path: Optional path to binary data file to load vectors from (DiskANN format)
            recreate_collection: If True, recreate the collection even if it exists
        """
        # Connect to Milvus with retry logic
        import time
        max_retries = 10
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host=milvus_host,
                    port=milvus_port
                )
                # Test connection by listing collections
                utility.list_collections()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to connect to Milvus at {milvus_host}:{milvus_port} after {max_retries} attempts: {e}")
        
        self.collection_name = collection_name
        # Ensure dimension is a Python int (not numpy type) for JSON serialization
        self.dim = int(dimension)
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        
        # Check if collection exists
        collection_exists = utility.has_collection(collection_name)
        
        if recreate_collection and collection_exists:
            utility.drop_collection(collection_name)
            collection_exists = False
        
        if not collection_exists:
            # Create collection
            # Ensure dimension is a Python int for JSON serialization
            dim_int = int(self.dim)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim_int)
            ]
            schema = CollectionSchema(fields=fields, description="Vector collection for QVCache")
            self.collection = Collection(name=collection_name, schema=schema)
            
            print(f"Created Milvus collection '{collection_name}' with dimension {dimension}")
            
            # Load data first, then create index for better accuracy
            if data_path and os.path.exists(data_path):
                self._load_data_from_file(data_path)
            
            # Create index after data is loaded for optimal recall
            # HNSW generally provides better recall than IVF_FLAT
            print(f"Creating HNSW index...")
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {
                    "M": 16,  # Number of connections per layer
                    "efConstruction": 200  # Build quality (higher = better recall, slower build)
                }
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            self.collection.load()
            print(f"Created HNSW index with M=16, efConstruction=200")
        else:
            self.collection = Collection(name=collection_name)
            self.collection.load()
            num_entities = self.collection.num_entities
            print(f"Using existing Milvus collection '{collection_name}'")
            print(f"Collection contains {num_entities} vectors")
        
        print(f"MilvusBackend initialized with collection '{collection_name}'")
    
    def _load_data_from_file(self, data_path: str, batch_size: int = 1000):
        """
        Load vectors from a binary file (DiskANN format) into Milvus.
        
        Args:
            data_path: Path to the binary data file
            batch_size: Number of vectors to insert per batch
        """
        print(f"Loading vectors from {data_path} into Milvus...")
        
        # Read metadata (first 2 uint32_t: num_vectors, dim)
        with open(data_path, 'rb') as f:
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            if dim != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, got {dim}")
            
            print(f"Loading {num_vectors} vectors of dimension {dim}...")
            
            # Load vectors in batches
            ids = []
            vectors = []
            for i in range(num_vectors):
                vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                vector = vector.astype(np.float32).tolist()
                
                ids.append(int(i))
                vectors.append(vector)
                
                # Insert batch when full
                if len(ids) >= batch_size:
                    entities = [ids, vectors]
                    self.collection.insert(entities)
                    ids = []
                    vectors = []
                    print(f"Loaded {i + 1}/{num_vectors} vectors...", end='\r')
            
            # Insert remaining vectors
            if ids:
                entities = [ids, vectors]
                self.collection.insert(entities)
            
            # Flush to ensure data is written
            self.collection.flush()
            
            print(f"\nLoaded {num_vectors} vectors into Milvus")
    
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for K nearest neighbors using Milvus.
        
        Args:
            query: Query vector as numpy array (1D, shape=(dim,))
            K: Number of nearest neighbors to return
            
        Returns:
            Tuple of (tags, distances) where:
            - tags: numpy array of shape (K,) containing vector IDs
            - distances: numpy array of shape (K,) containing L2 distances
        """
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Query must be 1D array of shape ({self.dim},), got {query.shape}")
        
        # Convert query to list and perform search
        query_vector = query.astype(np.float32).tolist()
        
        # Search parameters
        # For HNSW index, use ef parameter (should be >= K, recommended 2-4x K for good recall)
        # For IVF_FLAT index, use nprobe parameter (higher = better recall but slower)
        search_params = {
            "metric_type": "L2",
            "params": {"ef": max(K * 2, 200)}  # For HNSW: ef should be 2-4x K for good recall
        }
        
        try:
            # Perform search
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=K,
                output_fields=[]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to search Milvus collection: {e}")
        
        # Extract tags and distances
        # IMPORTANT: Milvus returns actual Euclidean distance (with square root), but BruteforceBackend
        # returns squared L2 distance (L2^2). We need to square Milvus distances to match.
        tags = []
        distances = []
        
        if results and len(results) > 0:
            # results is a list of hits for each query (we only have one query)
            for hit in results[0]:
                tags.append(hit.id)
                # Square the distance to match BruteforceBackend format (L2^2)
                distances.append(hit.distance * hit.distance)
        
        # Convert to numpy arrays
        tags = np.ascontiguousarray(np.array(tags, dtype=np.uint32))
        distances = np.ascontiguousarray(np.array(distances, dtype=np.float32))
        
        # Ensure we have exactly K results (pad if needed)
        # Note: We pad with invalid IDs (max uint32) instead of 0, since 0 might be a valid vector ID
        if len(tags) < K:
            padded_tags = np.full(K, np.iinfo(np.uint32).max, dtype=np.uint32)  # Use max uint32 as invalid marker
            padded_distances = np.full(K, np.finfo(np.float32).max, dtype=np.float32)
            padded_tags[:len(tags)] = tags
            padded_distances[:len(distances)] = distances
            tags = padded_tags
            distances = padded_distances
        
        return tags, distances
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Fetch vectors by their IDs from Milvus.
        
        Args:
            ids: List of vector IDs
            
        Returns:
            List of numpy arrays, each representing a vector
        """
        if not ids:
            return []
        
        try:
            # Retrieve vectors from Milvus
            # Format: "id in [1, 2, 3]" for Milvus query expression
            ids_str = ",".join(str(id) for id in ids)
            expr = f"id in [{ids_str}]"
            results = self.collection.query(
                expr=expr,
                output_fields=["vector"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve vectors from Milvus: {e}")
        
        # Create a mapping of ID to vector
        id_to_vector = {}
        for result in results:
            if "id" in result and "vector" in result and result["vector"] is not None:
                id_to_vector[result["id"]] = np.array(result["vector"], dtype=np.float32)
        
        # Return vectors in the same order as requested IDs
        result_vectors = []
        for vec_id in ids:
            if vec_id in id_to_vector:
                result_vectors.append(id_to_vector[vec_id])
            else:
                # Return zero vector for invalid IDs
                result_vectors.append(np.zeros(self.dim, dtype=np.float32))
        
        return result_vectors

