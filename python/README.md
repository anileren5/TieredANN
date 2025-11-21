# QVCache Python Bindings

Python bindings for QVCache - A tiered memory/disk approximate nearest neighbor search library.

## Installation

### From Source

1. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake ninja-build \
    libeigen3-dev libboost-dev libboost-program-options-dev \
    python3-dev python3-pip
```

2. Install Python dependencies:
```bash
pip install numpy pybind11
```

3. Clone the repository and install:
```bash
git clone <repository-url>
cd QVCache/python
pip install .
```

### Development Installation

For development with editable install:

```bash
pip install -e .
```

## Usage

```python
import qvcache
from bruteforce_backend import BruteforceBackend
import numpy as np

# Create a backend
backend = BruteforceBackend('data/sift/sift_base.bin')

# Create QVCache
index = qvcache.QVCache(
    data_path='data/sift/sift_base.bin',
    pca_prefix='./index/sift/sift',
    R=64, memory_L=128, B=8, M=8, alpha=1.2,
    build_threads=1, search_threads=1,
    use_reconstructed_vectors=False, p=0.75, deviation_factor=0.05,
    memory_index_max_points=200000, beamwidth=2,
    backend=backend
)

# Search
queries, _, _ = qvcache.load_aligned_binary_data('data/sift/sift_query.bin')
query = queries[0].astype(np.float32)
hit, tags, dists = index.search(query, 10)
```

## Custom Backends

You can implement custom Python backends by creating a class with `search()` and `fetch_vectors_by_ids()` methods:

```python
import numpy as np
from typing import List, Tuple

class MyCustomBackend:
    def search(self, query: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        # Returns (tags, distances)
        pass
    
    def fetch_vectors_by_ids(self, ids: List[int]) -> List[np.ndarray]:
        # Returns list of vectors
        pass
```

## Requirements

- Python >= 3.7
- NumPy >= 1.21.0
- CMake >= 3.12
- C++17 compatible compiler
- See system dependencies above

## License

MIT License

