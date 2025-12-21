#!/bin/bash
# Build script for SPTAG
# This script automates the build process including zstd dependency setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "Building SPTAG"
echo "=========================================="

# Step 1: Set up zstd dependency
echo "Setting up zstd dependency..."
if [ ! -d "ThirdParty/zstd" ] || [ -z "$(ls -A ThirdParty/zstd 2>/dev/null)" ]; then
    echo "Cloning zstd repository..."
    rm -rf ThirdParty/zstd
    mkdir -p ThirdParty/zstd
    cd ThirdParty/zstd
    if ! git clone -b release https://github.com/facebook/zstd.git .; then
        echo "Error: Failed to clone zstd repository"
        echo "Please ensure git is installed: apt-get update && apt-get install -y git"
        exit 1
    fi
    cd "$SCRIPT_DIR"
else
    echo "zstd dependency already exists, skipping clone..."
fi

# Step 2: Set compiler environment variables
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8

# Step 3: Create build directory
echo "Creating build directory..."
rm -rf build
mkdir -p build
cd build

# Step 4: Configure with CMake
echo "Configuring with CMake..."
if ! cmake -DSPDK=OFF -DROCKSDB=OFF ..; then
    echo "Error: CMake configuration failed"
    exit 1
fi

# Step 5: Build
echo "Building SPTAG (this may take a while)..."
if ! make -j$(nproc); then
    echo "Error: Build failed"
    exit 1
fi

# Step 6: Go back to root
cd ..

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Binaries should be in the Release/ directory"
echo "To verify, run: ls -lh Release/"

