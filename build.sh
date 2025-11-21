#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make -j"$(nproc)" qvcache_python_module