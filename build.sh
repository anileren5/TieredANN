#!/bin/bash
set -e

mkdir -p build
cd build
cmake ..
make -j"$(nproc)" tieredann_python_module