#!/bin/bash
sudo apt-get update
sudo apt-get install build-essential curl unzip clang nasm graphviz doxygen cmake ninja-build

export CC=clang and export CXX=clang++
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_DISPLAY_ENV=TRUE
export KMP_SETTINGS=TRUE
export KMP_BLOCKTIME=0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$((vCPUs / 2))

mkdir -p build && cd build && cmake -G Ninja .. -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON && ninja