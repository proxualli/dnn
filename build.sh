#!/bin/bash
sudo apt-get install nasm
sudo apt-get install doxygen
sudo apt-get install cmake
sudo apt-get install ninja-build
export CC=clang and export CXX=clang++
export KMP_AFFINITY=granularity=fine,compact,1,0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$((vCPUs / 2))
export OMP_DISPLAY_ENV=TRUE
export KMP_SETTINGS=TRUE
mkdir -p build && cd build && cmake -G Ninja ..
ninja