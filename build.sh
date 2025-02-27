#!/bin/bash
sudo apt update && sudo apt upgrade
sudo apt install build-essential curl unzip gzip libomp-dev clang nasm graphviz doxygen python3-sphinx cmake ninja-build

export CC=clang and export CXX=clang++
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_DISPLAY_ENV=TRUE
export KMP_SETTINGS=TRUE
export KMP_BLOCKTIME=0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$(vCPUs)
export ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS=1

rm -f -r ./build && mkdir -p build && cd build && cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON && ninja