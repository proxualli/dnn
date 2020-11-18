#!/bin/sh
mkdir build
cd build
cmake -G Ninja ..
ninja zlib
cp ~/dnn/deps/zlib-1.2.11/zlib.h ~/dnn/build/deps/zlib-1.2.11
cp ~/dnn/deps/libpng/scripts/pnglibconf.h.prebuilt pnglibconf.h
ninja