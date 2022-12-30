mkdir build
cd build
cmake -A x64 ..
msbuild dnn.sln /p:Configuration=Release