@ECHO OFF

IF "%~1"=="" GOTO release

mkdir build
cd build
cmake -A x64 .. -DCMAKE_BUILD_TYPE=%1 -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
msbuild dnn.sln /p:Configuration=%1
cd ..
GOTO :EOF

:release
mkdir build
cd build
cmake -A x64 .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
msbuild dnn.sln /p:Configuration=Release
cd ..


