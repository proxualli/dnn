cmake_minimum_required(VERSION 3.7...3.19)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

message(STATUS "Using CMake version ${CMAKE_VERSION}")

set(DNN_VERSION_MAJOR 1)
set(DNN_VERSION_MINOR 0)
set(DNN_VERSION_PATCH 0)
set(DNN_VERSION_NOTE "")

set(DNN_VERSION ${DNN_VERSION_MAJOR}.${DNN_VERSION_MINOR}.${DNN_VERSION_PATCH})
math(EXPR DNN_VERSION_NUMBER "10000*${DNN_VERSION_MAJOR} + 100*${DNN_VERSION_MINOR} + ${DNN_VERSION_PATCH}")

#####
# Enables link_directories() treat paths relative
# to the source dir.
if(POLICY CMP0015)
    cmake_policy(SET CMP0015 NEW)
endif(POLICY CMP0015)

#####
# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

project(dnn VERSION ${DNN_VERSION} LANGUAGES ASM C CXX)

INCLUDE(GNUInstallDirs)
INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCXXSourceCompiles)

option(BUILD_SHARED_LIBS "Build shared library" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32 OR MSVC)
  set (debug_flags /W4 /Od /arch:AVX2 /openmp:experimental)
  set (release_flags /W3 /O2 /Ob2 /Oi /Ot /GL /GF /Gy /fp:fast /arch:AVX2 /openmp:experimental)
else()
  set (debug_flags -g -Wextra -mfma -mavx2 -march=znver2 -fopenmp -fPIC)
  set (release_flags -Ofast -funroll-loops -ftree-vectorize -Wno-error -mfma -mavx2 -march=znver2 -fopenmp -ffast-math -fPIC)
endif()

MACRO(DNN_TARGET_ENABLE_CXX17 target)
  SET_TARGET_PROPERTIES(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS ON)
  target_compile_options(${target} PUBLIC
    "$<$<CONFIG:RELEASE>:${release_flags}>"
    "$<$<CONFIG:DEBUG>:${debug_flags}>")
ENDMACRO()

set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(WIN32 OR MSVC)
  set(DNNL_ARCH_OPT_FLAGS "/arch:AVX2" CACHE STRING "" FORCE)
else()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(DNNL_ARCH_OPT_FLAGS "-O3 -funroll-loops -ftree-vectorize -Wno-error -mfma -mavx2 -march=znver2 -fopenmp -fPIC" CACHE STRING "" FORCE)
  else()
    set(DNNL_ARCH_OPT_FLAGS "-O3 -funroll-loops -ftree-vectorize -Wno-error -mfma -mavx2 -march=znver2 -fopenmp -fPIC -unroll-aggressive" CACHE STRING "" FORCE)
  endif()
endif()
set(WITH_TURBOJPEG FALSE CACHE BOOL "" FORCE) 
set(PNG_EXECUTABLES FALSE CACHE BOOL "" FORCE)
set(PNG_TESTS FALSE CACHE BOOL "" FORCE)
set(PNG_BUILD_ZLIB ON CACHE BOOL "" FORCE)

if(BUILD_SHARED_LIBS)
  set(DNNL_LIBRARY_TYPE "SHARED" CACHE STRING "" FORCE)
  set(PNG_SHARED TRUE CACHE BOOL "" FORCE)
  set(PNG_STATIC FALSE CACHE BOOL "" FORCE)
  set(ENABLE_SHARED TRUE CACHE BOOL "" FORCE)
  set(ENABLE_STATIC FALSE CACHE BOOL "" FORCE)
else()
  set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
  set(PNG_SHARED FALSE CACHE BOOL "" FORCE)
  set(PNG_STATIC TRUE CACHE BOOL "" FORCE)
  set(ENABLE_SHARED FALSE CACHE BOOL "" FORCE)
  set(ENABLE_STATIC TRUE CACHE BOOL "" FORCE)
endif()

set(libdnn_headers
  include/Activation.h
  include/Add.h
  include/AlignedAllocator.h
  include/Average.h
  include/AvgPooling.h
  include/BatchNorm.h
  include/BatchNormActivation.h
  include/BatchNormActivationDropout.h
  include/BatchNormRelu.h
  include/ChannelMultiply.h
  include/ChannelShuffle.h
  include/ChannelSplit.h
  include/ChannelMultiply.h
  include/ChannelZeroPad.h
  include/Concat.h
  include/Convolution.h
  include/ConvolutionTranspose.h
  include/Cost.h
  include/Dataprovider.h
  include/Definition.h
  include/Dense.h
  include/DepthwiseConvolution.h
  include/Divide.h
  include/Dropout.h
  include/GlobalAvgPooling.h
  include/GlobalMaxPooling.h
  include/Image.h
  include/Input.h
  include/Layer.h
  include/LocalResponseNormalization.h
  include/Max.h
  include/MaxPooling.h
  include/Min.h
  include/Model.h
  include/Multiply.h
  include/ParallelFor.h
  include/PartialDepthwiseConvolution.h
  include/Resampling.h
  include/Scripts.h
  include/stdafx.h
  include/Substract.h
  include/Utils.h
  include/targetver.h
)

set(libdnn_sources
  ${libdnn_headers}
  src/dllmain.cpp
  src/stdafx.cpp
)

set(libdnn_test
  src/test.cpp
)

# ---[ Download deps
SET(DNN_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
SET(DNN_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

IF(NOT DEFINED ZLIB_SOURCE_DIR)
  MESSAGE(STATUS "Downloading ZLib to ${DNN_DEPENDENCIES_SOURCE_DIR}/zlib-1.2.11 (define ZLIB_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadZLIB.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-download")
  SET(ZLIB_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/zlib-1.2.11" CACHE STRING "zlib source directory")
  SET(ZLIB_INCLUDE_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-1.2.11" CACHE STRING "zlib include directory")
  set(ZLIB_LIBRARY zlib)
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${ZLIB_SOURCE_DIR}/zlib.h" "${ZLIB_INCLUDE_DIR}/zlib.h")
 ENDIF()
 
 IF(NOT DEFINED PNG_SOURCE_DIR)
  MESSAGE(STATUS "Downloading PNG to ${DNN_DEPENDENCIES_SOURCE_DIR}/libpng (define PNG_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPNG.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/png-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/png-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/png-download")
  SET(PNG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libpng" CACHE STRING "PNG source directory")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E copy "${PNG_SOURCE_DIR}/scripts/pnglibconf.h.prebuilt" "${DNN_DEPENDENCIES_BINARY_DIR}/libpng/scripts/pnglibconf.h")
 ENDIF()

IF(NOT DEFINED VECTORCLASS_SOURCE_DIR)
  MESSAGE(STATUS "Downloading VectorClass to ${DNN_DEPENDENCIES_SOURCE_DIR}/vectorclass (define VECTORCLASS_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadVectorClass.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass-download")
  SET(VECTORCLASS_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/vectorclass" CACHE STRING "VectorClass source directory")
 ENDIF()

 IF(NOT DEFINED JPEG_SOURCE_DIR)
  SET(JPEG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo" CACHE STRING "JPEG source directory")
  CONFIGURE_FILE(cmake/DownloadJPEG.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo-download")
  SET(JPEG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo" CACHE STRING "JPEG source directory")
 ENDIF()

 IF(NOT DEFINED MAGIC_ENUM_SOURCE_DIR)
  MESSAGE(STATUS "Downloading MagicEnum to ${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum (define MAGIC_ENUM_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadMagicEnum.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum-download")
  SET(MAGIC_ENUM_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum" CACHE STRING "MagicEnum source directory")
 ENDIF()

 IF(NOT DEFINED ONEDNN_SOURCE_DIR)
  MESSAGE(STATUS "Downloading oneDNN to ${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN (define ONEDNN_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadOneDNN.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN-download")
  SET(ONEDNN_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN" CACHE STRING "oneDNN source directory")
 ENDIF()

 IF(NOT DEFINED CIMG_SOURCE_DIR)
  MESSAGE(STATUS "Downloading CImg to ${DNN_DEPENDENCIES_SOURCE_DIR}/CImg (define CIMG_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadCImg.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/cimg-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/cimg-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/cimg-download")
  SET(CIMG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/CImg" CACHE STRING "CImg source directory")
 ENDIF()


add_library(${PROJECT_NAME} ${libdnn_sources})
DNN_TARGET_ENABLE_CXX17(${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PRIVATE DNN_EXPORTS DNN_CACHE_PRIMITIVES DNN_OMP cimg_use_cpp11 cimg_use_jpeg cimg_use_png cimg_use_zlib)
target_include_directories(${PROJECT_NAME}
    PUBLIC
       $<INSTALL_INTERFACE:include>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test ${libdnn_test})
DNN_TARGET_ENABLE_CXX17(test)
target_include_directories(test 
    PUBLIC
       $<INSTALL_INTERFACE:include>
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/zlib-1.2.11)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/zlib-1.2.11)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/libpng)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/libpng)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN/include)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN/include)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/oneDNN/src)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN/src)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/vectorclass)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum/include)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/CImg)

# ---[ Configure ZLIB
IF(NOT TARGET zlib)
  ADD_SUBDIRECTORY(
    "${ZLIB_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-1.2.11")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE zlib)

if(WIN32 OR MSVC)
  set(SKIP_INSTALL_ALL ON)
endif()
# ---[ Configure PNG
IF(NOT TARGET png)
  ADD_SUBDIRECTORY(
    "${PNG_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/libpng")
ENDIF()
if(WIN32 OR MSVC)
  TARGET_LINK_LIBRARIES(png PRIVATE zlib)
endif()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE png)

# ---[ Configure JPEG
IF(NOT TARGET jpeg)
  ADD_SUBDIRECTORY(
    "${JPEG_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE jpeg)

# ---[ Configure oneDNN
IF(NOT TARGET dnnl)
  ADD_SUBDIRECTORY(
    "${ONEDNN_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE dnnl)

# ---[ Configure VectorClass
IF(NOT TARGET vectorclass)
  ADD_SUBDIRECTORY(
    "${VECTORCLASS_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE vectorclass)

# ---[ Configure MagicEnum
IF(NOT TARGET magic_enum)
  ADD_SUBDIRECTORY(
    "${MAGIC_ENUM_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum")
ENDIF()

TARGET_LINK_LIBRARIES(test PRIVATE ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})