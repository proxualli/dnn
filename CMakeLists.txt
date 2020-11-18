cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

message(STATUS "Using CMake version ${CMAKE_VERSION}")

INCLUDE(GNUInstallDirs)
INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCXXSourceCompiles)

set(DNN_VERSION_MAJOR 1)
set(DNN_VERSION_MINOR 0)
set(DNN_VERSION_PATCH 0)
set(DNN_VERSION_NOTE "")

set(DNN_VERSION ${DNN_VERSION_MAJOR}.${DNN_VERSION_MINOR}.${DNN_VERSION_PATCH})
math(EXPR DNN_VERSION_NUMBER "10000*${DNN_VERSION_MAJOR} + 100*${DNN_VERSION_MINOR} + ${DNN_VERSION_PATCH}")

#####
# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

#####
# Enables link_directories() treat paths relative
# to the source dir.
if(POLICY CMP0015)
    cmake_policy(SET CMP0015 NEW)
endif(POLICY CMP0015)

project(dnn VERSION ${DNN_VERSION} LANGUAGES CXX)

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(BUILD_STATIC_LIBS "Build static library" ON)

####
# Create the library target
set(PACKAGE_NAME dnn)

add_compile_definitions(DNN_EXPORTS DNN_CACHE_PRIMITIVES DNN_OMP cimg_use_cpp11 cimg_use_jpeg cimg_use_png cimg_use_zlib)

MACRO(DNN_TARGET_ENABLE_CXX17 target)
  IF(WIN32 OR MSVC)
    TARGET_COMPILE_OPTIONS(${target} PRIVATE /arch:AVX2 /openmp:experimental)
  ELSE()
    TARGET_COMPILE_OPTIONS(${target} PRIVATE -mavx2 -mfma -march=znver2 -fopenmp)
  ENDIF()
  SET_TARGET_PROPERTIES(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF)
ENDMACRO()

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
  SET(ZLIB_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/zlib-1.2.11" CACHE STRING "ZLib source directory")
  SET(ZLIB_INCLUDE_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-1.2.11" CACHE STRING "ZLib include directory")
  SET(ZLIB_LIBRARY "${DNN_DEPENDENCIES_SOURCE_DIR}/zlib-1.2.11" CACHE STRING "ZLib bin directory")
 ENDIF()

 IF(NOT DEFINED PNG_SOURCE_DIR)
  MESSAGE(STATUS "Downloading PNG to ${DNN_DEPENDENCIES_SOURCE_DIR}/libpng (define PNG_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPNG.cmake "${DNN_DEPENDENCIES_BINARY_DIR}/png-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/png-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DNN_DEPENDENCIES_BINARY_DIR}/png-download")
  SET(PNG_SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libpng" CACHE STRING "PNG source directory")
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

set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_TURBOJPEG FALSE CACHE BOOL "" FORCE) 
set(PNG_EXECUTABLES FALSE CACHE BOOL "" FORCE)
set(PNG_TESTS FALSE CACHE BOOL "" FORCE)
set(PNG_BUILD_ZLIB TRUE CACHE BOOL "" FORCE)

if(BUILD_STATIC_LIBS)
  set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
  set(ENABLE_SHARED FALSE CACHE BOOL "" FORCE)
  set(ENABLE_STATIC TRUE CACHE BOOL "" FORCE)
  set(PNG_SHARED FALSE CACHE BOOL "" FORCE)
  set(PNG_STATIC TRUE CACHE BOOL "" FORCE)
  add_library(${PROJECT_NAME} ${libdnn_sources})
  DNN_TARGET_ENABLE_CXX17(${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME}
       PUBLIC
          ${PROJECT_NAME} PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
          $<INSTALL_INTERFACE:include>)
endif()

if(BUILD_SHARED_LIBS)
  set(DNNL_LIBRARY_TYPE "SHARED" CACHE STRING "" FORCE)
  set(ENABLE_SHARED TRUE CACHE BOOL "" FORCE)
  set(ENABLE_STATIC FALSE CACHE BOOL "" FORCE)
  set(PNG_SHARED TRUE CACHE BOOL "" FORCE)
  set(PNG_STATIC FALSE CACHE BOOL "" FORCE)
  add_library(${PROJECT_NAME} ${libdnn_sources})
  DNN_TARGET_ENABLE_CXX17(${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME}
       PRIVATE
          ${PROJECT_NAME} PRIVATE
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
          $<INSTALL_INTERFACE:include>)
endif()

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
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum/include)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum/include)
include_directories(${DNN_DEPENDENCIES_SOURCE_DIR}/CImg)
include_directories(${DNN_DEPENDENCIES_BINARY_DIR}/CImg)

# ---[ Configure ZLIB
IF(NOT TARGET zlib)
  ADD_SUBDIRECTORY(
    "${ZLIB_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/zlib-1.2.11")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC zlib)

# ---[ Configure PNG
IF(NOT TARGET png)
  ADD_SUBDIRECTORY(
    "${PNG_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/libpng")
ENDIF()
include_directories(libpng ${ZLIB_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE	png)

# ---[ Configure JPEG
IF(NOT TARGET jpeg)
  ADD_SUBDIRECTORY(
    "${JPEG_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE	jpeg)

# ---[ Configure oneDNN
IF(NOT TARGET dnnl)
  ADD_SUBDIRECTORY(
    "${ONEDNN_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/oneDNN")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE	dnnl)

# ---[ Configure VectorClass
IF(NOT TARGET vectorclass)
  ADD_SUBDIRECTORY(
    "${VECTORCLASS_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass")
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE	vectorclass)

# ---[ Configure MagicEnum
IF(NOT TARGET magic_enum)
  ADD_SUBDIRECTORY(
    "${MAGIC_ENUM_SOURCE_DIR}"
    "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum")
ENDIF()