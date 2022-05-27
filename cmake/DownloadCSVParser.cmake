CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0 FATAL_ERROR)

PROJECT(csv-parser-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(csv
	GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser
	GIT_TAG master
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/csv-parser"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/csv-parser"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
