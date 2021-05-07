CMAKE_MINIMUM_REQUIRED(VERSION 3.7.0 FATAL_ERROR)

PROJECT(googletest-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG master
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/googletest"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/googletest"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
