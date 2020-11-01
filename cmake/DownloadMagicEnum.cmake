CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0 FATAL_ERROR)

PROJECT(magic_enum-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(magic_enum
	GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
	GIT_TAG master
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/magic_enum"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/magic_enum"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)