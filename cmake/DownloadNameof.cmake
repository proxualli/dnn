CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0 FATAL_ERROR)

PROJECT(nameof-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(nameof
	GIT_REPOSITORY https://github.com/Neargye/nameof.git
	GIT_TAG master
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/nameof"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/nameof"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)