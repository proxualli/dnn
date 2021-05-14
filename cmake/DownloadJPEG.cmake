CMAKE_MINIMUM_REQUIRED(VERSION 3.7.0 FATAL_ERROR)

PROJECT(jpeg-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(jpeg
	GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
	GIT_TAG main
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/libjpeg-turbo"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/libjpeg-turbo"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
