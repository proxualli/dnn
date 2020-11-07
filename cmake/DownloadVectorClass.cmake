CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0 FATAL_ERROR)

PROJECT(vectorclass-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(vectorclass
	GIT_REPOSITORY https://github.com/zamir1001/vectorclass.git
	GIT_TAG master
	SOURCE_DIR "${DNN_DEPENDENCIES_SOURCE_DIR}/vectorclass"
	BINARY_DIR "${DNN_DEPENDENCIES_BINARY_DIR}/vectorclass"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)