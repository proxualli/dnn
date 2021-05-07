#include <gtest/gtest.h>

#include <include/Utils.h>

#include <testers/batchnormactivation.h>


TEST() {
	BatchnormActivationTester()
		.inputSize(8, 8)
		.iterations(100)
		.errorLimit(1.0e-5);
}

int main(int argc, char* argv[]) {
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}