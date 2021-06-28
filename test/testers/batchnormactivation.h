#pragma once

#include <cstddef>
#include <cstdlib>

#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#include <Model.h>


class BatchNormActivationTester
{
public:
	BatchNormActivationTester() :
		iterations_(1),
		errorLimit_(1.0e-5),
		batchSize_(1),
		channels_(1),
		height_(1),
		width_(1),
		activation_(dnn::Activations::HardSwish),
		layerType_(dnn::LayerTypes::BatchNormHardSwish),
		device_(dnn::Device(dnnl::engine(dnnl::engine::kind::cpu, 0), dnnl::stream(dnnl::engine(dnnl::engine::kind::cpu, 0), dnnl::stream::flags::default_flags)))
	{
	}

	BatchNormActivationTester(const BatchNormActivationTester&) = delete;

	inline BatchNormActivationTester(BatchNormActivationTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		batchSize_(tester.batchSize_),
		channels_(tester.channels_),
		width_(tester.width_),
		height_(tester.height_),
		activation_(tester.activation_),
		layerType_(tester.layerType_),
		device_(tester.device_)
	{
	}

	BatchNormActivationTester& operator=(const BatchNormActivationTester&) = delete;

	~BatchNormActivationTester()
	{
	}

	inline BatchNormActivationTester& iterations(size_t iterations)
	{
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const
	{
		return this->iterations_;
	}

	inline BatchNormActivationTester& errorLimit(float errorLimit)
	{
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const
	{
		return this->errorLimit_;
	}

	inline BatchNormActivationTester& batchSize(size_t batchSize)
	{
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const
	{
		return this->batchSize_;
	}

	inline BatchNormActivationTester& channels(size_t channels)
	{
		this->channels_ = channels;
		return *this;
	}

	inline size_t channels() const
	{
		return this->channels_;
	}

	inline BatchNormActivationTester& height(size_t height)
	{
		this->height_ = height;
		return *this;
	}

	inline size_t height() const
	{
		return this->height_;
	}

	inline BatchNormActivationTester& width(size_t width)
	{
		this->width_ = width;
		return *this;
	}

	inline size_t width() const
	{
		return this->width_;
	}

	inline BatchNormActivationTester& layerType(dnn::LayerTypes layerType)
	{
		this->layerType_ = layerType;
		return *this;
	}

	inline dnn::LayerTypes layerType() const
	{
		return this->layerType_;
	}

	inline BatchNormActivationTester& activation(dnn::Activations activation)
	{
		this->activation_ = activation;
		return *this;
	}

	inline dnn::Activations activation() const
	{
		return this->activation_;
	}

	/*inline BatchNormActivationTester& device(dnn::Device device)
	{
		this->device_ = device;
		return *this;
	}*/

	inline dnn::Device device() const
	{
		return this->device_;
	}
		
	void testForward(bool inference) const
	{
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-0.1, 1.0), std::mt19937(seed));

		std::vector<float> input(batchSize() * channels() * height() * width());
		std::vector<float> output(batchSize() * channels() * height() * width());
		std::vector<float> referenceOutput(batchSize() * channels() * height() * width());
				
		size_t scratchSize = 0;
		enum nnp_status status = nnp_convolution_output(
			algorithm,
			batchSize(), inputChannels(), outputChannels(),
			inputSize(), inputPadding(), kernelSize(),
			nullptr, nullptr, nullptr, nullptr, nullptr, &scratchSize,
			activation, nullptr,
			this->threadpool, nullptr);
		ASSERT_EQ(nnp_status_success, status);

		std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> scratchBuffer(scratchSize);
		std::vector<float> maxErrors;
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::generate(bias.begin(), bias.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));
			std::fill(scratchBuffer.begin(), scratchBuffer.end(), 0xA5);

			nnp_convolution_output__reference(
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), kernel.data(), bias.data(), referenceOutput.data(),
				this->threadpool);

			switch (activation) {
				case nnp_activation_identity:
					break;
				case nnp_activation_relu:
					nnp_relu_output__reference(
						batchSize(), outputChannels() * outputHeight() * outputWidth(),
						referenceOutput.data(), referenceOutput.data(), 0.0,
						this->threadpool);
					break;
				default:
					FAIL() << "Unexpected activation value: " << activation;
			}

			enum nnp_status status = nnp_convolution_output(
				algorithm,
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(),
				input.data(), kernel.data(), bias.data(), output.data(),
				scratchSize == 0 ? nullptr : scratchBuffer.data(),
				scratchSize == 0 ? nullptr : &scratchSize,
				activation, nullptr,
				this->threadpool, nullptr);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		EXPECT_LT(median(maxErrors), errorLimit());
	}

	void testBackward() const {
		ASSERT_EQ(1, batchSize());

		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		auto rng = std::bind(std::uniform_real_distribution<float>(-0.1, 1.0), std::mt19937(seed));

		std::vector<float> input(inputChannels() * inputHeight() * inputWidth());
		std::vector<float> kernel(outputChannels() * inputChannels() * kernelHeight() * kernelWidth());

		std::vector<float> bias(outputChannels());

		std::vector<float> output(outputChannels() * outputHeight() * outputWidth());
		std::vector<float> referenceOutput(outputChannels() * outputHeight() * outputWidth());

		size_t scratchSize = 0;
		enum nnp_status status = nnp_convolution_inference(
			algorithm,
			precompute ? nnp_convolution_transform_strategy_reuse : nnp_convolution_transform_strategy_compute,
			inputChannels(), outputChannels(),
			inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
			nullptr, nullptr, nullptr, nullptr, nullptr, &scratchSize,
			activation, nullptr,
			this->threadpool, nullptr);
		ASSERT_EQ(nnp_status_success, status);

		std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> scratchBuffer(scratchSize);

		std::vector<float> maxErrors;
		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			std::generate(bias.begin(), bias.end(), std::ref(rng));
			std::fill(output.begin(), output.end(), nanf(""));
			std::fill(scratchBuffer.begin(), scratchBuffer.end(), 0xA5);

			nnp_convolution_output__reference(
				1, inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), kernel.data(), bias.data(), referenceOutput.data(),
				this->threadpool);

			switch (activation) {
				case nnp_activation_identity:
					break;
				case nnp_activation_relu:
					nnp_relu_output__reference(
						batchSize(), outputChannels() * outputHeight() * outputWidth(),
						referenceOutput.data(), referenceOutput.data(), 0.0,
						this->threadpool);
					break;
				default:
					FAIL() << "Unexpected activation value: " << activation;
			}

			std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> transformedKernel;

			if (precompute) {
				size_t transformedKernelSize = 0;
				enum nnp_status status = nnp_convolution_inference(
					algorithm, nnp_convolution_transform_strategy_precompute,
					inputChannels(), outputChannels(),
					inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
					nullptr, nullptr, nullptr, nullptr, nullptr, &transformedKernelSize,
					activation, nullptr,
					threadpool, nullptr);
				ASSERT_EQ(nnp_status_success, status);

				transformedKernel.resize(transformedKernelSize);

				status = nnp_convolution_inference(
					algorithm, nnp_convolution_transform_strategy_precompute,
					inputChannels(), outputChannels(),
					inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
					nullptr, kernel.data(), nullptr, nullptr, transformedKernel.data(), &transformedKernelSize,
					activation, nullptr,
					threadpool, nullptr);
				ASSERT_EQ(nnp_status_success, status);
			}

			const void* kernelData = kernel.data();
			if (precompute) {
				kernelData = transformedKernel.data();
			}
			enum nnp_status status = nnp_convolution_inference(
				algorithm,
				precompute ? nnp_convolution_transform_strategy_reuse : nnp_convolution_transform_strategy_compute,
				inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), static_cast<const float*>(kernelData), bias.data(), output.data(),
				scratchSize == 0 ? nullptr : scratchBuffer.data(),
				scratchSize == 0 ? nullptr : &scratchSize,
				activation, nullptr,
				this->threadpool, nullptr);
			ASSERT_EQ(nnp_status_success, status);

			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		EXPECT_LT(median(maxErrors), errorLimit());
	}

private:
	inline static float relativeError(float reference, float actual)
	{
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

	inline static float median(std::vector<float>& array)
	{
		std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
		return array[array.size() / 2];
	}

	size_t iterations_;
	float errorLimit_;
	size_t batchSize_;
	size_t channels_;
	size_t width_;
    size_t height_;
	dnn::Activations activation_;
	dnn::LayerTypes layerType_;
	dnn::Device device_;
};