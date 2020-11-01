#pragma once
#include "Layer.h"

namespace dnn
{
	class Convolution final : public Layer
	{
	public:
		Convolution(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const size_t c, const size_t kernelH, const size_t kernelW, const size_t strideH = 1, const size_t strideW = 1, const size_t dilationH = 1, const size_t dilationW = 1, const size_t padH = 0, const size_t padW = 0, const size_t groups = 1, const bool hasBias = true);
		
		const size_t Groups;
		const size_t KernelH;
		const size_t KernelW;
		const size_t StrideH;
		const size_t StrideW;
		const size_t DilationH;
		const size_t DilationW;
		const size_t DilationKernelH;
		const size_t DilationKernelW;
						
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

		ByteVector GetImage(const Byte fillColor) final override;
		
	private:
		const dnnl::memory::dims Strides;
		const dnnl::memory::dims Dilates;
		const dnnl::memory::dims Padding;
	
		std::unique_ptr<dnnl::convolution_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::convolution_backward_weights::primitive_desc> bwdWeightsDesc;
		std::unique_ptr<dnnl::convolution_backward_data::primitive_desc> bwdDataDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;

#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::convolution_forward> fwd;
		std::unique_ptr<dnnl::convolution_backward_weights> bwdWeights;
		std::unique_ptr<dnnl::convolution_backward_data> bwdData;
#endif

		std::unique_ptr<dnnl::binary> bwdAdd;

		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
		bool reorderBwdWeights;
		bool reorderBwdDiffWeights;
	};
}