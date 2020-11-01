#pragma once
#include "Layer.h"

namespace dnn
{
	enum class Algorithms
	{
		Linear = 0,
		Nearest = 1
	};

	class Resampling final : public Layer
	{
	public:
		Resampling(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Algorithms algorithm, const Float factorH, const Float factorW);

		const Algorithms Algorithm;
		const Float FactorH;
		const Float FactorW;
		
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::unique_ptr<dnnl::resampling_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::resampling_backward::primitive_desc> bwdDesc;
	    std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;

		std::unique_ptr<dnnl::resampling_forward> fwd;
		std::unique_ptr<dnnl::resampling_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
	};
}