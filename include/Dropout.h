#pragma once
#include "Layer.h"

namespace dnn
{
	class Dropout final : public Layer
	{
	public:
		Dropout(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.3));
		
		const size_t PartialCDHW;
		const Float Keep;
		const Float Scale;
		
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;
		
		void InitializeDescriptors(const size_t batchSize) final override;

		void SetBatchSize(const size_t batchSize) override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::bernoulli_distribution DropoutDistribution;
		FloatVector NeuronsActive;
	};
}