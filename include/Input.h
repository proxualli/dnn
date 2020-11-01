#pragma once
#include "Layer.h"

namespace dnn
{
	class Input final : public Layer
	{
	public:
		Input(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const size_t c, const size_t d, const size_t h, const size_t w);
		
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) override {}
		void BackwardProp(const size_t batchSize) override {}
	};
}
