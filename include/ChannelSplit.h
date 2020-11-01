#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelSplit final : public Layer
	{
	public:
		ChannelSplit(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const size_t group, const size_t groups);

		const size_t Group;
		const size_t Groups;

		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;
	};
}
