#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelShuffle final : public Layer
	{
	public:
		ChannelShuffle(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const size_t groups);

		const size_t Groups;
		const size_t GroupSize;

		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::unique_ptr<dnnl::shuffle_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::shuffle_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::shuffle_forward> fwd;
		std::unique_ptr<dnnl::shuffle_backward> bwd;
	};
}