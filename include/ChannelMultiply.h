#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelMultiply final : public Layer
	{
	public:
		ChannelMultiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs);

		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::binary> fwd;
		std::unordered_map<int, dnnl::memory> fwdArgs;
	};
}
