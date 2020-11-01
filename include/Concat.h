#pragma once
#include "Layer.h"

namespace dnn
{
	class Concat final : public Layer
	{
	public:
		Concat(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs);

		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::unique_ptr<dnnl::concat::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::concat> fwd;
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unordered_map<int, dnnl::memory> bwdArgs;
	};
}
