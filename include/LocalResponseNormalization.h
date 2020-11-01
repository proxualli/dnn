#pragma once
#include "Layer.h"

namespace dnn
{
	class LocalResponseNormalization final : public Layer
	{
	public:
		LocalResponseNormalization(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool acrossChannels = false, const size_t localSize = 5, const Float alpha = Float(1), const Float beta = Float(5), const Float k = Float(1));
		
		const bool AcrossChannels;
		const size_t LocalSize;
		const Float Alpha;
		const Float Beta;
		const Float K;

		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::unique_ptr<dnnl::lrn_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::lrn_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;

		std::unique_ptr<dnnl::lrn_forward> fwd;
		std::unique_ptr<dnnl::lrn_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;

		std::unique_ptr<dnnl::memory> WorkspaceMemory;

		dnnl::algorithm algorithm;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
	};
}