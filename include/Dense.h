#pragma once
#include "Layer.h"

namespace dnn
{
	class Dense final : public Layer
	{
	public:
		Dense(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const size_t c, const std::vector<Layer*>& inputs, const bool hasBias = true);
		
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

		ByteVector GetImage(const Byte fillColor) final override;

	private:
		std::unique_ptr<dnnl::inner_product_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::inner_product_backward_weights::primitive_desc> bwdWeightsDesc;
		std::unique_ptr<dnnl::inner_product_backward_data::primitive_desc> bwdDataDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;

		std::unique_ptr<dnnl::inner_product_forward> fwd;
		std::unique_ptr<dnnl::inner_product_backward_weights> bwdWeights;
		std::unique_ptr<dnnl::inner_product_backward_data> bwdData;
		std::unique_ptr<dnnl::binary> bwdAdd;

		bool reorderFwdSrc;
	    bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
		bool reorderBwdWeights;
		bool reorderBwdDiffWeights;
	};
}
