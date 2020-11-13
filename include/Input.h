#pragma once
#include "Layer.h"

namespace dnn
{
	class Input final : public Layer
	{
	public:
		Input(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const size_t c, const size_t d, const size_t h, const size_t w) :
			Layer(device, format, name, LayerTypes::Input, 0, 0, c, d, h, w, 0, 0, 0, std::vector<Layer*>())
		{
		}
		
		std::string GetDescription() const final override 
		{
			return GetDescriptionHeader();
		}

		size_t FanIn() const final override
		{
			return 1;
		}

		size_t FanOut() const final override
		{
			return CDHW;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			chosenFormat = PlainFmt;
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C), int(H), int(W) }), dnnl::memory::data_type::f32, chosenFormat));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C), int(H), int(W) }), dnnl::memory::data_type::f32, chosenFormat));
		}

		void ForwardProp(const size_t batchSize, const bool training) override { }
		void BackwardProp(const size_t batchSize) override { }
	};
}
