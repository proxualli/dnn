#pragma once
#include "Layer.h"

namespace dnn
{
	class Input final : public Layer
	{
	public:
		Input(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const UInt c, const UInt d, const UInt h, const UInt w) :
			Layer(device, format, name, LayerTypes::Input, 0, 0, c, d, h, w, 0, 0, 0, std::vector<Layer*>())
		{
		}
		
		std::string GetDescription() const final override 
		{
			return GetDescriptionHeader();
		}

		UInt FanIn() const final override
		{
			return 1;
		}

		UInt FanOut() const final override
		{
			return CDHW();
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			ChosenFormat = PlainFmt;
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C), int(H), int(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C), int(H), int(W) }), dnnl::memory::data_type::f32, ChosenFormat));
		}

		void ForwardProp(const UInt batchSize, const bool training) override { DNN_UNREF_PAR(batchSize); DNN_UNREF_PAR(training); }
		void BackwardProp(const UInt batchSize) override { DNN_UNREF_PAR(batchSize); }
	};
}