#include "Model.h"

namespace dnn
{
	Input::Input(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const size_t c, const size_t d, const size_t h, const size_t w) :
		Layer(device, format, name, LayerTypes::Input, 0, 0, c, d, h, w, 0, 0, 0, std::vector<Layer*>())
	{
	}
	
	std::string Input::GetDescription() const
	{
		return GetDescriptionHeader();
	}

	size_t Input::FanIn() const
	{
		return 1;
	}

	size_t Input::FanOut() const
	{
		return CDHW;
	}

	void Input::InitializeDescriptors(const size_t batchSize)
	{
		DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C), int(H), int(W) }), dnnl::memory::data_type::f32, PlainFmt));
		DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C), int(H), int(W) }), dnnl::memory::data_type::f32, PlainFmt));
	}
}