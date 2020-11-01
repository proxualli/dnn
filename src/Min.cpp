#include "Model.h"

namespace dnn
{
	Min::Min(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
		Layer(device, format, name, LayerTypes::Min, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
	{
		assert(Inputs.size() == 2);

		for (size_t i = 0; i < Inputs.size(); i++)
		{
			assert(Inputs[i]->C == C);
			assert(Inputs[i]->D == D);
			assert(Inputs[i]->H == H);
			assert(Inputs[i]->W == W);
		}
	}

	std::string Min::GetDescription() const
	{
		return GetDescriptionHeader();
	}

	size_t Min::FanIn() const
	{
		return 1;
	}

	size_t Min::FanOut() const
	{
		return 1;
	}

	void Min::InitializeDescriptors(const size_t batchSize)
	{
		DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
		DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);

		for (size_t i = 1; i < Inputs.size(); i++)
		{
			assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
			if (*DstMemDesc != *Inputs[i]->DstMemDesc)
				throw std::invalid_argument("Incompatible memory formats in Min layer");
		}

		fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_min, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, *InputLayer->DstMemDesc), Device.first));
		fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
	}

	void Min::ForwardProp(const size_t batchSize, const bool training)
	{
		fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.first, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.first, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.first, Neurons.data()) }});
		Device.second.wait();

#ifndef DNN_LEAN
		if (training)
			ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
		DNN_UNREF_PAR(batchSize);
		DNN_UNREF_PAR(training);
#endif
	}

	void Min::BackwardProp(const size_t batchSize)
	{
#ifdef DNN_LEAN
		ZeroGradientMulti(batchSize);
#endif

#ifdef DNN_STOCHASTIC
		if (batchSize == 1)
		{
			for (auto n = 0ull; n < CDHW; n++)
			{
				Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
				Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
			}
		}
		else
		{
#endif
			for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
			{
				const auto start = b * PaddedCDHW;
				const auto end = start + CDHW;
				for (auto n = start; n < end; n++)
				{
					Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
					Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
				}
			});
#ifdef DNN_STOCHASTIC
		}
#endif
#ifdef DNN_LEAN
		ReleaseGradient();
#endif // DNN_LEAN
	}
}