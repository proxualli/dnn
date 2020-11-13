#pragma once
#include "Layer.h"

namespace dnn
{
	class Max final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::binary> fwd;

	public:
		Max(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Max, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
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

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			return description;
		}

		size_t FanIn() const final override
		{
			return 1;
		}

		size_t FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);
			chosenFormat = GetDataFmt(*DstMemDesc);

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in Max layer");
			}

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_max, *Inputs[0]->DstMemDesc, *Inputs[1]->DstMemDesc, *InputLayer->DstMemDesc), Device.engine));
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
		}

		void ForwardProp(const size_t batchSize, const bool training)  final override
		{
			fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[0]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[1]->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }});
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
			DNN_UNREF_PAR(training);
#endif
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (size_t n = 0; n < CDHW; n++)
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
	};
}
