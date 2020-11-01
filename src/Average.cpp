#include "Model.h"

namespace dnn
{
	Average::Average(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
		Layer(device, format, name, LayerTypes::Average, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
		Scale(Float(1) / Inputs.size())
	{
		assert(Inputs.size() > 1);

		for (auto i = 0ull; i < Inputs.size(); i++)
		{
			assert(Inputs[i]->C == C);
			assert(Inputs[i]->D == D);
			assert(Inputs[i]->H == H);
			assert(Inputs[i]->W == W);
		}

		Scales = std::vector<Float>(Inputs.size(), Scale);
	}

	std::string Average::GetDescription() const
	{
		std::string description = GetDescriptionHeader();

		description.append(nwl + " Scale:" + dtab + FloatToString(Scale));

		return description;
	}

	size_t Average::FanIn() const
	{
		return 1;
	}

	size_t Average::FanOut() const
	{
		return 1;
	}

	void Average::InitializeDescriptors(const size_t batchSize)
	{
		DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
		DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);

		for (auto i = 1ull; i < Inputs.size(); i++)
		{
			assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
			if (*DstMemDesc != *Inputs[i]->DstMemDesc)
				throw std::invalid_argument("Incompatible memory formats in Average layer");
		}

		srcsMemsDesc = std::vector<dnnl::memory::desc>(Inputs.size());
		for (auto i = 0ull; i < Inputs.size(); i++)
			srcsMemsDesc[i] = *Inputs[i]->DstMemDesc;

		fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, Scales, srcsMemsDesc, Device.first));

		fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.first, Neurons.data()) } };
		for (auto i = 0ull; i < Inputs.size(); i++)
			fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.first, Inputs[i]->Neurons.data()) });
	
		fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
	}

	void Average::ForwardProp(const size_t batchSize, const bool training)
	{
		fwd->execute(Device.second, fwdArgs);
		Device.second.wait();

#ifndef DNN_LEAN
		if (training)
			ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
		DNN_UNREF_PAR(batchSize);
#endif
	}

	void Average::BackwardProp(const size_t batchSize)
	{
#ifdef DNN_LEAN
		ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

#ifdef DNN_STOCHASTIC
		if (batchSize == 1)
		{
			for (auto i = 0ull; i < Inputs.size(); i++)
				for (auto n = 0ull; n < Inputs[i]->CDHW; n++)
					Inputs[i]->NeuronsD1[n] += Scale * NeuronsD1[n];
		}
		else
		{
#endif
			switch (Inputs.size())
			{
			case 2:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += Scale * NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += Scale * NeuronsD1[n];
					}
				});
			}
			break;

			case 3:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += Scale * NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += Scale * NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += Scale * NeuronsD1[n];
					}
				});
			}
			break;

			default:
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto i = 0ull; i < Inputs.size(); i++)
						for (auto n = start; n < end; n++)
							Inputs[i]->NeuronsD1[n] += Scale * NeuronsD1[n];
				});
			}
#ifdef DNN_STOCHASTIC
	}
#endif

#ifdef DNN_LEAN
		ReleaseGradient();
#endif // DNN_LEAN
	}
}