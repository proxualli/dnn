#include "Model.h"

namespace dnn
{
	Add::Add(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
		Layer(device, format, name, LayerTypes::Add, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
	{
		assert(Inputs.size() > 1);

		for (auto i = 0ull; i < Inputs.size(); i++)
		{
			assert(Inputs[i]->C == C);
			assert(Inputs[i]->D == D);
			assert(Inputs[i]->H == H);
			assert(Inputs[i]->W == W);
		}

		Scales = std::vector<Float>(Inputs.size(), Float(1));
	}

	std::string Add::GetDescription() const
	{
		return GetDescriptionHeader();
	}

	size_t Add::FanIn() const
	{
		return 1;
	}

	size_t Add::FanOut() const
	{
		return 1;
	}

	void Add::InitializeDescriptors(const size_t batchSize)
	{
		if (InputLayer->DstMemDesc->data.ndims == 2)
		{
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
		}
		else
		{
			if (Format == dnnl::memory::format_tag::any)
			{
				Format = GetDataFmt(*InputLayer->DstMemDesc);
				if (Format != GetDataFmt(*InputLayer->DiffDstMemDesc))
					throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
			}

			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
		}

		for (size_t i = 1; i < Inputs.size(); i++)
		{
			assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
			if (*DstMemDesc != *Inputs[i]->DstMemDesc)
				throw std::invalid_argument("Incompatible memory formats in Add layer");
		}

		srcsMemsDesc = std::vector<dnnl::memory::desc>(Inputs.size());
		
		for (size_t i = 0; i < Inputs.size(); i++)
			srcsMemsDesc[i] = *Inputs[i]->DstMemDesc;
				
		fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, Scales, srcsMemsDesc, Device.first));
		
		fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.first, Neurons.data()) } };
		for (size_t i = 0; i < Inputs.size(); i++)
			fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.first, Inputs[i]->Neurons.data()) });
		
		fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
	}

	void Add::ForwardProp(const size_t batchSize, const bool training)
	{
		fwd->execute(Device.second, fwdArgs);
		Device.second.wait();

#ifndef DNN_LEAN
		if (training)
			ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
		DNN_UNREF_PAR(batchSize);
		DNN_UNREF_PAR(training);
#endif
	}

	void Add::BackwardProp(const size_t batchSize)
	{	
		
#ifdef DNN_LEAN
		ZeroGradientMulti(batchSize);
#endif

#ifdef DNN_STOCHASTIC
		if (batchSize == 1)
		{
			for (auto i = 0ull; i < Inputs.size(); i++)
				for (auto n = 0ull; n < Inputs[i]->CDHW; n++)
					Inputs[i]->NeuronsD1[n] += NeuronsD1[n];
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
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
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
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
					}
				});
			}
			break;

			case 4:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
					}
				});
			}
			break;

			case 5:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
					}
				});
			}
			break;

			case 6:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
					}
				});
			}
			break;

			case 7:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[6]->NeuronsD1[n] += NeuronsD1[n];
					}
				});
			}
			break;

			case 8:
			{
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[6]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[7]->NeuronsD1[n] += NeuronsD1[n];
					}
				});
			}
			break;

			default:
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto end = start + CDHW;
					for (auto n = start; n < end; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[3]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[4]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[5]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[6]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[7]->NeuronsD1[n] += NeuronsD1[n];
					}
					for (auto i = 8ull; i < Inputs.size(); i++)
						for (auto n = start; n < end; n++)
							Inputs[i]->NeuronsD1[n] += NeuronsD1[n];
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