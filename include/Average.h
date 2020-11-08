#pragma once
#include "Layer.h"

namespace dnn
{
	class Average final : public Layer
	{
	private:
		std::unique_ptr<dnnl::sum::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::sum> fwd;
		std::vector<Float> Scales;
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;

	public:
		const Float Scale;

		Average(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
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

		std::string GetDescription() const final override
		{
			std::string description = GetDescriptionHeader();

			description.append(nwl + std::string(" Scale:") + dtab + FloatToString(Scale));

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
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				Format = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, Format));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, Format));
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

		void ForwardProp(const size_t batchSize, const bool training) final override
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

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto size = IsPlainFormat() ? CDHW : PaddedCDHW;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto i = 0ull; i < Inputs.size(); i++)
					for (auto n = 0ull; n < size; n++)
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
						const auto start = b * size;
						const auto end = start + size;
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
						const auto start = b * size;
						const auto end = start + size;
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
						const auto start = b * size;
						const auto end = start + size;
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
	};
}
