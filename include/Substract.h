#pragma once
#include "Layer.h"

namespace dnn
{
	class Substract final : public Layer
	{
	private:
		std::unique_ptr<dnnl::sum::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::sum> fwd;
		std::vector<Float> Scales;
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;

	public:
		Substract(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Substract, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);

			for (size_t i = 0; i < Inputs.size(); i++)
			{
				assert(Inputs[i]->C == C);
				assert(Inputs[i]->D == D);
				assert(Inputs[i]->H == H);
				assert(Inputs[i]->W == W);
			}

			Scales = std::vector<Float>(Inputs.size(), Float(-1));
			Scales[0] = Float(1);
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
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			DNN_UNREF_PAR(batchSize);

			DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);
			chosenFormat = GetDataFmt(*DstMemDesc);

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in Substract layer");
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>(Inputs.size());
			for (auto i = 0ull; i < Inputs.size(); i++)
				srcsMemsDesc[i] = *Inputs[i]->DstMemDesc;

			fwdDesc = std::make_unique<dnnl::sum::primitive_desc>(dnnl::sum::primitive_desc(*DstMemDesc, Scales, srcsMemsDesc, Device.engine));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

			fwd = std::make_unique<dnnl::sum>(dnnl::sum(*fwdDesc));
		}

		void ForwardProp(const size_t batchSize, const bool training)  final override
		{
			fwd->execute(Device.stream, fwdArgs);
			Device.stream.wait();

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
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (inputs == 2)
				{
#pragma omp simd
					for (auto n = 0ull; n < size; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						Inputs[1]->NeuronsD1[n] -= NeuronsD1[n];
					}
				}
				else
				{
#pragma omp simd
					for (auto n = 0ull; n < size; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
						for (auto i = 1ull; i < inputs; i++)
							Inputs[i]->NeuronsD1[n] -= NeuronsD1[n];
					}
				}
			}
			else
			{
#endif
				if (inputs == 2)
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;

						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							Inputs[1]->NeuronsD1[n] -= NeuronsD1[n];
						}
					});
				}
				else
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * size;
						const auto end = start + size;

						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n];
							for (auto i = 1ull; i < inputs; i++)
								Inputs[i]->NeuronsD1[n] -= NeuronsD1[n];
						}
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