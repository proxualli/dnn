#pragma once
#include "Layer.h"

namespace dnn
{
	class Divide final : public Layer
	{
	public:
		Divide(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<std::shared_ptr<Layer>>& inputs) :
			Layer(device, format, name, LayerTypes::Divide, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1 && Inputs.size() < 4);

			if (Inputs.size() > 3)
				throw std::invalid_argument("Maximum 3 inputs in Divide layer");

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
			chosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
			DstMemDesc = std::make_unique<dnnl::memory::desc>(*Inputs[0]->DstMemDesc);
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*Inputs[0]->DiffDstMemDesc);

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in Divide layer");
			}
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto n = 0ull; n < CDHW; n++)
				{
					Neurons[n] = Inputs[0]->Neurons[n];
#ifndef DNN_LEAN
					NeuronsD1[n] = Float(0);
#endif // DNN_LEAN
				}
				for (auto i = 1ull; i < Inputs.size(); i++)
					for (size_t n = 0; n < CDHW; n++)
						Neurons[n] /= Inputs[i]->Neurons[n];
			}
			else
			{
#endif
				if (Inputs.size() == 2)
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * CDHW;
						const auto end = start + CDHW;
						for (auto n = start; n < end; n++)
						{
							Neurons[n] = Inputs[0]->Neurons[n] / Inputs[1]->Neurons[n];
#ifndef DNN_LEAN
							NeuronsD1[n] = Float(0);
#endif // DNN_LEAN
						}
					});

				}
				else
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * CDHW;
						const auto end = start + CDHW;
						for (auto n = start; n < end; n++)
						{
							Neurons[n] = Inputs[0]->Neurons[n];
#ifndef DNN_LEAN
							NeuronsD1[n] = Float(0);
#endif // DNN_LEAN
						}
						for (auto i = 1ull; i < Inputs.size(); i++)
							for (auto n = start; n < end; n++)
								Neurons[n] /= Inputs[i]->Neurons[n];
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (Inputs.size() == 2)
				{
					for (auto n = 0ull; n < CDHW; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / Inputs[1]->Neurons[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / FloatSquare(Inputs[1]->Neurons[n]);
					}
				}
				else if (Inputs.size() == 3)
				{
					for (auto n = 0ull; n < CDHW; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / (Inputs[1]->Neurons[n] * Inputs[2]->Neurons[n]);
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (FloatSquare(Inputs[1]->Neurons[n]) * Inputs[2]->Neurons[n]);
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (Inputs[1]->Neurons[n] * FloatSquare(Inputs[2]->Neurons[n]));
					}
				}
				else
				{
				}
			}
			else
			{
#endif
				if (Inputs.size() == 2)
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * CDHW;
						const auto end = start + CDHW;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / Inputs[1]->Neurons[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / FloatSquare(Inputs[1]->Neurons[n]);
						}
					});
				}
				else if (Inputs.size() == 3)
				{
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * CDHW;
						const auto end = start + CDHW;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / (Inputs[1]->Neurons[n] * Inputs[2]->Neurons[n]);
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (FloatSquare(Inputs[1]->Neurons[n]) * Inputs[2]->Neurons[n]);
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / (Inputs[1]->Neurons[n] * FloatSquare(Inputs[2]->Neurons[n]));
						}
					});
				}
				else
				{
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
