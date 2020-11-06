#pragma once
#include "Layer.h"

namespace dnn
{
	class Dropout final : public Layer
	{
	private:
		std::bernoulli_distribution DropoutDistribution;
		FloatVector NeuronsActive;

	public:
		const size_t PartialCDHW;
		const Float Keep;
		const Float Scale;

		Dropout(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.3)) :
			Layer(device, format, name, LayerTypes::Dropout, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			PartialCDHW(((inputs[0]->C* inputs[0]->D* inputs[0]->H* inputs[0]->W) / VectorSize)* VectorSize),
			Keep(Float(1) - dropout),
			Scale(Float(1) / (Float(1) - dropout)),
			DropoutDistribution(std::bernoulli_distribution(double(1) - dropout)),
			NeuronsActive(FloatVector())
		{
			assert(Inputs.size() == 1);
		}

		std::string Dropout::GetDescription() const final override
		{
			std::string description = GetDescriptionHeader();

			description.append(nwl + " Dropout:" + tab + FloatToString(Float(1) - Keep));
			description.append(nwl + " Scale:" + dtab + FloatToString(Scale));

			return description;
		}

		size_t Dropout::FanIn() const final override
		{
			return 1;
		}

		size_t Dropout::FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);
		}

		void SetBatchSize(const size_t batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			ZeroFloatVectorAllocate(NeuronsActive, batchSize * PaddedCDHW);
			for (auto n = 0ull; n < batchSize; n++)
				for (auto i = 0ull; i < CDHW; i++)
					NeuronsActive[n * PaddedCDHW + i] = Float(1);
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			if (training)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (size_t i = 0; i < PartialCDHW; i += VectorSize)
					{
						const auto neuronsActive = BernoulliVecFloat(Keep);
						neuronsActive.store_a(&NeuronsActive[i]);
						(neuronsActive * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
#ifndef DNN_LEAN
						VecFloat(0).store_nt(&NeuronsD1[i]);
#endif
					}
					for (size_t i = PartialCDHW; i < CDHW; i++)
					{
						NeuronsActive[i] = Bernoulli<Float>(Keep);
						Neurons[i] = NeuronsActive[i] * Scale * InputLayer->Neurons[i];
#ifndef DNN_LEAN
						NeuronsD1[i] = Float(0);
#endif
					}
				}
				else

				{
#endif
					for_i(batchSize, [=](size_t b)
					{
						const auto start = b * PaddedCDHW;
						const auto part = start + PartialCDHW;

						for (auto i = start; i < part; i += VectorSize)
						{
							const auto neuronsActive = BernoulliVecFloat(Keep);
							neuronsActive.store_a(&NeuronsActive[i]);
							(neuronsActive * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
#ifndef DNN_LEAN
							VecFloat(0).store_nt(&NeuronsD1[i]);
#endif
						}
						for (auto i = part; i < start + CDHW; i++)
						{
							NeuronsActive[i] = Bernoulli<Float>(Keep);
							Neurons[i] = NeuronsActive[i] * Scale * InputLayer->Neurons[i];
#ifndef DNN_LEAN
							NeuronsD1[i] = Float(0);
#endif
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto i = 0ull; i < PartialCDHW; i += VectorSize)
						VecFloat().load_a(&InputLayer->Neurons[i]).store_a(&Neurons[i]);
					for (auto i = PartialCDHW; i < CDHW; i++)
						Neurons[i] = InputLayer->Neurons[i];
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
					{
						const auto start = b * PaddedCDHW;
						const auto part = start + PartialCDHW;
						for (auto i = start; i < part; i += VectorSize)
							(VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
						for (auto i = part; i < start + CDHW; i++)
							Neurons[i] = InputLayer->Neurons[i];
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto i = 0ull; i < PartialCDHW; i += VectorSize)
					mul_add(VecFloat().load_a(&NeuronsActive[i]), VecFloat().load_a(&NeuronsD1[i]), VecFloat().load_a(&InputLayer->NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
				for (auto i = PartialCDHW; i < CDHW; i++)
					InputLayer->NeuronsD1[i] += NeuronsActive[i] * NeuronsD1[i];
			}
			else
			{
#endif
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t b)
				{
					const auto start = b * PaddedCDHW;
					const auto part = start + PartialCDHW;

					for (auto i = start; i < part; i += VectorSize)
						mul_add(VecFloat().load_a(&NeuronsActive[i]), VecFloat().load_a(&NeuronsD1[i]), VecFloat().load_a(&InputLayer->NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
					for (auto i = part; i < start + CDHW; i++)
						InputLayer->NeuronsD1[i] += NeuronsActive[i] * NeuronsD1[i];
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