#pragma once
#include "Layer.h"

namespace dnn
{
	class Dropout final : public Layer
	{
	private:
		std::bernoulli_distribution DropoutDistribution;
		FloatArray NeuronsActive;
	
	public:
		const Float Keep;
		const Float Scale;

		Dropout(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.3)) :
			Layer(device, format, name, LayerTypes::Dropout, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Keep(Float(1) - dropout),
			Scale(Float(1) / (Float(1) - dropout)),
			DropoutDistribution(std::bernoulli_distribution(double(1) - dropout)),
			NeuronsActive(FloatArray())
		{
			assert(Inputs.size() == 1);
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Dropout:") + tab + FloatToString(Float(1) - Keep));
			description.append(nwl + std::string(" Scale:") + dtab + FloatToString(Scale));

			return description;
		}

		UInt FanIn() const final override
		{
			return 1;
		}

		UInt FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			DNN_UNREF_PAR(batchSize);

			ChosenFormat = GetDataFmt(*InputLayer->DstMemDesc);

			DstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DstMemDesc);
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(*InputLayer->DiffDstMemDesc);
		}

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			NeuronsActive.resize(batchSize * PaddedCDHW);
			for (auto n = 0ull; n < batchSize; n++)
				for (auto i = 0ull; i < CDHW; i++)
					NeuronsActive[n * PaddedCDHW + i] = Float(1);
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto size = IsPlainFormat() ? CDHW : PaddedCDHW;
	        const auto part = (size / VectorSize) * VectorSize;

			if (training)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto i = 0ull; i < part; i += VectorSize)
					{
						const auto neuronsActive = BernoulliVecFloat(Keep);
						neuronsActive.store_a(&NeuronsActive[i]);
						(neuronsActive * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
#ifndef DNN_LEAN
						VecFloat(0).store_nt(&NeuronsD1[i]);
#endif
					}
					for (auto i = part; i < size; i++)
					{
						NeuronsActive[i] = Bernoulli<Float>(Keep);
						Neurons[i] = NeuronsActive[i] * Scale * InputLayer->Neurons[i];
#ifndef DNN_LEAN
						NeuronsD1[i] = Float(0);
#endif
					}
				}
				else
#endif
					for_i(batchSize, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;

						for (auto i = start; i < end; i += VectorSize)
						{
							const auto neuronsActive = BernoulliVecFloat(Keep);
							neuronsActive.store_a(&NeuronsActive[i]);
							(neuronsActive * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
#ifndef DNN_LEAN
							VecFloat(0).store_nt(&NeuronsD1[i]);
#endif
						}
						for (auto i = end; i < start + size; i++)
						{
							NeuronsActive[i] = Bernoulli<Float>(Keep);
							Neurons[i] = NeuronsActive[i] * Scale * InputLayer->Neurons[i];
#ifndef DNN_LEAN
							NeuronsD1[i] = Float(0);
#endif
						}
					});
			}
			else
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto i = 0ull; i < part; i += VectorSize)
						VecFloat().load_a(&InputLayer->Neurons[i]).store_a(&Neurons[i]);
					for (auto i = part; i < size; i++)
						Neurons[i] = InputLayer->Neurons[i];
				}
				else
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;
						for (auto i = start; i < end; i += VectorSize)
							(VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
						for (auto i = end; i < start + size; i++)
							Neurons[i] = InputLayer->Neurons[i];
					});
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif

			const auto size = IsPlainFormat() ? CDHW : PaddedCDHW;
	        const auto part = (size / VectorSize) * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				for (auto i = 0ull; i < part; i += VectorSize)
					mul_add(VecFloat().load_a(&NeuronsActive[i]), VecFloat().load_a(&NeuronsD1[i]), VecFloat().load_a(&InputLayer->NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
				for (auto i = part; i < size; i++)
					InputLayer->NeuronsD1[i] += NeuronsActive[i] * NeuronsD1[i];
			}
			else
#endif
				for_i(batchSize, LIGHT_COMPUTE, [=](UInt b)
				{
					const auto start = b * size;
					const auto end = start + part;

					for (auto i = start; i < end; i += VectorSize)
						mul_add(VecFloat().load_a(&NeuronsActive[i]), VecFloat().load_a(&NeuronsD1[i]), VecFloat().load_a(&InputLayer->NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
					for (auto i = end; i < start + size; i++)
						InputLayer->NeuronsD1[i] += NeuronsActive[i] * NeuronsD1[i];
				});


#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}
