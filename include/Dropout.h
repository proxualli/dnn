#pragma once
#include "Layer.h"

namespace dnn
{
	class Dropout final : public Layer
	{
	public:
		const bool LocalValue;
		Float Keep;
		Float Scale;

		FloatArray NeuronsActive;

		Dropout(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.5), const bool localValue = false) :
			Layer(device, format, name, LayerTypes::Dropout, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, false, false, dropout > 0),
			LocalValue(localValue),
			Keep(Float(1) - dropout),
			Scale(Float(1) / (Float(1) - dropout)),
			NeuronsActive(FloatArray())
		{
			assert(Inputs.size() == 1);
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
			Layer::UpdateResolution();
		}

		void UpdateDropout(const Float dropout)
		{
			if (!LocalValue)
			{
				Enabled = dropout > 0;
				Keep = Float(1) - dropout;
				Scale = Float(1) / Keep;
			}
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

			if (Enabled)
			{
				NeuronsActive.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto n = 0ull; n < batchSize; n++)
					for (auto i = 0ull; i < CDHW; i++)
						NeuronsActive[n * PaddedCDHW + i] = Float(1);
			}
			else
				NeuronsActive.release();
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto part = GetVectorPart(size);
			const auto elements = batchSize * size;
			const auto threads = GetThreads(elements);

			if (Enabled && training)
			{
				
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto i = 0ull; i < part; i += VectorSize)
					{
						VecFloat mask = BernoulliVecFloat(Keep);
						mask.store_a(&NeuronsActive[i]);
						(mask * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
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
					for_i(batchSize, threads, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;
						VecFloat mask;
						for (auto i = start; i < end; i += VectorSize)
						{
							mask = BernoulliVecFloat(Keep);
							mask.store_a(&NeuronsActive[i]);
							(mask * Scale * VecFloat().load_a(&InputLayer->Neurons[i])).store_a(&Neurons[i]);
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
					for_i(batchSize, threads, [=](UInt b)
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
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto part = GetVectorPart(size);
			const auto elements = batchSize * size;
			const auto threads = GetThreads(elements);
			
			if (Enabled)
			{
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
					for_i(batchSize, threads, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;
						for (auto i = start; i < end; i += VectorSize)
							mul_add(VecFloat().load_a(&NeuronsActive[i]), VecFloat().load_a(&NeuronsD1[i]), VecFloat().load_a(&InputLayer->NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
						for (auto i = end; i < start + size; i++)
							InputLayer->NeuronsD1[i] += NeuronsActive[i] * NeuronsD1[i];
					});
			}
			else
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto i = 0ull; i < part; i += VectorSize)
						(VecFloat().load_a(&InputLayer->NeuronsD1[i]) + VecFloat().load_a(&NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
					for (auto i = part; i < size; i++)
						InputLayer->NeuronsD1[i] += NeuronsD1[i];
				}
				else
#endif
					for_i(batchSize, threads, [=](UInt b)
					{
						const auto start = b * size;
						const auto end = start + part;
						for (auto i = start; i < end; i += VectorSize)
							(VecFloat().load_a(&InputLayer->NeuronsD1[i]) + VecFloat().load_a(&NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
						for (auto i = end; i < start + size; i++)
							InputLayer->NeuronsD1[i] += NeuronsD1[i];
					});
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}