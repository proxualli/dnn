#pragma once
#include "Layer.h"

namespace dnn
{
	class Divide final : public Layer
	{
	public:
		Divide(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Divide, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1 && Inputs.size() < 4);

			if (Inputs.size() > 3)
				throw std::invalid_argument("Maximum 3 inputs in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);

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
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				if (Format == dnnl::memory::format_tag::any)
					chosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					chosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
					if (chosenFormat != GetDataFmt(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(*DstMemDesc == *Inputs[i]->DstMemDesc);
				if (*DstMemDesc != *Inputs[i]->DstMemDesc)
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer");
			}
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto elements = batchSize * size;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto part = (size / VectorSize) * VectorSize;
			const auto inputs = Inputs.size();

			DNN_UNREF_PAR(training);

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
				for (auto i = 1ull; i < inputs; i++)
					for (size_t n = 0; n < CDHW; n++)
						Neurons[n] /= Inputs[i]->Neurons[n];
			}
			else
			{
#endif
				if (inputs == 2)
				{
					for_i(batchSize, threads, [=](size_t b)
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
					for_i(batchSize, threads, [=](size_t b)
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

			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW : PaddedCDHW;
			const auto elements = batchSize * size;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto part = (size / VectorSize) * VectorSize;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (inputs == 2)
				{
					for (auto n = 0ull; n < CDHW; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n] / Inputs[1]->Neurons[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] / FloatSquare(Inputs[1]->Neurons[n]);
					}
				}
				else if (inputs == 3)
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
				if (inputs == 2)
				{
					for_i(batchSize, threads, [=](size_t b)
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
				else if (inputs == 3)
				{
					for_i(batchSize, threads, [=](size_t b)
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
