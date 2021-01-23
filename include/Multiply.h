#pragma once
#include "Layer.h"

namespace dnn
{
	class Multiply final : public Layer
	{
	public:
		Multiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Multiply, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);

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

		size_t FanOut() const  final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize)  final override
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

		void ForwardProp(const size_t batchSize, const bool training)  final override
		{
			DNN_UNREF_PAR(training);

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat In, Out;
					VecFloat vecZero = VecFloat(0);
					for (auto n = 0ull; n < PaddedCDHW; n+=VectorSize)
					{
						In.load_a(&Inputs[0]->Neurons[n]);
						In.store_a(&Neurons[n]);
#ifndef DNN_LEAN
						vecZero.store_nt(&NeuronsD1[n]);
#endif // DNN_LEAN
					}
					for (auto i = 1ull; i < inputs; i++)
						for (auto n = 0ull; n < PaddedCDHW; n += VectorSize)
						{
							Out.load_a(&Neurons[n]);
							In.load_a(&Inputs[i]->Neurons[n]);
							Out *= In;
							Out.store_a(&Neurons[n]);
						}
				}
				else
					for (auto n = 0ull; n < CDHW; n++)
					{
						Neurons[n] = Inputs[0]->Neurons[n];
#ifndef DNN_LEAN
						NeuronsD1[n] = Float(0);
#endif // DNN_LEAN
					}
					for (auto i = 1ull; i < inputs; i++)
						for (auto n = 0ull; n < CDHW; n++)
							Neurons[n] *= Inputs[i]->Neurons[n];
			    }
			}
			else
			{
#endif
				if (inputs == 2)
				{
					if (!plain)
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							VecFloat InA, InB;
							const auto vecZero = VecFloat(0);
							for (auto w = start; w < end; w+=VectorSize)
							{
								InA.load_a(&Inputs[0]->Neurons[w]);
								InB.load_a(&Inputs[1]->Neurons[w]);
								(InA * InB).store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
						});
					else
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto w = start; w < end; w++)
							{
								Neurons[w] = Inputs[0]->Neurons[w] * Inputs[1]->Neurons[w];
	#ifndef DNN_LEAN
								NeuronsD1[w] = Float(0);
	#endif // DNN_LEAN
							}
						});
				}
				else
				{
					if (!plain)
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							VecFloat In, Out;
							const auto vecZero = VecFloat(0);
							for (auto w = start; w < end; w+=VectorSize)
							{
								In.load_a(&Inputs[0]->Neurons[w]);
								In.store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
							for (auto i = 1ull; i < inputs; i++)
								for (auto w = start; w < end; w+=VectorSize)
								{
									In.load_a(&Inputs[i]->Neurons[w]);
									Out.load_a(&Neurons[w]);
									Out *= In;
									Out.store_a(&Neurons[w]);
								}
						});
					else
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto w = start; w < end; w++)
							{
								Neurons[w] = Inputs[0]->Neurons[w];
	#ifndef DNN_LEAN
								NeuronsD1[w] = Float(0);
	#endif // DNN_LEAN
							}
							for (auto i = 1ull; i < inputs; i++)
								for (auto w = start; w < end; w++)
									Neurons[w] *= Inputs[i]->Neurons[w];
						});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const size_t batchSize)  final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto inputs = Inputs.size();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (inputs == 2)
				{
					for (auto n = 0ull; n < CDHW; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n] * Inputs[1]->Neurons[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n];
					}
				}
				else if (inputs == 3)
				{
					for (auto n = 0ull; n < CDHW; n++)
					{
						Inputs[0]->NeuronsD1[n] += NeuronsD1[n] * Inputs[1]->Neurons[n] * Inputs[2]->Neurons[n];
						Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] * Inputs[2]->Neurons[n];
						Inputs[2]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] * Inputs[1]->Neurons[n];
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
					if (!plain)
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * PaddedCDHW;
							const auto end = start + PaddedCDHW;
							for (auto w = start; w < end; w+=VectorSize)
							{
								mul_add(VecFloat().load_a(&Inputs[1]->Neurons[w]), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&Inputs[0]->NeuronsD1[w])).store_a(&Inputs[0]->NeuronsD1[w]);
								mul_add(VecFloat().load_a(&Inputs[0]->Neurons[w]), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&Inputs[1]->NeuronsD1[w])).store_a(&Inputs[1]->NeuronsD1[w]);
							}
						});
					else
						for_i(batchSize, threads, [=](size_t n)
						{
							const auto start = n * CDHW;
							const auto end = start + CDHW;
							for (auto w = start; w < end; w++)
							{
								Inputs[0]->NeuronsD1[w] += NeuronsD1[w] * Inputs[1]->Neurons[w];
								Inputs[1]->NeuronsD1[w] += NeuronsD1[w] * Inputs[0]->Neurons[w];
							}
						});
				}
				else if (inputs == 3)
				{
					for_i(batchSize, threads, [=](size_t b)
					{
						const auto start = b * PaddedCDHW;
						const auto end = start + CDHW;
						for (auto n = start; n < end; n++)
						{
							Inputs[0]->NeuronsD1[n] += NeuronsD1[n] * Inputs[1]->Neurons[n] * Inputs[2]->Neurons[n];
							Inputs[1]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] * Inputs[2]->Neurons[n];
							Inputs[2]->NeuronsD1[n] += NeuronsD1[n] * Inputs[0]->Neurons[n] * Inputs[1]->Neurons[n];
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
