#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelZeroPad final : public Layer
	{
	public:
		ChannelZeroPad(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const UInt c) :
			Layer(device, format, name, LayerTypes::ChannelZeroPad, 0, 0, c, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() == 1);

			assert(InputLayer->C >= 1);
			assert(InputLayer->C < C);
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader();
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
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
					if (ChosenFormat != GetDataFmt(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			assert(ChosenFormat == GetDataFmt(*InputLayer->DstMemDesc));
			if (ChosenFormat != GetDataFmt(*InputLayer->DstMemDesc))
				throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + InputLayer->Name);
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;

			DNN_UNREF_PAR(training);

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
						for (auto c = 0ull; c < PaddedC; c++)
						{
							Neurons[c] = c < InputLayer->PaddedC ? InputLayer->Neurons[c] : Float(0);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
						}
					else
						for (auto c = 0ull; c < C; c++)
						{
							Neurons[c] = c < InputLayer->C ? InputLayer->Neurons[c] : Float(0);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
						}
				}
				else
				{
#endif
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto outputOffset = n * PaddedCDHW;
							const auto inputOffset = n * InputLayer->PaddedCDHW;

							for (auto c = 0ull; c < PaddedC; c++)
							{
								Neurons[c + outputOffset] = c < InputLayer->PaddedC ? InputLayer->Neurons[c + inputOffset] : Float(0);
#ifndef DNN_LEAN
								NeuronsD1[c + outputOffset] = Float(0);
#endif // DNN_LEAN
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto outputOffset = n * CDHW;
							const auto inputOffset = n * InputLayer->CDHW;

							for (auto c = 0ull; c < C; c++)
							{
								Neurons[c + outputOffset] = c < InputLayer->C ? InputLayer->Neurons[c + inputOffset] : Float(0);
#ifndef DNN_LEAN
								NeuronsD1[c + outputOffset] = Float(0);
#endif // DNN_LEAN
							}
						});
			
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto strideH = HW * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						const auto vecZero = VecFloat(0);
						VecFloat In;
						UInt inputOffset;
						for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
						{
							inputOffset = c * HW;
							for (auto w = 0ull; w < strideH; w += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[w + inputOffset]);
								In.store_a(&Neurons[w + inputOffset]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w + inputOffset]);
#endif // DNN_LEAN
							}
							
						}
						for (auto c = InputLayer->PaddedC; c < PaddedC; c += VectorSize)
						{
							inputOffset = c * HW;
							for (auto w = 0ull; w < strideH; w += VectorSize)
							{
								vecZero.store_a(&Neurons[w + inputOffset]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w + inputOffset]);
#endif // DNN_LEAN
							}
						}
					}
					else
						for (auto c = 0ull; c < C; c++)
						{
							const auto offsetC = c * HW;
							const auto skip = c >= InputLayer->PaddedC;

							for (auto w = 0ull; w < HW; w++)
							{
								Neurons[w + offsetC] = skip ? Float(0) : InputLayer->Neurons[w + offsetC];
#ifndef DNN_LEAN
								NeuronsD1[w + offsetC] = Float(0);
#endif // DNN_LEAN
							}
							
						}
				}
				else
				{
#endif
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto vecZero = VecFloat(0);
							VecFloat In;
							UInt inputOffset, outputOffset;
							for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
							{
								inputOffset = n * InputLayer->PaddedCDHW + c * HW;
								outputOffset = n * PaddedCDHW + c * HW;
								for (auto w = 0ull; w < strideH; w += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[w + inputOffset]);
									In.store_a(&Neurons[w + outputOffset]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[w + outputOffset]);
#endif // DNN_LEAN
								}
								
								
							}
							for (auto c = InputLayer->PaddedC; c < PaddedC; c += VectorSize)
							{
								outputOffset = n * PaddedCDHW + c * HW;
								for (auto w = 0ull; w < strideH; w += VectorSize)
								{
									vecZero.store_a(&Neurons[w + outputOffset]);
#ifndef DNN_LEAN								
									vecZero.store_nt(&NeuronsD1[w + outputOffset]);
#endif // DNN_LEAN
								}
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							UInt inputOffset, outputOffset;
							for (auto c = 0ull; c < InputLayer->C; c++)
							{
								inputOffset = n * InputLayer->CDHW + c * HW;
								outputOffset = n * CDHW + c * HW;
								
								for (auto w = 0ull; w < HW; w++)
								{
									Neurons[w + outputOffset] = InputLayer->Neurons[w + inputOffset];
#ifndef DNN_LEAN
									NeuronsD1[w + outputOffset] = Float(0);
#endif // DNN_LEAN
								}
							}
							for (auto c = InputLayer->C; c < C; c++)
							{
								outputOffset = n * CDHW + c * HW;
								for (auto w = 0ull; w < HW; w++)
								{
									Neurons[w + outputOffset] = Float(0);
#ifndef DNN_LEAN
									NeuronsD1[w + outputOffset] = Float(0);
#endif // DNN_LEAN
								}
							}
						});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
					for (auto c = 0ull; c < InputLayer->C; c++)
						InputLayer->NeuronsD1[c] += NeuronsD1[c];
				else
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](UInt n)
					{
						const auto offsetN = n * CDHW;
						const auto offsetNinput = n * InputLayer->CDHW;

						for (auto c = 0ull; c < InputLayer->C; c++)
							InputLayer->NeuronsD1[c + offsetNinput] += NeuronsD1[c + offsetN];
					});
			}
			else
			{
				const auto strideH = HW * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					UInt inputOffset;
					for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
					{
						inputOffset = c * HW;
						for (auto w = 0ull; w < strideH; w += VectorSize)
							(VecFloat().load_a(&InputLayer->NeuronsD1[w + inputOffset]) + VecFloat().load_a(&NeuronsD1[w + inputOffset])).store_a(&InputLayer->NeuronsD1[w + inputOffset]);
					}
				}
				else
				{
#endif
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							UInt inputOffset, outputOffset;
							for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
							{
								outputOffset = n * PaddedCDHW + c * HW;
								inputOffset = n * InputLayer->PaddedCDHW + c * HW;

								for (auto w = 0ull; w < strideH; w += VectorSize)
									(VecFloat().load_a(&InputLayer->NeuronsD1[w + inputOffset]) + VecFloat().load_a(&NeuronsD1[w + outputOffset])).store_a(&InputLayer->NeuronsD1[w + inputOffset]);
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							UInt inputOffset, outputOffset;
							for (auto c = 0ull; c < InputLayer->C; c++)
							{
								inputOffset = n * InputLayer->CDHW + c * HW;
								outputOffset = n * CDHW + c * HW;
								
								for (auto w = 0ull; w < HW; w++)
									InputLayer->NeuronsD1[w + inputOffset] += NeuronsD1[w + outputOffset];
							}
						});
#ifdef DNN_STOCHASTIC
				}
#endif
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}
