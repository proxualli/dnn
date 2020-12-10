#pragma once
#include "Layer.h"
#include <stdexcept>

namespace dnn
{
	class ChannelZeroPad final : public Layer
	{
	public:
		ChannelZeroPad(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<std::shared_ptr<Layer>>& inputs, const size_t c) :
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
			if (GetDataFmt(*InputLayer->DstMemDesc) != BlockedFmt)
				throw std::runtime_error("Blocked format expected in ChannelZeroPad");

			chosenFormat = BlockedFmt;
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			DNN_UNREF_PAR(training);

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
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
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;
						const auto offsetNinput = n * InputLayer->CDHW;

						for (auto c = 0ull; c < C; c++)
						{
							Neurons[c + offsetN] = c < InputLayer->C ? InputLayer->Neurons[c + offsetNinput] : Float(0);

#ifndef DNN_LEAN
							NeuronsD1[c + offsetN] = Float(0);
#endif // DNN_LEAN
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					const auto vecZero = VecFloat(Float(0));

					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						const auto offsetC = c * HW;
						const auto skip = c >= InputLayer->PaddedC;

						for (auto h = 0ull; h < H; h++)
						{
							const auto offsetH = offsetC + h * strideH;

							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								skip ? vecZero.store_a(&Neurons[w]) : (VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						const auto vecZero = VecFloat(Float(0));

						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto offsetC = n * PaddedCDHW + c * HW;
							const auto offsetCinput = n * InputLayer->PaddedCDHW + c * HW;

							if (c >= InputLayer->PaddedC)
							{
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									const auto offsetHinput = offsetCinput + h * strideH;

									for (auto w = 0ull; w < strideH; w += VectorSize)
									{
										vecZero.store_a(&Neurons[w + offsetH]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[w + offsetH]);
#endif // DNN_LEAN
									}
								}
							}
							else
							{
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									const auto offsetHinput = offsetCinput + h * strideH;

									for (auto w = 0ull; w < strideH; w += VectorSize)
									{
										(VecFloat().load_a(&InputLayer->Neurons[w + offsetHinput])).store_a(&Neurons[w + offsetH]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[w + offsetH]);
#endif // DNN_LEAN
									}
								}
							}

						}
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						InputLayer->NeuronsD1[c] += NeuronsD1[c];
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;
						const auto offsetNinput = n * InputLayer->CDHW;

						for (auto c = 0ull; c < InputLayer->C; c++)
							InputLayer->NeuronsD1[c + offsetNinput] += NeuronsD1[c + offsetN];
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const size_t strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
					{
						const auto offsetC = c * HW;
						for (auto h = 0ull; h < H; h++)
						{
							const auto offsetH = offsetC + h * strideH;
							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								(VecFloat().load_a(&InputLayer->NeuronsD1[w]) + VecFloat().load_a(&NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
						{
							const auto offsetC = n * PaddedCDHW + c * HW;
							const auto offsetCinput = n * InputLayer->PaddedCDHW + c * HW;

							for (auto h = 0ull; h < H; h++)
							{
								const auto offsetH = offsetC + h * strideH;
								const auto offsetHinput = offsetCinput + h * strideH;

								for (auto w = 0ull; w < strideH; w += VectorSize)
									(VecFloat().load_a(&InputLayer->NeuronsD1[w + offsetHinput]) + VecFloat().load_a(&NeuronsD1[w + offsetH])).store_a(&InputLayer->NeuronsD1[w + offsetHinput]);
							}
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
