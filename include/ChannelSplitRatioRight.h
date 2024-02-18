#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelSplitRatioRight final : public Layer
	{
	public:
		const Float Ratio;
		const UInt ChannelsLeft;
		
		ChannelSplitRatioRight(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float ratio = Float(0.375)) :
			Layer(device, format, name, LayerTypes::ChannelSplitRatioRight, 0, 0, UInt(std::roundf(Float(inputs[0]->C)) * ratio), inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Ratio(ratio),
			ChannelsLeft(UInt(std::roundf(Float(inputs[0]->C)) * (std::roundf(Float(1)) - ratio)))
		{
			assert(Inputs.size() == 1);
			assert(Ratio > Float(0));
			assert(Ratio < Float(1));
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Ratio:") + dtab + FloatToString(Ratio));

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
			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (NeuronsFormat == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetMemoryFormat(*InputLayer->DstMemDesc);
					if (ChosenFormat != GetMemoryFormat(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto threads = GetThreads(batchSize * (plain ? CDHW() : PaddedCDHW()), Float(10));
			const auto strideHW = HW() * VectorSize;

			const auto padded = C == PaddedC;
			const auto part = padded ? PaddedC : (PaddedC - VectorSize);

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (training)
				{
					if (!plain)
					{
						VecFloat In;
						for (auto c = 0ull; c < part; c += VectorSize)
						{
							const auto inputOffset = InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, 0, 0);
							const auto outputOffset = OffsetPaddedMem(0, c, 0, 0);
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[hw + inputOffset]);
								In.store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
								VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif // DNN_LEAN
							}
						}

						for (auto c = part; c < C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
								{
									Neurons[OffsetPaddedMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, h, w)];
#ifndef DNN_LEAN
									NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif  // DNN_LEAN
								}

						for (auto c = C; c < PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
								{
									Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
#ifndef DNN_LEAN
									NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif  // DNN_LEAN
								}

					}
					else
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (c + ChannelsLeft) * HW();
							const auto outputOffset = c * HW();
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
							{
								Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
								NeuronsD1[hw + outputOffset] = Float(0);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
					if (!plain)
					{
						VecFloat In;
						for (auto c = 0ull; c < part; c += VectorSize)
						{
							const auto inputOffset = InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, 0, 0);
							const auto outputOffset = OffsetPaddedMem(0, c, 0, 0);
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[hw + inputOffset]);
								In.store_a(&Neurons[hw + outputOffset]);
							}
						}

						for (auto c = part; c < C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									Neurons[OffsetPaddedMem(0, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, h, w)];

						for (auto c = C; c < PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
					}
					else
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (c + ChannelsLeft) * HW();
							const auto outputOffset = c * HW();
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
						}
					}
				}
			}
			else
			{
#endif
				if (training)
				{
					if (!plain)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							VecFloat In;
							for (auto c = 0ull; c < part; c += VectorSize)
							{
								const auto inputOffset = InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, 0, 0);
								const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[hw + inputOffset]);
									In.store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
									VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif // DNN_LEAN
								}
							}

							for (auto c = part; c < C; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, h, w)];
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif  // DNN_LEAN
									}

							for (auto c = C; c < PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif  // DNN_LEAN
									}
						});
					}
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto inputOffset = (n * InputLayer->CDHW()) + ((c + ChannelsLeft) * HW());
								const auto outputOffset = (n * CDHW()) + (c * HW());
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
									NeuronsD1[hw + outputOffset] = Float(0);
#endif // DNN_LEAN
								}
							}
						});
				}
				else
				{
					if (!plain)
					{						
						for_i(batchSize, threads, [=](UInt n)
						{
							VecFloat In;
							for (auto c = 0ull; c < part; c += VectorSize)
							{
								const auto inputOffset = InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, 0, 0);
								const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									In.load_a(&InputLayer->Neurons[hw + inputOffset]);
									In.store_a(&Neurons[hw + outputOffset]);
								}
							}

							for (auto c = part; c < C; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPaddedMem(n, c, h, w)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, h, w)];

							for (auto c = C; c < PaddedC; c++)
								for (auto h = 0ull; h < H; h++)
									PRAGMA_OMP_SIMD()
									for (auto w = 0ull; w < W; w++)
										Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);
						});
					}
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto inputOffset = (n * InputLayer->CDHW()) + ((c + ChannelsLeft) * HW());
								const auto outputOffset = (n * CDHW()) + (c * HW());
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
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
			const auto threads = GetThreads(batchSize * (plain ? CDHW() : PaddedCDHW()), Float(10));
			const auto strideHW = HW() * VectorSize;

			const auto padded = C == PaddedC;
			const auto part = padded ? PaddedC : (PaddedC - VectorSize);

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat inputD1, D1;
					for (auto c = 0ull; c < part; c += VectorSize)
					{
						const auto inputOffset = InputLayer->OffsetPaddedMem(9, c + ChannelsLeft, 0, 0);
						const auto outputOffset = OffsetPaddedMem(0, c, 0, 0);
						for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
						{
							D1.load_a(&NeuronsD1[hw + outputOffset]);
							inputD1.load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
							inputD1 += D1;
							inputD1.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
						}
					}

					for (auto c = part; c < C; c++)
						for (auto h = 0ull; h < H; h++)
							PRAGMA_OMP_SIMD()
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, h, w)] += NeuronsD1[OffsetPaddedMem(0, c, h, w)];

					/*for (auto c = C; c < PaddedC; c++)
						for (auto h = 0ull; h < H; h++)
							PRAGMA_OMP_SIMD()
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c + ChannelsLeft, h, w)] += NeuronsD1[OffsetPaddedMem(0, c, h, w)];*/
					
				}
				else
					for (auto c = 0ull; c < C; c++)
					{
						const auto inputOffset = (c + ChannelsLeft) * HW();
						const auto outputOffset = c * HW();
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
					}
			}
			else
			{
#endif
				if (!plain)
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat inputD1, D1;
						for (auto c = 0ull; c < part; c += VectorSize)
						{
							const auto inputOffset = InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, 0, 0);
							const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								D1.load_a(&NeuronsD1[hw + outputOffset]);
								inputD1.load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								inputD1 += D1;
								inputD1.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
							}
						}

						for (auto c = part; c < C; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, h, w)] += NeuronsD1[OffsetPaddedMem(n, c, h, w)];

						/*for (auto c = C; c < PaddedC; c++)
							for (auto h = 0ull; h < H; h++)
								PRAGMA_OMP_SIMD()
								for (auto w = 0ull; w < W; w++)
									InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c + ChannelsLeft, h, w)] += NeuronsD1[OffsetPaddedMem(n, c, h, w)];*/
					});
				}
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto inputOffset = (n * InputLayer->CDHW()) + ((c + ChannelsLeft) * HW());
							const auto outputOffset = (n * CDHW()) + (c * HW());
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
						}
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