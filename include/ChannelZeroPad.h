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

			FwdZeroGradient = Float(1);
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
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

			assert(ChosenFormat == GetMemoryFormat(*InputLayer->DstMemDesc));
			if (ChosenFormat != GetMemoryFormat(*InputLayer->DstMemDesc))
				throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + InputLayer->Name);
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto threads = GetThreads(batchSize * GetElementsCount());
			const auto strideHW = HW() * VectorSize;

			DNN_UNREF_PAR(training);

			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						PRAGMA_OMP_SIMD()
						for (auto c = 0ull; c < InputLayer->PaddedC; c++)
						{
							Neurons[OffsetPaddedMem(0, c, 0, 0)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, 0, 0)];
#ifndef DNN_LEAN
							NeuronsD1[OffsetPaddedMem(0, c, 0, 0)] = Float(0);
						}
#endif // DNN_LEAN
						PRAGMA_OMP_SIMD()
						for (auto c = InputLayer->PaddedC; c < PaddedC; c++)
						{
							Neurons[OffsetPaddedMem(0, c, 0, 0)] = Float(0);
#ifndef DNN_LEAN
							NeuronsD1[OffsetPaddedMem(0, c, 0, 0)] = Float(0);
#endif // DNN_LEAN		
						}
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
							PRAGMA_OMP_SIMD()
							for (auto c = 0ull; c < InputLayer->PaddedC; c++)
							{
								Neurons[OffsetPaddedMem(n, c, 0, 0)] = InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, 0, 0)];
#ifndef DNN_LEAN
								NeuronsD1[OffsetPaddedMem(n, c, 0, 0)] = Float(0);
							}
#endif // DNN_LEAN
							PRAGMA_OMP_SIMD()
							for (auto c = InputLayer->PaddedC; c < PaddedC; c++)
							{
								Neurons[OffsetPaddedMem(n, c, 0, 0)] = Float(0);
#ifndef DNN_LEAN
								NeuronsD1[OffsetPaddedMem(n, c, 0, 0)] = Float(0);
#endif // DNN_LEAN		
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto inputOffset = n * InputLayer->CDHW();
							const auto outputOffset = n * CDHW();
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
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						VecFloat In;
						for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
						{
							const auto offset = c * HW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								In.load_a(&InputLayer->Neurons[hw + offset]);
								In.store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
								VecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
							}
							
						}
						for (auto c = InputLayer->PaddedC; c < PaddedC; c += VectorSize)
						{
							const auto offset = c * HW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								VecFloat(0).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
								VecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
							}
						}
					}
					else
						for (auto c = 0ull; c < C; c++)
						{
							const auto offsetC = c * HW();
							const auto skip = c >= InputLayer->PaddedC;
							for (auto hw = 0ull; hw < HW(); hw++)
							{
								Neurons[hw + offsetC] = skip ? Float(0) : InputLayer->Neurons[hw + offsetC];
#ifndef DNN_LEAN
								NeuronsD1[hw + offsetC] = Float(0);
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
							VecFloat In;
							for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayer->OffsetPaddedMem(n, c, 0, 0);
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
							for (auto c = InputLayer->PaddedC; c < PaddedC; c += VectorSize)
							{
								const auto outputOffset = n * PaddedCDHW() + c * HW();
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									VecFloat(0).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN								
									VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif // DNN_LEAN
								}
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->C; c++)
							{
								const auto inputOffset = n * InputLayer->CDHW() + c * HW();
								const auto outputOffset = n * CDHW() + c * HW();
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Neurons[hw + outputOffset] = InputLayer->Neurons[hw + inputOffset];
#ifndef DNN_LEAN
									NeuronsD1[hw + outputOffset] = Float(0);
#endif // DNN_LEAN
								}
							}
							for (auto c = InputLayer->C; c < C; c++)
							{
								const auto outputOffset = n * CDHW() + c * HW();
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Neurons[hw + outputOffset] = Float(0);
#ifndef DNN_LEAN
									NeuronsD1[hw + outputOffset] = Float(0);
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
			const auto threads = GetThreads(batchSize * GetElementsCount());
			const auto strideHW = HW() * VectorSize;

			if (GetMemoryNDims(*InputLayerBwd->DstMemDesc) == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)

					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						InputLayerBwd->NeuronsD1[c] += NeuronsD1[c];
				else
#endif
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputOffset = n * CDHW();
						const auto inputOffset = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							InputLayerBwd->NeuronsD1[c + inputOffset] += NeuronsD1[c + outputOffset];
					});
			}
			else
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						for (auto c = 0ull; c < InputLayerBwd->PaddedC; c += VectorSize)
						{
							const auto inputOffset = InputLayerBwd->OffsetPaddedMem(0, c, 0, 0);
							const auto outputOffset = OffsetPaddedMem(0, c, 0, 0);
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								(VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]) + VecFloat().load_a(&NeuronsD1[hw + outputOffset])).store_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
						}
					}
					else
					{
						for (auto c = 0ull; c < InputLayerBwd->PaddedC; c += VectorSize)
						{
							const auto inputOffset = c * HW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								(VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]) + VecFloat().load_a(&NeuronsD1[hw + inputOffset])).store_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
						}
					}
				}
				else
				{
#endif
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayerBwd->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayerBwd->OffsetPaddedMem(n, c, 0, 0);
								const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									(VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]) + VecFloat().load_a(&NeuronsD1[hw + outputOffset])).store_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayerBwd->C; c++)
							{
								const auto inputOffset = n * InputLayerBwd->CDHW() + c * HW();
								const auto outputOffset = n * CDHW() + c * HW();
								for (auto hw = 0ull; hw < HW(); hw++)
									InputLayerBwd->NeuronsD1[hw + inputOffset] += NeuronsD1[hw + outputOffset];
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