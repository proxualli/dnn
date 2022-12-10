#pragma once
#include "Layer.h"

namespace dnn
{
	class Concat final : public Layer
	{
	private:
		std::vector<dnnl::memory::desc> srcsMemsDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::concat::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::concat> fwd;
#endif

		auto InputChannels(const std::vector<Layer*>& inputs)
		{
			auto channels = 0ull;
			for (auto layer : inputs)
				channels += layer->C;

			return channels;
		}

	public:
		Concat(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Concat, 0, 0, InputChannels(inputs), inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
			assert(Inputs.size() > 1);
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
			if (InputLayer->DstMemDesc->get_ndims() == 2)
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

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(ChosenFormat == GetDataFmt(*Inputs[i]->DstMemDesc));
				if (ChosenFormat != GetDataFmt(*Inputs[i]->DstMemDesc))
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Inputs[i]->Name);
			}

			srcsMemsDesc = std::vector<dnnl::memory::desc>();
			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				if (Inputs[i]->DstMemDesc->get_ndims() == 2)
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C) }), dnnl::memory::data_type::f32, ChosenFormat));
				else
					srcsMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C), dnnl::memory::dim(Inputs[i]->H), dnnl::memory::dim(Inputs[i]->W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::concat::primitive_desc>(dnnl::concat::primitive_desc(Device.engine , *DstMemDesc, 1, srcsMemsDesc));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < InputsFwd.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcsMemsDesc[i], Device.engine, Inputs[i]->Neurons.data())});

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::concat>(dnnl::concat(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			if (training)
			{
#ifdef DNN_LEAN
				DNN_UNREF_PAR(batchSize);

#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
#else
				dnnl::concat(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
#else
				const auto plain = IsPlainFormat();
				const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
				const auto threads = GetThreads(elements, Float(0.1));
				
				const auto strideHW = HW() * VectorSize;

#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						const auto vecZero = VecFloat(0);
						VecFloat In;
						auto channelOffset = 0ull;
						UInt inputIndex, outputIndex;
						for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
						{
							for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->PaddedC; c += VectorSize)
							{
								inputIndex = (c - channelOffset) * HW();
								outputIndex = c * HW();
								for (auto w = 0ull; w < strideHW; w += VectorSize)
								{
									In.load_a(&Inputs[inputLayer]->Neurons[w + inputIndex]);
									In.store_a(&Neurons[w + outputIndex]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[w + outputIndex]);
#endif
								}
							}
							channelOffset += Inputs[inputLayer]->PaddedC;
						}
					}
					else
					{
						auto channelOffset = 0ull;
						UInt inputIndex, outputIndex;
						for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
						{
							for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->C; c++)
							{
								inputIndex = (c - channelOffset) * HW();
								outputIndex = c * HW();
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Neurons[outputIndex + hw] = Inputs[inputLayer]->Neurons[inputIndex + hw];
#ifndef DNN_LEAN
									NeuronsD1[outputIndex + hw] = Float(0);
#endif
								}
							}
							channelOffset += Inputs[inputLayer]->C;
						}
					}
				}
				else
				{
#endif
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto outputSampleOffset = n * PaddedCDHW();
							auto channelOffset = 0ull;
							UInt inputIndex, outputIndex;
							const auto vecZero = VecFloat(0);
							VecFloat In;
							for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
							{
								const auto inputSampleOffset = n * Inputs[inputLayer]->PaddedCDHW();
								for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->PaddedC; c += VectorSize)
								{
									inputIndex = ((c - channelOffset) * HW()) + inputSampleOffset;
									outputIndex = (c * HW()) + outputSampleOffset;
									for (auto w = 0ull; w < strideHW; w += VectorSize)
									{
										In.load_a(&Inputs[inputLayer]->Neurons[w + inputIndex]);
										In.store_a(&Neurons[w + outputIndex]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[w + outputIndex]);
#endif
									}
								}
								channelOffset += Inputs[inputLayer]->PaddedC;
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto outputSampleOffset = n * CDHW();
							auto channelOffset = 0ull;
							UInt inputIndex, outputIndex;
							for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
							{
								const auto inputSampleOffset = n * Inputs[inputLayer]->CDHW();
								for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->C; c++)
								{
									inputIndex = ((c - channelOffset) * HW()) + inputSampleOffset;
									outputIndex = (c * HW()) + outputSampleOffset;
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++) 
									{
										Neurons[outputIndex + hw] = Inputs[inputLayer]->Neurons[inputIndex + hw];
#ifndef DNN_LEAN
										NeuronsD1[outputIndex + hw] = Float(0);
#endif
									}
								}
								channelOffset += Inputs[inputLayer]->C;
							}
						});
#ifdef DNN_STOCHASTIC
				}
#endif
#endif
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
#else
				dnnl::concat(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
			const auto threads = GetThreads(elements, Float(0.1));
						
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					const auto strideH = HW() * VectorSize;
					auto channelOffset = 0ull;
					UInt inputIndex, outputIndex;
					VecFloat inputD1, D1;
					for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
					{
						for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->PaddedC; c += VectorSize)
						{
							inputIndex = ((c - channelOffset) * HW());
							outputIndex = (c * HW());
							for (auto w = 0ull; w < strideH; w += VectorSize)
							{
								inputD1.load_a(&Inputs[inputLayer]->NeuronsD1[w + inputIndex]);
								D1.load_a(&NeuronsD1[w + outputIndex]);
								(inputD1 + D1).store_a(&Inputs[inputLayer]->NeuronsD1[w + inputIndex]);
							}
						}									
						channelOffset += Inputs[inputLayer]->PaddedC;
					}
				}
				else
				{
					auto channelOffset = 0ull;
					UInt inputIndex, outputIndex;
					for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
					{
						for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->C; c++)
						{
							inputIndex = ((c - channelOffset) * HW());
							outputIndex = (c * HW());
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								Inputs[inputLayer]->NeuronsD1[inputIndex + hw] += NeuronsD1[outputIndex + hw];
						}
						channelOffset += Inputs[inputLayer]->C;
					}
				}
			}
			else
			{
#endif
				if (!plain)
				{
					const auto strideH = HW() * VectorSize;
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputSampleOffset = n * PaddedCDHW();
						auto channelOffset = 0ull;
						UInt inputIndex, outputIndex;
						VecFloat inputD1, D1;
						for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
						{
							const auto inputSampleOffset = n * Inputs[inputLayer]->PaddedCDHW();
							for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->PaddedC; c += VectorSize)
							{
							    inputIndex = ((c - channelOffset) * HW()) + inputSampleOffset;
								outputIndex = (c * HW()) + outputSampleOffset;
								for (auto w = 0ull; w < strideH; w += VectorSize)
								{
									inputD1.load_a(&Inputs[inputLayer]->NeuronsD1[w + inputIndex]);
									D1.load_a(&NeuronsD1[w + outputIndex]);
									(inputD1 + D1).store_a(&Inputs[inputLayer]->NeuronsD1[w + inputIndex]);
								}
							}
							channelOffset += Inputs[inputLayer]->PaddedC;
						}
					});
				}
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputSampleOffset = n * CDHW();
						auto channelOffset = 0ull;
						UInt inputIndex, outputIndex;
						for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
						{
							const auto inputSampleOffset = n * Inputs[inputLayer]->CDHW();
							for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->C; c++)
							{
								inputIndex = ((c - channelOffset) * HW()) + inputSampleOffset;
								outputIndex = (c * HW()) + outputSampleOffset;
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
									Inputs[inputLayer]->NeuronsD1[inputIndex + hw] += NeuronsD1[outputIndex + hw];
							}
							channelOffset += Inputs[inputLayer]->C;
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