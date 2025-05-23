#pragma once
#include "Layer.h"

namespace dnn
{
	class Concat final : public Layer
	{
	private:
		std::unique_ptr<dnnl::concat::primitive_desc> fwdDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::vector<dnnl::memory::desc> srcMemsDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::concat> fwd;
#endif

		auto InputChannels(const std::vector<Layer*>& inputs) const
		{
			auto channels = 0ull;
			for (const auto& layer : inputs)
				channels += layer->C;

			return channels;
		}

	public:
		FloatArray OutputNeurons;
		
		Concat(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Concat, 0, 0, InputChannels(inputs), inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			OutputNeurons(FloatArray())
		{
			assert(Inputs.size() > 1);

			FwdZeroGradient = Float(1);
			FwdInferenceWeight = Float(10);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
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

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			if constexpr (TestConcat)
				OutputNeurons.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
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

			for (auto i = 1ull; i < Inputs.size(); i++)
			{
				assert(ChosenFormat == GetMemoryFormat(*Inputs[i]->DstMemDesc));
				if (ChosenFormat != GetMemoryFormat(*Inputs[i]->DstMemDesc))
					throw std::invalid_argument("Incompatible memory formats in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Inputs[i]->Name);
			}

			srcMemsDesc = std::vector<dnnl::memory::desc>();
			for (auto i = 0ull; i < Inputs.size(); i++)
			{
				if (GetMemoryNDims(*Inputs[i]->DstMemDesc) == 2)
					srcMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C) }), dnnl::memory::data_type::f32, ChosenFormat));
				else
					srcMemsDesc.push_back(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(Inputs[i]->C), dnnl::memory::dim(Inputs[i]->H), dnnl::memory::dim(Inputs[i]->W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::concat::primitive_desc>(dnnl::concat::primitive_desc(Device.engine, *DstMemDesc, 1, srcMemsDesc));

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			for (auto i = 0ull; i < Inputs.size(); i++)
				fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcMemsDesc[i], Device.engine, Inputs[i]->Neurons.data())});

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::concat>(dnnl::concat(*fwdDesc));
#endif
		}
		

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::concat(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);		
#endif // DNN_LEAN
		}

		/*
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
				if constexpr (!Reference && !ReferenceConcat)
				{
					const auto plain = IsPlainFormat();
					const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (!plain)
						{
							auto channelOffset = 0ull;
							for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
							{
								for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->C; c++)
									for (auto h = 0ull; h < H; h++)
										for (auto w = 0ull; w < W; w++)
										{
											Neurons[OffsetPaddedMem(0, c, h, w)] = Inputs[inputLayer]->Neurons[Inputs[inputLayer]->OffsetPaddedMem(0, c - channelOffset, h, w)];
#ifndef DNN_LEAN
											NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif
										}

								channelOffset += Inputs[inputLayer]->C;
							}

							for (auto c = channelOffset; c < DivUp(channelOffset); c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
									{
										Neurons[OffsetPaddedMem(0, c, h, w)] = Float(0);
#ifndef DNN_LEAN
										NeuronsD1[OffsetPaddedMem(0, c, h, w)] = Float(0);
#endif
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
								auto channelOffset = 0ull;
								for (auto inputLayer = 0ull; inputLayer < Inputs.size(); inputLayer++)
								{
									for (auto c = channelOffset; c < channelOffset + Inputs[inputLayer]->C; c++)
										for (auto h = 0ull; h < H; h++)
											for (auto w = 0ull; w < W; w++)
											{
												Neurons[OffsetPaddedMem(n, c, h, w)] = Inputs[inputLayer]->Neurons[Inputs[inputLayer]->OffsetPaddedMem(n, c - channelOffset, h, w)];
#ifndef DNN_LEAN
												NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif
											}

									channelOffset += Inputs[inputLayer]->C;
								}

								for (auto c = channelOffset; c < DivUp(channelOffset); c++)
									for (auto h = 0ull; h < H; h++)
										for (auto w = 0ull; w < W; w++)
										{
											Neurons[OffsetPaddedMem(n, c, h, w)] = Float(0);
#ifndef DNN_LEAN
											NeuronsD1[OffsetPaddedMem(n, c, h, w)] = Float(0);
#endif
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
					if constexpr (TestConcat)
					{
						fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, OutputNeurons.data()) } };
						for (auto i = 0ull; i < Inputs.size(); i++)
							fwdArgs.insert({ DNNL_ARG_MULTIPLE_SRC + int(i), dnnl::memory(srcMemsDesc[i], Device.engine, Inputs[i]->Neurons.data()) });

						PRAGMA_OMP_SIMD()
						for (auto i = 0ull; i < OutputNeurons.size(); i++)
							OutputNeurons[i] = Float(0);

#ifdef DNN_CACHE_PRIMITIVES
						fwd->execute(Device.stream, fwdArgs);
#else
						dnnl::concat(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
						Device.stream.wait();


						const auto margin = Float(0.0001);

						for (auto i = 0ull; i < Neurons.size(); i++)
						{
							if (((OutputNeurons[i] - margin) > Neurons[i]) || ((OutputNeurons[i] + margin) < Neurons[i]))
							{
								cimg_library::cimg::dialog("Concat Sanity Check", (std::string("Forward Check not passed: ") + Name).c_str(), "OK");
								break;
							}

							if (NeuronsD1[i] != Float(0))
							{
								cimg_library::cimg::dialog("Concat Sanity Check", (std::string("Forward Check D1 not passed: ") + Name).c_str(), "OK");
								break;
							}
						}
					}
				}
				else
				{
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::concat(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();
#ifndef DNN_LEAN
					InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
#endif
				}
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
		*/


		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					auto channelOffset = 0ull;
					for (auto inputLayer = 0ull; inputLayer < InputsBwd.size(); inputLayer++)
					{
						for (auto c = channelOffset; c < channelOffset + InputsBwd[inputLayer]->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputsBwd[inputLayer]->NeuronsD1[InputsBwd[inputLayer]->OffsetPaddedMem(0, c - channelOffset, h, w)] += NeuronsD1[OffsetPaddedMem(0, c, h, w)];

						channelOffset += InputsBwd[inputLayer]->C;
					}
				}
				else
				{
					auto channelOffset = 0ull;
					UInt inputIndex, outputIndex;
					for (auto inputLayer = 0ull; inputLayer < InputsBwd.size(); inputLayer++)
					{
						for (auto c = channelOffset; c < channelOffset + InputsBwd[inputLayer]->C; c++)
						{
							inputIndex = ((c - channelOffset) * HW());
							outputIndex = (c * HW());
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputsBwd[inputLayer]->NeuronsD1[inputIndex + hw] += NeuronsD1[outputIndex + hw];
						}
						channelOffset += InputsBwd[inputLayer]->C;
					}
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						auto channelOffset = 0ull;
						for (auto inputLayer = 0ull; inputLayer < InputsBwd.size(); inputLayer++)
						{
							for (auto c = channelOffset; c < channelOffset + InputsBwd[inputLayer]->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputsBwd[inputLayer]->NeuronsD1[InputsBwd[inputLayer]->OffsetPaddedMem(n, c - channelOffset, h, w)] += NeuronsD1[OffsetPaddedMem(n, c, h, w)];

							channelOffset += InputsBwd[inputLayer]->C;
						}
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputSampleOffset = n * CDHW();
						auto channelOffset = 0ull;
						UInt inputIndex, outputIndex;
						for (auto inputLayer = 0ull; inputLayer < InputsBwd.size(); inputLayer++)
						{
							const auto inputSampleOffset = n * InputsBwd[inputLayer]->CDHW();
							for (auto c = channelOffset; c < channelOffset + InputsBwd[inputLayer]->C; c++)
							{
								inputIndex = ((c - channelOffset) * HW()) + inputSampleOffset;
								outputIndex = (c * HW()) + outputSampleOffset;
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
									InputsBwd[inputLayer]->NeuronsD1[inputIndex + hw] += NeuronsD1[outputIndex + hw];
							}
							channelOffset += InputsBwd[inputLayer]->C;
						}
					});
#ifdef DNN_STOCHASTIC
			}
#endif

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}


	
		/*
		void BackwardProp(const UInt batchSize)
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif



#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
*/
	};
}