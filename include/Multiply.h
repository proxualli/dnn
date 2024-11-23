#pragma once
#include "Layer.h"

namespace dnn
{
	class Multiply final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif
		
	public:
		const Byte first, second;
		FloatArray InputNeurons;

		Multiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Multiply, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs)),
			InputNeurons(FloatArray())
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C >= Inputs[1]->C);
			assert(Inputs[0]->D >= Inputs[1]->D);
			assert(Inputs[0]->H >= Inputs[1]->H);
			assert(Inputs[0]->W >= Inputs[1]->W);

			FwdInferenceWeight = Float(5);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
		}

		void UpdateResolution() final override
		{
			H = Inputs[first]->H;
			W = Inputs[first]->W;
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader();
		}

		UInt FanIn() const final override
		{
			return 1;
		}

		UInt FanOut() const  final override
		{
			return 1;
		}

		void InitializeDescriptorsFwd(const UInt batchSize)  final override
		{
			if (GetMemoryNDims(*Inputs[first]->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (NeuronsFormat == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetMemoryFormat(*Inputs[first]->DstMemDesc);
					if (ChosenFormat != GetMemoryFormat(*Inputs[first]->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_mul, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void InitializeDescriptorsBwd(const UInt batchSize) final override
		{
		}

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			if constexpr (TestMultiply)
				InputNeurons.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
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
				dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
#else
				if constexpr (!Reference && !ReferenceMultiply)
				{
					const auto plain = IsPlainFormat();
					const auto strideHW = HW() * VectorSize;

#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (!plain)
						{
							if (EqualChannels(Inputs))
							{
								if (EqualDimensions(Inputs))
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = c * HW();
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											(VecFloat().load_a(&Inputs[0]->Neurons[hw + outputOffset]) * VecFloat().load_a(&Inputs[1]->Neurons[hw + outputOffset])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
										}
									}
								}
								else
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = c * HW();
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * VecFloat().load_a(&Inputs[second]->Neurons[c])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
										}
									}
								}
							}
							else
							{
								if (EqualDimensions(Inputs))
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = c * HW();
										auto offset = 0ull;;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * VecFloat().load_a(&Inputs[second]->Neurons[offset++])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
										}
									}
								}
							}
						}
						else
						{
							if (EqualChannels(Inputs))
							{
								if (EqualDimensions(Inputs))
								{
									PRAGMA_OMP_SIMD()
									for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
									{
										Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = 0;
#endif
									}
								}
								else
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto outputOffset = c * HW();
										PRAGMA_OMP_SIMD()
											for (auto hw = 0ull; hw < HW(); hw++)
											{
												Neurons[hw + outputOffset] = Inputs[first]->Neurons[hw + outputOffset] * Inputs[second]->Neurons[c];
#ifndef DNN_LEAN
												NeuronsD1[hw + outputOffset] = 0;
#endif
											}
									}
								}
							}
							else
							{
								if (EqualDimensions(Inputs))
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto outputOffset = c * HW();
										auto offset = 0ull;
										PRAGMA_OMP_SIMD()
										for (auto hw = 0ull; hw < HW(); hw++)
										{
											Neurons[hw + outputOffset] = Inputs[first]->Neurons[hw + outputOffset] * Inputs[second]->Neurons[offset++];
#ifndef DNN_LEAN
											NeuronsD1[hw + outputOffset] = 0;
#endif
										}
									}
								}
							}
						}
					}
					else
					{
#endif
						const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

						if (!plain)
						{
							if (EqualChannels(Inputs))
							{
								if (EqualDimensions(Inputs))
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = n * PaddedCDHW() + c * HW();

											for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[0]->Neurons[hw + outputOffset]) * VecFloat().load_a(&Inputs[1]->Neurons[hw + outputOffset])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
								}
								else
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = n * PaddedCDHW() + c * HW();
											const auto channelOffset = n * PaddedC + c;
											for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * VecFloat().load_a(&Inputs[second]->Neurons[channelOffset])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
								}
							}
							else
							{
								if (EqualDimensions(Inputs))
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto channelOffset = n * Inputs[second]->PaddedCDHW();
										for (auto c = 0ull; c < PaddedC; c += VectorSize)
										{
											const auto outputOffset = n * PaddedCDHW() + c * HW();
											for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * Inputs[second]->Neurons[hw + channelOffset]).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
								}
							}
						}
						else
						{
							if (EqualChannels(Inputs))
							{
								if (EqualDimensions(Inputs))
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto start = n * CDHW();
										const auto end = start + CDHW();
										PRAGMA_OMP_SIMD()
										for (auto cdhw = start; cdhw < end; cdhw++)
										{
											Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
											NeuronsD1[cdhw] = 0;
#endif
										}
									});
								}
								else
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = n * CDHW() + c * HW();
											const auto channelOffset = n * C + c;
											PRAGMA_OMP_SIMD()
											for (auto hw = 0ull; hw < HW(); hw++)
											{
												Neurons[hw + outputOffset] = Inputs[first]->Neurons[hw + outputOffset] * Inputs[second]->Neurons[channelOffset];
#ifndef DNN_LEAN
												NeuronsD1[hw + outputOffset] = 0;
#endif
											}
										}
									});
								}
							}
							else
							{
								if (EqualDimensions(Inputs))
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto channelOffset = n * HW();
										for (auto c = 0ull; c < C; c++)
										{
											const auto outputOffset = n * CDHW() + c * HW();
											for (auto hw = 0ull; hw < HW(); hw++)
											{
												(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * VecFloat().load_a(&Inputs[second]->Neurons[hw + channelOffset])).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
											}
										}
									});
								}
							}
						}
#ifdef DNN_STOCHASTIC
					}
#endif
					if constexpr (TestMultiply)
					{
						for (auto i = 0ull; i < InputNeurons.size(); i++)
							InputNeurons[i] = Float(0);

						//fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsFwd[first]->DstMemDesc, Device.engine, InputsFwd[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*InputsFwd[second]->DstMemDesc, Device.engine, InputsFwd[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, InputNeurons.data()) } };
												

#ifdef DNN_CACHE_PRIMITIVES
						fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsFwd[first]->DstMemDesc, Device.engine, InputsFwd[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*InputsFwd[second]->DstMemDesc, Device.engine, InputsFwd[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, InputNeurons.data()) } });
#else
						dnnl::binary(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsFwd[first]->DstMemDesc, Device.engine, InputsFwd[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*InputsFwd[second]->DstMemDesc, Device.engine, InputsFwd[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, InputNeurons.data()) } });
#endif
						Device.stream.wait();


						const auto margin = Float(0.0001);

						for (auto i = 0ull; i < Neurons.size(); i++)
						{
							const auto nA = InputNeurons[i];
							const auto nB = Neurons[i];
							if ( ((nA - margin) > nB) || ((nA + margin) < nB) )
							{
								cimg_library::cimg::dialog("Multiply Sanity Check", (std::string("Forward Check not passed: ") + Name).c_str(), "OK");
								break;
							}

							if (NeuronsD1[i] != Float(0))
							{
								cimg_library::cimg::dialog("Multiply Sanity Check", (std::string("Forward Check D1 not passed: ") + Name).c_str(), "OK");
								break;
							}
						}

						//fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsFwd[first]->DstMemDesc, Device.engine, InputsFwd[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*InputsFwd[second]->DstMemDesc, Device.engine, InputsFwd[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
					}
				}
				else
				{
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();
#ifndef DNN_LEAN
					InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#endif
				}
#endif
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
#else
				dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
				Device.stream.wait();
			}
		}

		void BackwardProp(const UInt batchSize)  final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
			const auto strideHW = HW() * VectorSize;

			if (EqualChannels(Inputs))
			{
				if (EqualDimensions(Inputs))
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (!plain)
						{
							for (auto cdhw = 0ull; cdhw < PaddedCDHW(); cdhw += VectorSize)
							{
								mul_add(VecFloat().load_a(&InputsFwd[1]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(VecFloat().load_a(&InputsFwd[0]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
							}
						}
						else
						{
							PRAGMA_OMP_SIMD()
							for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
							{
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[1]->Neurons[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[0]->Neurons[cdhw];
							}
						}
					}
					else
					{
#endif
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = n * PaddedCDHW() + c * HW();
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
										mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[second]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
										mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[second]->NeuronsD1[hw + outputOffset]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * CDHW();
								const auto end = start + CDHW();
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									Inputs[first]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[second]->Neurons[cdhw];
									Inputs[second]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[first]->Neurons[cdhw];
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
							VecFloat neuronsD1;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto outputOffset = c * HW();
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
									mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[second]->Neurons[c]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
									mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[c])).store_a(&Inputs[second]->NeuronsD1[c]);
								}
							}
						}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto outputOffset = c * HW();
								PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[second]->Neurons[c];
										Inputs[second]->NeuronsD1[c] += NeuronsD1[hw + outputOffset] * InputsFwd[first]->Neurons[hw + outputOffset];
									}
							}
						}
					}
					else
					{
#endif
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = n * PaddedCDHW() + c * HW();
									const auto channelOffset = n * PaddedC + c;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
										mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[second]->Neurons[channelOffset]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
										mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[channelOffset])).store_a(&Inputs[second]->NeuronsD1[channelOffset]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = n * CDHW() + c * HW();
									const auto channelOffset = n * C + c;
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[second]->Neurons[channelOffset];
										Inputs[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[first]->Neurons[hw + outputOffset];
									}
								}
							});
#ifdef DNN_STOCHASTIC
					}
#endif
				}
			}
			else
			{
				if (EqualDimensions(Inputs))
				{
					if (!plain)
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto channelOffset = n * Inputs[second]->PaddedCDHW();
							VecFloat neuronsD1;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto outputOffset = n * PaddedCDHW() + c * HW();
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
									mul_add(neuronsD1, InputsFwd[second]->Neurons[hw + channelOffset], VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
									//mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[first]->Neurons[hw + outputOffset]), Inputs[second]->NeuronsD1[hw + channelOffset]).store_a(&Inputs[second]->NeuronsD1[hw + channelOffset]);
									Inputs[second]->NeuronsD1[hw + channelOffset] += horizontal_add(neuronsD1 * VecFloat().load_a(&InputsFwd[first]->Neurons[hw + outputOffset]));
								}
							}
						});
					else
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto channelOffset = n * HW();
							for (auto c = 0ull; c < C; c++)
							{
								const auto outputOffset = n * CDHW() + c * HW();
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[second]->Neurons[hw + channelOffset];
									Inputs[second]->NeuronsD1[hw + channelOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[first]->Neurons[hw + outputOffset];
								}
							}
						});
				}
			}
#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}