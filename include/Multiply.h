#pragma once
#include "Layer.h"

namespace dnn
{
	class Multiply final : public Layer
	{
	private:
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::reduction::primitive_desc> bwdDescReduction;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdDescA, bwdDescB;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
		std::unique_ptr<dnnl::reduction> bwdReduction;
		std::unique_ptr<dnnl::binary> bwdA, bwdB;
#endif
		
	public:
		const Byte first, second;
		FloatVector SurvivalProbability;

		Multiply(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Multiply, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs)),
			SurvivalProbability(FloatVector(2, Float(1)))
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C >= Inputs[1]->C);
			assert(Inputs[0]->D >= Inputs[1]->D);
			assert(Inputs[0]->H >= Inputs[1]->H);
			assert(Inputs[0]->W >= Inputs[1]->W);
			//scales = std::vector<Float>(2, Float(1));
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

		void InitializeDescriptors(const UInt batchSize)  final override
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
			

			bwdDescReduction = std::make_unique<dnnl::reduction::primitive_desc>(dnnl::reduction::primitive_desc(Device.engine, dnnl::algorithm::reduction_sum, *DiffDstMemDesc, *Inputs[second]->DiffDstMemDesc, 0.f, 0.f));
#ifdef DNN_CACHE_PRIMITIVES
			bwdReduction = std::make_unique<dnnl::reduction>(dnnl::reduction(*bwdDescReduction));
#endif

			dnnl::post_ops binary_ops;
			binary_ops.append_sum();
			dnnl::primitive_attr binary_attr;
			binary_attr.set_post_ops(binary_ops);
			
			bwdDescA = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_mul, *DiffDstMemDesc, *Inputs[second]->DstMemDesc, *Inputs[first]->DiffDstMemDesc, binary_attr));
			bwdDescB = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_mul, *DiffDstMemDesc, *Inputs[first]->DstMemDesc, *Inputs[first]->DiffDstMemDesc));
			
#ifdef DNN_CACHE_PRIMITIVES
			bwdA = std::make_unique<dnnl::binary>(dnnl::binary(*bwdDescA));
			bwdB = std::make_unique<dnnl::binary>(dnnl::binary(*bwdDescB));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			//const auto fullDepth = true; // SurvivalProbability[0] == Float(1) && SurvivalProbability[1] == Float(1);
			
			if (training)
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

				/*const auto plain = IsPlainFormat();
				const auto size = plain ? CDHW() : PaddedCDHW();
				const auto part = GetVectorPart(size);
				const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * size, Float(10));

				const auto strideHW = HW() * VectorSize;

#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						if (EqualDimensions(Inputs))
						{
							if (fullDepth)
							{
								for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
								{
									(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) * VecFloat().load_a(&Inputs[1]->Neurons[cdhw])).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
								}
								for (auto cdhw = part; cdhw < size; cdhw++)
								{
									Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = 0;
#endif
								}
							}
							else
							{
								VecFloat In0, In1;
								for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
								{
									In0 = Inputs[0]->Skip ? VecFloat(1) : VecFloat().load_a(&Inputs[0]->Neurons[cdhw]);
									In1 = Inputs[1]->Skip ? VecFloat(1) : VecFloat().load_a(&Inputs[1]->Neurons[cdhw]);
									(In0 * In1).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
									VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
								}
								for (auto cdhw = part; cdhw < size; cdhw++)
								{
									Neurons[cdhw] = (Inputs[0]->Skip ? Float(1) : Inputs[0]->Neurons[cdhw]) * (Inputs[1]->Skip ? Float(1) : Inputs[1]->Neurons[cdhw]);
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = 0;
#endif
								}
							}
						}
						else
						{
							if (fullDepth)
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
							else
							{
								VecFloat In0, In1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = c * HW();
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										In0 = Inputs[first]->Skip ? VecFloat(1) : VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]);
										In1 = Inputs[second]->Skip ? VecFloat(1) : VecFloat().load_a(&Inputs[second]->Neurons[c]);
										(In0 * In1).store_a(&Neurons[hw + outputOffset]);
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
						if (EqualDimensions(Inputs))
						{
							if (fullDepth)
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
								PRAGMA_OMP_SIMD()
								for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
								{
									Neurons[cdhw] = (Inputs[0]->Skip ? Float(1) : Inputs[0]->Neurons[cdhw]) * (Inputs[1]->Skip ? Float(1) : Inputs[1]->Neurons[cdhw]);
#ifndef DNN_LEAN
									NeuronsD1[cdhw] = 0;
#endif
								}
							}
						}
						else
						{
							if (fullDepth)
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
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = c * HW();
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										Neurons[hw + outputOffset] = ((Inputs[first]->Skip ? Float(1) : Inputs[first]->Neurons[hw + outputOffset]) * (Inputs[second]->Skip ? Float(1) : Inputs[second]->Neurons[c]));
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
					if (!plain)
					{
						if (EqualChannels(Inputs))
						{
							if (EqualDimensions(Inputs))
							{
								if (fullDepth)
								{
									for_i(batchSize, threads, [=](UInt n)
										{
											const auto start = n * size;
											for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
											{
												(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) * VecFloat().load_a(&Inputs[1]->Neurons[cdhw])).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
											}
											for (auto cdhw = start + part; cdhw < start + size; cdhw++)
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
											const auto skip0 = Inputs[0]->Skip;
											const auto skip1 = Inputs[1]->Skip;
											const auto start = n * size;
											VecFloat In0, In1;
											for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
											{
												In0 = skip0 ? VecFloat(1) : VecFloat().load_a(&Inputs[0]->Neurons[cdhw]);
												In1 = skip1 ? VecFloat(1) : VecFloat().load_a(&Inputs[1]->Neurons[cdhw]);
												(In0 * In1).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
												VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
											}
											for (auto cdhw = start + part; cdhw < start + size; cdhw++)
											{
												Neurons[cdhw] = (skip0 ? Float(1) : Inputs[0]->Neurons[cdhw]) * (skip1 ? Float(1) : Inputs[1]->Neurons[cdhw]);
#ifndef DNN_LEAN
												NeuronsD1[cdhw] = 0;
#endif
											}
										});
								}
							}
							else
							{
								if (fullDepth)
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
								else
								{
									for_i(batchSize, threads, [=](UInt n)
										{
											const auto skipFirst = Inputs[first]->Skip;
											const auto skipSecond = Inputs[second]->Skip;
											VecFloat In0, In1;
											for (auto c = 0ull; c < PaddedC; c += VectorSize)
											{
												const auto outputOffset = n * PaddedCDHW() + c * HW();
												const auto channelOffset = n * PaddedC + c;
												for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
												{
													In0 = skipFirst ? VecFloat(1) : VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]);
													In1 = skipSecond ? VecFloat(1) : VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]);
													(In0 * In1).store_a(&Neurons[hw + outputOffset]);
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
							if (EqualDimensions(Inputs))
							{
								if (fullDepth)
								{
									for_i(batchSize, threads, [=](UInt n)
									{
										const auto start = n * size;
										const auto startSecond = n * HW();
										auto offset = startSecond;
										for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
										{ 
											(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) * VecFloat().load_a(&Inputs[1]->Neurons[offset])).store_a(&Neurons[cdhw]);
											offset = (cdhw % HW() != 0ull) ? (offset + VectorSize) : startSecond;
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
										}
										for (auto cdhw = start + part; cdhw < start + size; cdhw++)
										{
											Neurons[cdhw] = Inputs[0]->Neurons[cdhw] * Inputs[1]->Neurons[offset];
											offset = (cdhw % HW() != 0ull) ? (offset + 1) : startSecond;
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
										const auto skip0 = Inputs[0]->Skip;
										const auto skip1 = Inputs[1]->Skip;
										const auto start = n * size;
										VecFloat In0, In1;
										for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
										{
											In0 = skip0 ? VecFloat(1) : VecFloat().load_a(&Inputs[0]->Neurons[cdhw]);
											In1 = skip1 ? VecFloat(1) : VecFloat().load_a(&Inputs[1]->Neurons[cdhw]);
											(In0 * In1).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
										}
										for (auto cdhw = start + part; cdhw < start + size; cdhw++)
										{
											Neurons[cdhw] = (skip0 ? Float(1) : Inputs[0]->Neurons[cdhw]) * (skip1 ? Float(1) : Inputs[1]->Neurons[cdhw]);
#ifndef DNN_LEAN
											NeuronsD1[cdhw] = 0;
#endif
										}
									});
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
								if (fullDepth)
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
											const auto skip0 = Inputs[0]->Skip;
											const auto skip1 = Inputs[1]->Skip;
											const auto start = n * CDHW();
											const auto end = start + CDHW();
											PRAGMA_OMP_SIMD()
												for (auto cdhw = start; cdhw < end; cdhw++)
												{
													Neurons[cdhw] = (skip0 ? Float(1) : Inputs[0]->Neurons[cdhw]) * (skip1 ? Float(1) : Inputs[1]->Neurons[cdhw]);
#ifndef DNN_LEAN
													NeuronsD1[cdhw] = 0;
#endif
												}
										});
								}
							}
							else
							{
								if (fullDepth)
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
								else
								{
									for_i(batchSize, threads, [=](UInt n)
										{
											const auto skipFirst = Inputs[first]->Skip;
											const auto skipSecond = Inputs[second]->Skip;
											for (auto c = 0ull; c < C; c++)
											{
												const auto outputOffset = n * CDHW() + c * HW();
												const auto channelOffset = n * C + c;
												PRAGMA_OMP_SIMD()
													for (auto hw = 0ull; hw < HW(); hw++)
													{
														Neurons[hw + outputOffset] = (skipFirst ? Float(1) : Inputs[first]->Neurons[hw + outputOffset]) * (skipSecond ? Float(1) : Inputs[second]->Neurons[channelOffset]);
#ifndef DNN_LEAN
														NeuronsD1[hw + outputOffset] = 0;
#endif
													}
											}
										});
								}
							}
						}
						else
						{
						}
					}
#ifdef DNN_STOCHASTIC
				}
#endif
*/
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

			if (EqualChannels(Inputs))
			{
				const auto plain = IsPlainFormat();
				const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
				const auto threads = batchSize == 1 ? 1ull : GetThreads(elements, Float(10));

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
						{
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * PaddedCDHW();
								const auto end = start + PaddedCDHW();
								for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
								{
									mul_add(VecFloat().load_a(&InputsFwd[1]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
									mul_add(VecFloat().load_a(&InputsFwd[0]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
								}
							});
						}
						else
						{
							for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * CDHW();
									const auto end = start + CDHW();
									PRAGMA_OMP_SIMD()
									for (auto cdhw = start; cdhw < end; cdhw++)
									{
										Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[1]->Neurons[cdhw];
										Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[0]->Neurons[cdhw];
									}
								});
						}
#ifdef DNN_STOCHASTIC
					}
#endif
				}
				else
				{
					const auto strideHW = HW() * VectorSize;

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
						{
							for_i(batchSize, threads, [=](UInt n)
								{
									VecFloat neuronsD1;
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = n * PaddedCDHW() + c * HW();
										const auto channelOffset = n * (Inputs[second]->C == 1 ? HW() : PaddedC) + c;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
											mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[second]->Neurons[channelOffset]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
											mul_add(neuronsD1, VecFloat().load_a(&InputsFwd[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[channelOffset])).store_a(&Inputs[second]->NeuronsD1[channelOffset]);
										}
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
												Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[second]->Neurons[channelOffset];
												Inputs[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset] * InputsFwd[first]->Neurons[hw + outputOffset];
											}
									}
								});
						}
#ifdef DNN_STOCHASTIC
					}
#endif
				}
			}
			else
			{
				auto memDiffDst = dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data());
				auto memDiffDstFirst = dnnl::memory(*Inputs[first]->DiffDstMemDesc, Device.engine, Inputs[first]->NeuronsD1.data());
				auto memDiffDstSecond = dnnl::memory(*Inputs[second]->DiffDstMemDesc, Device.engine, Inputs[second]->NeuronsD1.data());
				auto memDstFirst = dnnl::memory(*InputsFwd[first]->DstMemDesc, Device.engine, InputsFwd[first]->Neurons.data());
				auto memDstSecond = dnnl::memory(*InputsFwd[second]->DstMemDesc, Device.engine, InputsFwd[second]->Neurons.data());

				bwdA->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, memDiffDst }, { DNNL_ARG_SRC_1, memDstSecond }, { DNNL_ARG_DST, memDiffDstFirst }, { DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, memDiffDstFirst }});
				Device.stream.wait();

				if (*DiffDstMemDesc != *Inputs[second]->DiffDstMemDesc)
				{
					auto memDiffDstSecondFull = dnnl::memory(*Inputs[first]->DiffDstMemDesc, Device.engine);
#ifdef DNN_CACHE_PRIMITIVES
					bwdB->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, memDiffDst }, { DNNL_ARG_SRC_1, memDstFirst }, { DNNL_ARG_DST, memDiffDstSecondFull }});
#else
					dnnl::binary(*bwdDescB).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, memDiffDst }, { DNNL_ARG_SRC_1, memDstFirst }, { DNNL_ARG_DST, memDiffDstSecondFull }});
#endif
					Device.stream.wait();
#ifdef DNN_CACHE_PRIMITIVES
					bwdReduction->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC, memDiffDstSecondFull }, { DNNL_ARG_DST, memDiffDstSecond }});
#else
					dnnl::reduction(*bwdDescReduction).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC, memDiffDstSecondFull }, { DNNL_ARG_DST, memDiffDstSecond }});
#endif
					Device.stream.wait();
				}
				else
				{
#ifdef DNN_CACHE_PRIMITIVES
					bwdB->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, memDiffDst }, { DNNL_ARG_SRC_1, memDstFirst }, { DNNL_ARG_DST, memDiffDstSecond }});
#else
					dnnl::binary(*bwdDescB).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, memDiffDst }, { DNNL_ARG_SRC_1, memDstFirst }, { DNNL_ARG_DST, memDiffDstSecond }});
#endif

					Device.stream.wait();
				}
			}
#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}