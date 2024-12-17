#pragma once
#include "Layer.h"

namespace dnn
{
	class Average final : public Layer
	{
	private:
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif
		std::vector<Float> scales;
		FloatVector scale;

	public:
		const Byte first, second;
		FloatVector SurvivalProbability;

		Average(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Average, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs)),
			SurvivalProbability(FloatVector(2, Float(1))),
			scales(std::vector<Float>(2, Float(1))),
			scale(FloatVector(1, Float(0.5)))
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->D == Inputs[1]->D);

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
					ChosenFormat = GetMemoryFormat(*Inputs[first]->DstMemDesc);
					if (ChosenFormat != GetMemoryFormat(*Inputs[first]->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			
			dnnl::primitive_attr attr;
			attr.set_scales_mask(DNNL_ARG_SRC_0, 0);
			attr.set_scales_mask(DNNL_ARG_SRC_1, 0);

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc, attr));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

		    auto ScaleMem = dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale.data());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, ScaleMem }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, ScaleMem } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto fullDepth = SurvivalProbability[0] == Float(1) && SurvivalProbability[1] == Float(1);
			scales[0] = fullDepth ? Float(1) : (Inputs[0]->Skip ? Float(0) : Float(1));
			scales[1] = fullDepth ? Float(1) : (Inputs[1]->Skip ? Float(0) : Float(1));

			if (training)
			{
				if (Reference && fullDepth)
				{
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();
#ifndef DNN_LEAN
					InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#endif // DNN_LEAN
				}
				else
				{
					const auto plain = IsPlainFormat();
					const auto size = plain ? CDHW() : PaddedCDHW();
					const auto part = GetVectorPart(size);
					const auto threads = GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);
					const auto strideHW = HW() * VectorSize;

					if (plain)
					{
						if (EqualDimensions(Inputs))
						{
							if (fullDepth)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * CDHW();
									const auto end = start + CDHW();
									PRAGMA_OMP_SIMD()
									for (auto cdhw = start; cdhw < end; cdhw++)
									{
										Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw]) / Float(2);
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = Float(0);
#endif
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * CDHW();
									const auto end = start + CDHW();
									const auto skip0 = Inputs[0]->Skip;
									const auto skip1 = Inputs[1]->Skip;
									const auto div = (!skip0 && !skip1) ? Float(2) : Float(1);
									PRAGMA_OMP_SIMD()
									for (auto cdhw = start; cdhw < end; cdhw++)
									{
										Neurons[cdhw] = ((skip0 ? Float(0) : Inputs[0]->Neurons[cdhw]) + (skip1 ? Float(0) : Inputs[1]->Neurons[cdhw])) / div;
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = Float(0);
#endif
									}
								});
						}
						else
						{
							if (fullDepth)
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto outputOffset = n * CDHW() + c * HW();
										const auto channelOffset = n * C + c;
										PRAGMA_OMP_SIMD()
										for (auto hw = 0ull; hw < HW(); hw++)
										{
											Neurons[hw + outputOffset] = (Inputs[first]->Neurons[hw + outputOffset] + Inputs[second]->Neurons[channelOffset]) / Float(2);
#ifndef DNN_LEAN
											NeuronsD1[hw + outputOffset] = Float(0);
#endif
										}
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto skipFirst = Inputs[first]->Skip;
									const auto skipSecond = Inputs[second]->Skip;
									const auto div = (!skipFirst && !skipSecond) ? Float(2) : Float(1);

									for (auto c = 0ull; c < C; c++)
									{
										const auto outputOffset = n * CDHW() + c * HW();
										const auto channelOffset = n * C + c;
										PRAGMA_OMP_SIMD()
										for (auto hw = 0ull; hw < HW(); hw++)
										{
											Neurons[hw + outputOffset] = ((skipFirst ? Float(0) : Inputs[first]->Neurons[hw + outputOffset]) + (skipSecond ? Float(0) : Inputs[second]->Neurons[channelOffset])) / div;
#ifndef DNN_LEAN
											NeuronsD1[hw + outputOffset] = Float(0);
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
							if (fullDepth)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * size;
									for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
									{
										((VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) + VecFloat().load_a(&Inputs[1]->Neurons[cdhw])) / Float(2)).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
										VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
									}
									for (auto cdhw = start + part; cdhw < start + size; cdhw++)
									{
										Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw]) / Float(2);
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = Float(0);
#endif
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * size;
									const auto skip0 = Inputs[0]->Skip;
									const auto skip1 = Inputs[1]->Skip;
									const auto div = (!skip0 && !skip1) ? Float(2) : Float(1);
									VecFloat In0, In1;
									for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
									{
										In0 = skip0 ? VecFloat(0) : VecFloat().load_a(&Inputs[0]->Neurons[cdhw]);
										In1 = skip1 ? VecFloat(0) : VecFloat().load_a(&Inputs[1]->Neurons[cdhw]);
										((In0 + In1) / div).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
										VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
									}
									for (auto cdhw = start + part; cdhw < start + size; cdhw++)
									{
										Neurons[cdhw] = ((skip0 ? Float(0) : Inputs[0]->Neurons[cdhw]) + (skip1 ? Float(0) : Inputs[1]->Neurons[cdhw])) / div;
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = Float(0);
#endif
									}
								});
						}
						else
						{
							if (fullDepth)
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = n * PaddedCDHW() + c * HW();
										const auto channelOffset = n * PaddedC + c;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											((VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) + VecFloat().load_a(&Inputs[second]->Neurons[channelOffset])) / Float(2)).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
										}
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto skipFirst = Inputs[first]->Skip;
									const auto skipSecond = Inputs[second]->Skip;
									const auto div = (!skipFirst && !skipSecond) ? VecFloat(2) : VecFloat(1);
									VecFloat In0, In1;
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = n * PaddedCDHW() + c * HW();
										const auto channelOffset = n * PaddedC + c;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											In0 = skipFirst ? VecFloat(0) : VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]);
											In1 = skipSecond ? VecFloat(0) : VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]);
											((In0 + In1) / div).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
#endif
										}
									}
								});
						}
					}
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
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto plain = IsPlainFormat();
			const auto size = GetElementsCount();
			const auto part = GetVectorPart(size);

			const auto fullDepth = SurvivalProbability[0] == Float(1) && SurvivalProbability[1] == Float(1);
			scales[0] = fullDepth ? Float(1) : (Inputs[0]->Skip ? Float(0) : Float(1));
			scales[1] = fullDepth ? Float(1) : (Inputs[1]->Skip ? Float(0) : Float(1));

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				const auto scales0 = scales[0];
				const auto scales1 = scales[1];

				if (EqualDimensions(InputsBwd))
				{
					if (plain)
					{
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
						{
							InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales0 * Float(0.5);
							InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales1 * Float(0.5);
						}
					}
					else
					{
						VecFloat inputD1, D1;
						for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);
							D1 *= Float(0.5);
							inputD1.load_a(&InputsBwd[0]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales0;
							inputD1.store_a(&InputsBwd[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&InputsBwd[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales1;
							inputD1.store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = part; cdhw < size; cdhw++)
						{
							InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales0 * Float(0.5);
							InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales1 * Float(0.5);
						}
					}
				}
				else
				{
					if (plain)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto outputOffset = c * HW();
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
							{
								InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * Float(0.5);;
								InputsBwd[second]->NeuronsD1[c] += NeuronsD1[hw + outputOffset] * Float(0.5);;
							}
						}
					}
					else
					{
						const auto strideHW = HW() * VectorSize;
						VecFloat D1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto outputOffset = c * HW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								D1.load_a(&NeuronsD1[hw + outputOffset]);
								D1 *= Float(0.5);
								(D1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset])).store_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]);
								(D1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[c])).store_a(&InputsBwd[second]->NeuronsD1[c]);
							}
						}
					}
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (EqualDimensions(Inputs))
				{
					if (plain)
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * size;
								const auto end = start + size;
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Float(0.5);
									InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Float(0.5);
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * size;
								const auto end = start + size;
								const auto scale0 = scales[0];
								const auto scale1 = scales[1];
								PRAGMA_OMP_SIMD()
								for (auto cdhw = start; cdhw < end; cdhw++)
								{
									InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale0;
									InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale1;
								}
							});
					}
					else
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * size;

								VecFloat D1;
								for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
								{
									D1.load_a(&NeuronsD1[cdhw]);
									D1 *= Float(0.5);
									(VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw]) + D1).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
									(VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw]) + D1).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
								}
								for (auto cdhw = start + part; cdhw < start + size; cdhw++)
								{
									InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Float(0.5);
									InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * Float(0.5);
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * size;
								const auto scale0 = scales[0];
								const auto scale1 = scales[1];

								VecFloat D1;
								for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
								{
									D1.load_a(&NeuronsD1[cdhw]);
									mul_add(D1, scale0, VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw])).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
									mul_add(D1, scale1, VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw])).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
								}
								for (auto cdhw = start + part; cdhw < start + size; cdhw++)
								{
									InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale0;
									InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scale1;
								}
							});
					}
				}
				else
				{
					if (plain)
					{
						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = n * CDHW() + c * HW();
									const auto channelOffset = n * C + c;
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * Float(0.5);
										InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset] * Float(0.5);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto scale0 = scales[first];
								const auto scale1 = scales[second];
								for (auto c = 0ull; c < C; c++)
								{
									const auto outputOffset = n * CDHW() + c * HW();
									const auto channelOffset = n * C + c;
									PRAGMA_OMP_SIMD()
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * scale0;
										InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset] * scale1;
									}
								}
							});
					}
					else
					{
						const auto strideHW = HW() * VectorSize;

						if (fullDepth)
							for_i(batchSize, threads, [=](UInt n)
							{
								VecFloat D1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = n * PaddedCDHW() + c * HW();
									const auto channelOffset = n * PaddedC + c;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										D1.load_a(&NeuronsD1[hw + outputOffset]);
										D1 *= Float(0.5);
										(D1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset])).store_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]);
										(D1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto scale0 = scales[first];
								const auto scale1 = scales[second];
								VecFloat D1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = n * PaddedCDHW() + c * HW();
									const auto channelOffset = n * PaddedC + c;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										D1.load_a(&NeuronsD1[hw + outputOffset]);
										mul_add(D1, scale0, VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset])).store_a(&InputsBwd[first]->NeuronsD1[hw + outputOffset]);
										mul_add(D1, scale1, VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
									}
								}
							});
					}
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