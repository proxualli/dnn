#pragma once
#include "Layer.h"

namespace dnn
{
	class Add final : public Layer
	{
	private:
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
		//std::unique_ptr<dnnl::binary::primitive_desc> bwd0Desc;
		//std::unique_ptr<dnnl::binary::primitive_desc> bwd1Desc;
		//std::unique_ptr<dnnl::reduction::primitive_desc> bwdReductionDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
		//std::unique_ptr<dnnl::binary> bwd0;
		//std::unique_ptr<dnnl::binary> bwd1;
		//std::unique_ptr<dnnl::reduction> bwdReduction;
#endif
		std::vector<Float> scales;
		FloatVector scale0;
		FloatVector scale1;
		
	public:
		const Byte first, second;
		FloatVector SurvivalProbability;

		Add(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Add, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs)),
			SurvivalProbability(FloatVector(2, Float(1))),
			scale0(FloatVector(1, Float(1))),
			scale1(FloatVector(1, Float(1)))
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->D == Inputs[1]->D);
			
			scales = std::vector<Float>(2, Float(1));

			FwdZeroGradient = Float(1);
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

			
			dnnl::primitive_attr attr;
			attr.set_scales_mask(DNNL_ARG_SRC_0, 0);
			attr.set_scales_mask(DNNL_ARG_SRC_1, 0);
			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc, attr));
			

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc));
			/*
			bwd0Desc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[first]->DiffDstMemDesc, *DiffDstMemDesc, *Inputs[first]->DiffDstMemDesc, attr));
			if (!EqualDimensions(Inputs))
			{
				//dnnl::post_ops ops;
				//ops.append_eltwise(dnnl::algorithm::eltwise_linear, scale0[0], scale1[0]);
				//ops.append_binary(dnnl::algorithm::binary_add, *Inputs[second]->DiffDstMemDesc);
				//dnnl::primitive_attr binary_attr;
				//binary_attr.set_post_ops(ops);
				
				bwdReductionDesc = std::make_unique<dnnl::reduction::primitive_desc>(dnnl::reduction::primitive_desc(Device.engine, dnnl::algorithm::reduction_sum, *DiffDstMemDesc, *Inputs[second]->DiffDstMemDesc, Float(0), Float(0)));
				bwd1Desc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[second]->DiffDstMemDesc, *Inputs[second]->DiffDstMemDesc, *Inputs[second]->DiffDstMemDesc, attr));
			}
			else
				bwd1Desc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *Inputs[second]->DiffDstMemDesc, *DiffDstMemDesc, *Inputs[second]->DiffDstMemDesc, attr));
			*/

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } };
			//fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
			/*
			bwd0 = std::make_unique<dnnl::binary>(dnnl::binary(*bwd0Desc));
			if (!EqualDimensions(Inputs))
				bwdReduction = std::make_unique<dnnl::reduction>(dnnl::reduction(*bwdReductionDesc));
			bwd1 = std::make_unique<dnnl::binary>(dnnl::binary(*bwd1Desc));
			*/
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto fullDepth = !training || (SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1));
			scale0[0] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scale1[0] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);

			#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, fwdArgs);
			#else
				dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
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
			const auto fullDepth = !training || (SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1));
			scales[first] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scales[second] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);
					
			scale0[0] = (!fullDepth && Inputs[first]->Skip) ? Float(0) : Float(1);
			scale1[0] = (!fullDepth && Inputs[second]->Skip) ? Float(0) : Float(1);

			if (training)
			{
				if ((Reference || ReferenceAdd ) && fullDepth)
				{
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, fwdArgs);
#else
					dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
					Device.stream.wait();

#ifndef DNN_LEAN
					InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
#endif // DNN_LEAN
				}
				else
				{
					const auto padded = C == PaddedC;
					const auto plain = IsPlainFormat();
					const auto size = GetElementsCount();
					const auto part = GetVectorPart(size);
					const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * size, FwdTrainingWeight);
					const auto strideHW = HW() * VectorSize;

					if (plain)
					{
						if (EqualDimensions(Inputs))
						{
							if (fullDepth)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * size;
									const auto end = start + size;
									PRAGMA_OMP_SIMD()
									for (auto cdhw = start; cdhw < end; cdhw++)
									{
										Neurons[cdhw] = Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = 0;
#endif
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * size;
									const auto end = start + size;
									const auto scales0 = scales[0];
									const auto scales1 = scales[1];
									PRAGMA_OMP_SIMD()
									for (auto cdhw = start; cdhw < end; cdhw++)
									{
										Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * scales0) + (Inputs[1]->Neurons[cdhw] * scales1);
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = 0;
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
										for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
										{
											Neurons[hw] = Inputs[first]->Neurons[hw] + Inputs[second]->Neurons[channelOffset];
#ifndef DNN_LEAN
											NeuronsD1[hw] = 0;
#endif
										}
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto scales0 = scales[first];
									const auto scales1 = scales[second];

									for (auto c = 0ull; c < C; c++)
									{
										const auto outputOffset = n * CDHW() + c * HW();
										const auto channelOffset = n * C + c;
										PRAGMA_OMP_SIMD()
										for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
										{
											Neurons[hw] = ((Inputs[first]->Neurons[hw] * scales0) + (Inputs[second]->Neurons[channelOffset] * scales1));
#ifndef DNN_LEAN
											NeuronsD1[hw] = 0;
#endif
										}
									}
								});
						}
					}
					else
					{
						if (EqualDimensions(Inputs)) // same H and W
						{
							if (fullDepth)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * size;
									for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
									{
										(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) + VecFloat().load_a(&Inputs[1]->Neurons[cdhw])).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
										VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
									}
									for (auto cdhw = start + part; cdhw < start + size; cdhw++)
									{
										Neurons[cdhw] = Inputs[0]->Neurons[cdhw] + Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = 0;
#endif
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * size;
									const auto scales0 = scales[0];
									const auto scales1 = scales[1];

									VecFloat In0, In1;
									for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
									{
										In0.load_a(&Inputs[0]->Neurons[cdhw]);
										In1.load_a(&Inputs[1]->Neurons[cdhw]);
										((In0 * scales0) + (In1 * scales1)).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
										VecZero.store_nt(&NeuronsD1[cdhw]);
#endif
									}
									for (auto cdhw = start + part; cdhw < start + size; cdhw++)
									{
										Neurons[cdhw] = (Inputs[0]->Neurons[cdhw] * scales0) + (Inputs[1]->Neurons[cdhw] * scales1);
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = 0;
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
										const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
										const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, c, 0, 0);

										for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
										{
											(VecFloat().load_a(&Inputs[first]->Neurons[hw]) + VecFloat().load_a(&Inputs[second]->Neurons[channelOffset])).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw]);
#endif
										}
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto scales0 = scales[first];
									const auto scales1 = scales[second];

									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
										const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, c, 0, 0);
										for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
										{
											((VecFloat().load_a(&Inputs[first]->Neurons[hw]) * scales0) + (VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]) * scales1)).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw]);
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
		*/

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

				if (EqualDimensions(Inputs))
				{
					if (plain)
					{
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < size; cdhw++)
						{
							InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales0;
							InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales1;
						}
					}
					else
					{
						VecFloat inputD1, D1;
						for (auto cdhw = 0ull; cdhw < part; cdhw += VectorSize)
						{
							D1.load_a(&NeuronsD1[cdhw]);

							inputD1.load_a(&InputsBwd[0]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales0;
							inputD1.store_a(&InputsBwd[0]->NeuronsD1[cdhw]);

							inputD1.load_a(&InputsBwd[1]->NeuronsD1[cdhw]);
							inputD1 += D1 * scales1;
							inputD1.store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
						}
						for (auto cdhw = part; cdhw < size; cdhw++)
						{
							InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales0;
							InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * scales1;
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
							for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
							{
								InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw];
								InputsBwd[second]->NeuronsD1[c] += NeuronsD1[hw];
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
							for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
							{
								D1.load_a(&NeuronsD1[hw]);
								(D1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
								(D1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[c])).store_a(&InputsBwd[second]->NeuronsD1[c]);
							}
						}
					}
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * size, BwdTrainingWeight);

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
									InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
									InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
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

								VecFloat neuronsD1;
								for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
								{
									neuronsD1.load_a(&NeuronsD1[cdhw]);
									(VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw]) + neuronsD1).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
									(VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw]) + neuronsD1).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
								}
								for (auto cdhw = start + part; cdhw < start + size; cdhw++)
								{
									InputsBwd[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
									InputsBwd[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw];
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto start = n * size;
								const auto scale0 = scales[0];
								const auto scale1 = scales[1];

								VecFloat neuronsD1;
								for (auto cdhw = start; cdhw < start + part; cdhw += VectorSize)
								{
									neuronsD1.load_a(&NeuronsD1[cdhw]);
									mul_add(neuronsD1, scale0, VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw])).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
									mul_add(neuronsD1, scale1, VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw])).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
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
									for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw];
										InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw];
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
									for (auto hw = outputOffset; hw < outputOffset + HW(); hw++)
									{
										InputsBwd[first]->NeuronsD1[hw] += NeuronsD1[hw] * scale0;
										InputsBwd[second]->NeuronsD1[channelOffset] += NeuronsD1[hw] * scale1;
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
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw]);
										(neuronsD1 + VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										(neuronsD1 + VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
									}
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto scale0 = scales[first];
								const auto scale1 = scales[second];
								VecFloat neuronsD1;
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto outputOffset = OffsetPaddedMem(n, c, 0, 0);
									const auto channelOffset = Inputs[second]->OffsetPaddedMem(n, c, 0, 0);
									for (auto hw = outputOffset; hw < outputOffset + strideHW; hw += VectorSize)
									{
										neuronsD1.load_a(&NeuronsD1[hw]);
										mul_add(neuronsD1, scale0, VecFloat().load_a(&InputsBwd[first]->NeuronsD1[hw])).store_a(&InputsBwd[first]->NeuronsD1[hw]);
										mul_add(neuronsD1, scale1, VecFloat().load_a(&InputsBwd[second]->NeuronsD1[channelOffset])).store_a(&InputsBwd[second]->NeuronsD1[channelOffset]);
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
		
/*
		void BackwardProp(const UInt batchSize)
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif

			const auto fullDepth = SurvivalProbability[first] == Float(1) && SurvivalProbability[second] == Float(1);
			scale0[0] = fullDepth ? Float(1) : (Inputs[first]->Skip ? Float(0) : Float(1));
			scale1[0] = fullDepth ? Float(1) : (Inputs[second]->Skip ? Float(0) : Float(1));

			if (EqualDimensions(Inputs))
			{

#ifdef DNN_CACHE_PRIMITIVES
				bwd0->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#else
				dnnl::binary(*bwd0Desc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#endif
				Device.stream.wait();

#ifdef DNN_CACHE_PRIMITIVES
				bwd1->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#else
				dnnl::binary(*bwd1Desc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#endif
				Device.stream.wait();
			}
			else
			{
#ifdef DNN_CACHE_PRIMITIVES
				bwd0->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#else
				dnnl::binary(*bwd0Desc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[first]->DiffDstMemDesc, Device.engine, InputsBwd[first]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#endif
				Device.stream.wait();

				auto reductionMem = dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine);

#ifdef DNN_CACHE_PRIMITIVES
				bwdReduction->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, reductionMem } });
				
#else
				dnnl::reduction(*bwdReduction).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, reductionMem } });
#endif
				Device.stream.wait();

#ifdef DNN_CACHE_PRIMITIVES
				bwd1->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, reductionMem }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#else
				dnnl::binary(*bwd1Desc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, reductionMem }, { DNNL_ARG_DST, dnnl::memory(*InputsBwd[second]->DiffDstMemDesc, Device.engine, InputsBwd[second]->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#endif
				Device.stream.wait();
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
*/
	};
}