#pragma once
#include "Layer.h"

namespace dnn
{
	class Max final : public Layer
	{
	private:
		std::vector<Float> scales;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif

	public:
		const Byte first, second;
		FloatVector SurvivalProbability;

		Max(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Max, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
			first(GetFirst(inputs)),
			second(GetSecond(inputs)),
			SurvivalProbability(FloatVector(2, Float(1)))
		{
			assert(Inputs.size() == 2);
			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->D == Inputs[1]->D);

			scales = std::vector<Float>(2, Float(1));

			FwdZeroGradient = Float(1);
			FwdInferenceWeight = Float(2);
			FwdTrainingWeight = Float(4);
			BwdTrainingWeight = Float(4);
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

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_max, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

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
					InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
#endif // DNN_LEAN
				}
				else
				{
					const auto plain = IsPlainFormat();
					const auto size = plain ? CDHW() : PaddedCDHW();
					const auto part = GetVectorPart(size);
					const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);
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
										Neurons[cdhw] = std::max(Inputs[0]->Neurons[cdhw], Inputs[1]->Neurons[cdhw]);
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
									const auto scales0 = scales[0];
									const auto scales1 = scales[1];
									PRAGMA_OMP_SIMD()
									for (auto cdhw = start; cdhw < end; cdhw++)
									{
										Neurons[cdhw] = std::max(Inputs[0]->Neurons[cdhw] * scales0, Inputs[1]->Neurons[cdhw] * scales1);
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
											Neurons[hw + outputOffset] = std::max(Inputs[first]->Neurons[hw + outputOffset], Inputs[second]->Neurons[channelOffset]);
#ifndef DNN_LEAN
											NeuronsD1[hw + outputOffset] = Float(0);
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
										for (auto hw = 0ull; hw < HW(); hw++)
										{
											Neurons[hw + outputOffset] = std::max(Inputs[first]->Neurons[hw + outputOffset] * scales0, Inputs[second]->Neurons[channelOffset] * scales1);
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
										(max(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]), VecFloat().load_a(&Inputs[1]->Neurons[cdhw]))).store_a(&Neurons[cdhw]);
										VecZero.store_nt(&NeuronsD1[cdhw]);
									}
									for (auto cdhw = start + part; cdhw < start + size; cdhw++)
									{
										Neurons[cdhw] = std::max(Inputs[0]->Neurons[cdhw], Inputs[1]->Neurons[cdhw]);
#ifndef DNN_LEAN
										NeuronsD1[cdhw] = Float(0);
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
										(max(In0 * scales0, In1 * scales1)).store_a(&Neurons[cdhw]);
										VecZero.store_nt(&NeuronsD1[cdhw]);
									}
									for (auto cdhw = start + part; cdhw < start + size; cdhw++)
									{
										Neurons[cdhw] = std::max(Inputs[0]->Neurons[cdhw] * scales0, Inputs[1]->Neurons[cdhw] * scales1);
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
											(max(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]))).store_a(&Neurons[hw + outputOffset]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[hw + outputOffset]);
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
										const auto outputOffset = n * PaddedCDHW() + c * HW();
										const auto channelOffset = n * PaddedC + c;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											(max(VecFloat().load_a(&Inputs[first]->Neurons[hw + outputOffset]) * scales0, VecFloat().load_a(&Inputs[second]->Neurons[channelOffset]) * scales1)).store_a(&Neurons[hw + outputOffset]);
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
			const auto size = GetElementsCount();
			const auto part = GetVectorPart(size);
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				VecFloat In0, In1, D1;
				for (auto cdhw = 0ull; cdhw < part; cdhw+=VectorSize)
				{
					In0.load_a(&Inputs[0]->Neurons[cdhw]);
					In1.load_a(&Inputs[1]->Neurons[cdhw]);
					D1.load_a(&NeuronsD1[cdhw]);
					if_add(In0 >= In1, VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw]), D1).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
					if_add(In0 < In1, VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw]), D1).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
				}
				for (auto cdhw = part; cdhw < size; cdhw++)
				{
					InputsBwd[0]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? NeuronsD1[cdhw] : Float(0);
					InputsBwd[1]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? Float(0) : NeuronsD1[cdhw];
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * size, BwdTrainingWeight);

				for_i(batchSize, threads, [=](UInt n)
				{
					const auto start = n * size;
					const auto end = start + part;

					VecFloat In0, In1, D1;
					for (auto cdhw = start; cdhw < end; cdhw+=VectorSize)
					{
						In0.load_a(&Inputs[0]->Neurons[cdhw]);
						In1.load_a(&Inputs[1]->Neurons[cdhw]);
						D1.load_a(&NeuronsD1[cdhw]);

						if_add(In0 >= In1, VecFloat().load_a(&InputsBwd[0]->NeuronsD1[cdhw]), D1).store_a(&InputsBwd[0]->NeuronsD1[cdhw]);
						if_add(In0 < In1, VecFloat().load_a(&InputsBwd[1]->NeuronsD1[cdhw]), D1).store_a(&InputsBwd[1]->NeuronsD1[cdhw]);
					}
					for (auto cdhw = end; cdhw < start + size; cdhw++)
					{
						InputsBwd[0]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? NeuronsD1[cdhw] : Float(0);
						InputsBwd[1]->NeuronsD1[cdhw] += Inputs[0]->Neurons[cdhw] >= Inputs[1]->Neurons[cdhw] ? Float(0) : NeuronsD1[cdhw];
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