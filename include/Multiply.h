#pragma once
#include "Layer.h"

namespace dnn
{
	class Multiply final : public Layer
	{
	private:
		std::vector<Float> scales;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> fwdDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::binary> fwd;
#endif
		auto EqualDimensions(const std::vector<Layer*>& inputs)
		{
			return ((inputs[0]->H == inputs[1]->H) && (inputs[0]->W == inputs[1]->W));
		}
		auto GetFirst(const std::vector<Layer*>& inputs)
		{
			return EqualDimensions(inputs) ? Byte(0) : ((inputs[0]->H == 1 && inputs[0]->W == 1) ? Byte(1) : Byte(0));
		}
		auto GetSecond(const std::vector<Layer*>& inputs)
		{
			return EqualDimensions(inputs) ? Byte(1) : ((inputs[0]->H == 1 && inputs[0]->W == 1) ? Byte(0) : Byte(1));
		}

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
			assert(Inputs[0]->C == Inputs[1]->C);
			assert(Inputs[0]->D == Inputs[1]->D);

			scales = std::vector<Float>(2, Float(1));
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
			if (Inputs[first]->DstMemDesc->get_ndims() == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetDataFmt(*Inputs[first]->DstMemDesc);
					if (ChosenFormat != GetDataFmt(*Inputs[first]->DiffDstMemDesc))
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

		void ForwardProp(const UInt batchSize, const bool training)  final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::binary(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#else
			DNN_UNREF_PAR(batchSize);
#endif
		}

		void BackwardProp(const UInt batchSize)  final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
			const auto threads = GetThreads(elements);
			
			if (EqualDimensions(Inputs))
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (!plain)
					{
						for (auto cdhw = 0ull; cdhw < PaddedCDHW(); cdhw += VectorSize)
						{
							mul_add(VecFloat().load_a(&InputsOriginal[1]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&InputsOriginal[0]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
						}
					}
					else
					{
						PRAGMA_OMP_SIMD()
						for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
						{
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsOriginal[1]->Neurons[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsOriginal[0]->Neurons[cdhw];
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
								mul_add(VecFloat().load_a(&InputsOriginal[1]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
								mul_add(VecFloat().load_a(&InputsOriginal[0]->Neurons[cdhw]), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
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
								Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsOriginal[1]->Neurons[cdhw];
								Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsOriginal[0]->Neurons[cdhw];
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
								mul_add(neuronsD1, VecFloat().load_a(&InputsOriginal[second]->Neurons[c]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
								mul_add(neuronsD1, VecFloat().load_a(&InputsOriginal[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[c])).store_a(&Inputs[second]->NeuronsD1[c]);
							}
						}
					}
					else
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto outputOffset = c * HW();
							for (auto hw = 0ull; hw < HW(); hw++)
							{
								Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * InputsOriginal[second]->Neurons[c];
								Inputs[second]->NeuronsD1[c] += NeuronsD1[hw + outputOffset] * InputsOriginal[first]->Neurons[hw + outputOffset];
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
								const auto channelOffset = n * PaddedC + c;
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									neuronsD1.load_a(&NeuronsD1[hw + outputOffset]);
									mul_add(neuronsD1, VecFloat().load_a(&InputsOriginal[second]->Neurons[channelOffset]), VecFloat().load_a(&Inputs[first]->NeuronsD1[hw + outputOffset])).store_a(&Inputs[first]->NeuronsD1[hw + outputOffset]);
									mul_add(neuronsD1, VecFloat().load_a(&InputsOriginal[first]->Neurons[hw + outputOffset]), VecFloat().load_a(&Inputs[second]->NeuronsD1[channelOffset])).store_a(&Inputs[second]->NeuronsD1[channelOffset]);
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
								for (auto hw = 0ull; hw < HW(); hw++)
								{
									Inputs[first]->NeuronsD1[hw + outputOffset] += NeuronsD1[hw + outputOffset] * InputsOriginal[second]->Neurons[channelOffset];
									Inputs[second]->NeuronsD1[channelOffset] += NeuronsD1[hw + outputOffset] * InputsOriginal[first]->Neurons[hw + outputOffset];
								}
							}
						});
					}
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