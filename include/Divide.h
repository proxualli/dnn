#pragma once
#include "Layer.h"

namespace dnn
{
	class Divide final : public Layer
	{
	public:
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

		Divide(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Divide, 0, 0, inputs[GetFirst(inputs)]->C, inputs[GetFirst(inputs)]->D, inputs[GetFirst(inputs)]->H, inputs[GetFirst(inputs)]->W, 0, 0, 0, inputs),
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
			H = InputsFwd[first]->H;
			W = InputsFwd[first]->W;
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

			fwdDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_div, *Inputs[first]->DstMemDesc, *Inputs[second]->DstMemDesc, *DstMemDesc));

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*Inputs[first]->DstMemDesc, Device.engine, Inputs[first]->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*Inputs[second]->DstMemDesc, Device.engine, Inputs[second]->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::binary>(dnnl::binary(*fwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW(): PaddedCDHW();
			const auto part = GetVectorPart(size);
			const auto threads = GetThreads(batchSize * size, Float(0.25));
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					VecFloat vecZero = VecFloat(0);
					VecFloat In, Out;
					for (auto cdhw = 0ull; cdhw < PaddedCDHW(); cdhw += VectorSize)
					{
						In.load_a(&Inputs[0]->Neurons[cdhw]);
						In.store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
						vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
					}
					for (auto i = 1ull; i < 2ull; i++)
						for (auto cdhw = 0ull; cdhw < PaddedCDHW(); cdhw += VectorSize)
						{
							In.load_a(&Inputs[i]->Neurons[cdhw]);
							Out.load_a(&Neurons[cdhw]);
							Out /= In;
							Out.store_a(&Neurons[cdhw]);
						}
				}
				else
					for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
					{
						Neurons[cdhw] = Inputs[0]->Neurons[cdhw];
#ifndef DNN_LEAN
						NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
					}
				for (auto i = 1ull; i < 2ull; i++)
					for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
						Neurons[cdhw] /= Inputs[i]->Neurons[cdhw];
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
							const auto start = n * PaddedCDHW();
							const auto end = start + PaddedCDHW();
							const auto vecZero = VecFloat(0);
							VecFloat In0, In1;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								(In0 / In1).store_a(&Neurons[cdhw]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[cdhw]);
#endif // DNN_LEAN
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * CDHW();
							const auto end = start + CDHW();
							for (auto cdhw = start; cdhw < end; cdhw++)
							{
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw];
#ifndef DNN_LEAN
								NeuronsD1[cdhw] = Float(0);
#endif // DNN_LEAN
							}
						});
					}
				}
				else
				{
					if (!plain)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * PaddedCDHW();
							const auto end = start + PaddedCDHW();
							const auto vecZero = VecFloat(0);
							VecFloat In0, In1;
							for (auto cdhw = start; cdhw < end; cdhw += VectorSize)
							{
								In0.load_a(&Inputs[0]->Neurons[cdhw]);
								In1.load_a(&Inputs[1]->Neurons[cdhw]);
								(In0 / In1).store_a(&Neurons[cdhw]);
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * CDHW();
							const auto end = start + CDHW();
							for (auto cdhw = start; cdhw < end; cdhw++)
								Neurons[cdhw] = Inputs[0]->Neurons[cdhw] / Inputs[1]->Neurons[cdhw];
						});						
					}
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto size = plain ? CDHW() : PaddedCDHW();
			const auto part = GetVectorPart(size);
			const auto threads = GetThreads(batchSize * size, Float(0.25));
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto cdhw = 0ull; cdhw < PaddedCDHW(); cdhw += VectorSize)
					{
						mul_add(approx_recipr(VecFloat().load_a(&InputsFwd[1]->Neurons[cdhw])), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
						mul_add(VecFloat().load_a(&Inputs[0]->Neurons[cdhw]) * VecFloat().load_a(&NeuronsD1[cdhw]), approx_recipr(square(VecFloat().load_a(&InputsFwd[1]->Neurons[cdhw]))), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
					}

				}
				else
				{
					PRAGMA_OMP_SIMD()
					for (auto cdhw = 0ull; cdhw < CDHW(); cdhw++)
					{
						Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / InputsFwd[1]->Neurons[cdhw];
						Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[0]->Neurons[cdhw] / Square<Float>(InputsFwd[1]->Neurons[cdhw]);
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
							mul_add(approx_recipr(VecFloat().load_a(&InputsFwd[1]->Neurons[cdhw])), VecFloat().load_a(&NeuronsD1[cdhw]), VecFloat().load_a(&Inputs[0]->NeuronsD1[cdhw])).store_a(&Inputs[0]->NeuronsD1[cdhw]);
							mul_add(VecFloat().load_a(&InputsFwd[0]->Neurons[cdhw]) * VecFloat().load_a(&NeuronsD1[cdhw]), approx_recipr(square(VecFloat().load_a(&InputsFwd[1]->Neurons[cdhw]))), VecFloat().load_a(&Inputs[1]->NeuronsD1[cdhw])).store_a(&Inputs[1]->NeuronsD1[cdhw]);
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
							Inputs[0]->NeuronsD1[cdhw] += NeuronsD1[cdhw] / InputsFwd[1]->Neurons[cdhw];
							Inputs[1]->NeuronsD1[cdhw] += NeuronsD1[cdhw] * InputsFwd[0]->Neurons[cdhw] / Square<Float>(InputsFwd[1]->Neurons[cdhw]);
						}
					});
					
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