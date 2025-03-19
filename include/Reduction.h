#pragma once
#include "Layer.h"

namespace dnn
{
	class Reduction final : public Layer
	{
	private:
		std::unique_ptr<dnnl::reduction::primitive_desc> fwdDesc;
		std::unordered_map<int, dnnl::memory> fwdArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdBinaryAddDesc;
		std::unordered_map<int, dnnl::memory> bwdBinaryAddArgs;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdBinaryEqualDesc;
		//std::unordered_map<int, dnnl::memory> bwdBinaryEqualArgs;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::reduction> fwd;
		std::unique_ptr<dnnl::binary> bwdBinaryAdd; 
		std::unique_ptr<dnnl::binary> bwdBinaryEqual;
#endif
		dnnl::algorithm algorithm; 
		FloatVector scale0;
		FloatVector scale1;
	public:
		const ReduceOperations Op;
		const Float P;
		const Float Eps;

		Reduction(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const ReduceOperations op, const Float p = Float(0), const Float eps = Float(0)) :
			Layer(device, format, name, LayerTypes::Reduction, 0, 0, 1, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Op(op),
			Eps(eps),
			P(p),
			algorithm(dnnl::algorithm::reduction_mean),
			scale0(FloatVector(1, Float(1))),
			scale1(FloatVector(1, (Float(1) / Float(inputs[0]->C))))
		{
			FwdZeroGradient = Float(1);
			FwdInferenceWeight = Float(5);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
		}

		void UpdateResolution() final override
		{
			D = Inputs[0]->D;
			H = Inputs[0]->H;
			W = Inputs[0]->W;
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader().append(nwl + std::string(" Operation:  ") + tab + std::string(magic_enum::enum_name<ReduceOperations>(Op)));
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

			switch (Op)
			{
			case ReduceOperations::Avg:
				algorithm = dnnl::algorithm::reduction_mean;
				break;
			case ReduceOperations::Max:
				algorithm = dnnl::algorithm::reduction_max;
				break;
			case ReduceOperations::Min:
				algorithm = dnnl::algorithm::reduction_min;
				break;			
			case ReduceOperations::Sum:
				algorithm = dnnl::algorithm::reduction_sum;
				break;
			}

			fwdDesc = std::make_unique<dnnl::reduction::primitive_desc>(dnnl::reduction::primitive_desc(Device.engine, algorithm, *InputLayer->DstMemDesc, *DstMemDesc, P, Eps));

			dnnl::primitive_attr attr;
			attr.set_scales_mask(DNNL_ARG_SRC_0, 0);
			attr.set_scales_mask(DNNL_ARG_SRC_1, 0);
			bwdBinaryAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, *InputLayer->DiffDstMemDesc, attr));

			
			dnnl::post_ops binary_ops;
			binary_ops.append_binary(dnnl::algorithm::binary_mul, *DiffDstMemDesc);
			binary_ops.append_binary(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc);
			dnnl::primitive_attr binary_attr;
			binary_attr.set_post_ops(binary_ops);

			bwdBinaryEqualDesc= std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_eq, *InputLayer->DstMemDesc, *DstMemDesc, *InputLayer->DstMemDesc, binary_attr));
			

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::reduction>(dnnl::reduction(*fwdDesc));
			bwdBinaryAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdBinaryAddDesc));
			bwdBinaryEqual = std::make_unique<dnnl::binary>(dnnl::binary(*bwdBinaryEqualDesc));
#endif

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC, dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
			bwdBinaryAddArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } };
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, fwdArgs);
#else
			dnnl::reduction(*fwdDesc).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
#endif
		}


		void BackwardPropAvgRef(const UInt batchSize)
		{
			auto output = FloatArray();
			if constexpr (TestReduction)
			{
				output.resize(batchSize, InputLayerBwd->C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);

#ifdef DNN_CACHE_PRIMITIVES
				bwdBinaryAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, output.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#else
				dnnl::binary(*bwdBinaryAdd).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, output.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#endif
				Device.stream.wait();
			}
			
			const auto plain = IsPlainFormat();
			const auto factor = Float(1) / Float(InputLayer->C);

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)] * factor;

				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += NeuronsD1[hw] * factor;
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)] * factor;
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw] * factor;
					});
#ifdef DNN_STOCHASTIC
			}
#endif


			if constexpr (TestReduction)
			{
				const auto margin = Float(0.0005);

				for (auto i = 0ull; i < InputLayerBwd->NeuronsD1.size(); i++)
				{
					if (((output[i] - margin) > InputLayerBwd->NeuronsD1[i]) || ((output[i] + margin) < InputLayerBwd->NeuronsD1[i]))
					{
						cimg_library::cimg::dialog("Reduction Sanity Check", (std::string("Backward Check not passed: ") + Name).c_str(), "OK");
						break;
					}
				}
			}
		}

		void BackwardPropAvg(const UInt batchSize)
		{
#ifdef DNN_CACHE_PRIMITIVES
			bwdBinaryAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#else
			dnnl::binary(*bwdBinaryAdd).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale0.data()) }, { DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, dnnl::memory(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), Device.engine, scale1.data()) } });
#endif
			Device.stream.wait();
			/*
			const auto plain = IsPlainFormat();
			const auto factor = Float(1) / Float(InputLayer->C);
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)] * factor;
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += NeuronsD1[hw] * factor;
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				
				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)] * factor;
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw] * factor;
					});
#ifdef DNN_STOCHASTIC
			}
#endif
*/
		}

		void BackwardPropMaxRef(const UInt batchSize)
		{
			auto output = FloatArray();
			if constexpr (TestReduction)
			{
				output.resize(batchSize, InputLayerBwd->C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
					
#ifdef DNN_CACHE_PRIMITIVES
				bwdBinaryEqual->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DstMemDesc, Device.engine, output.data()) }, { DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) } });
#else
				dnnl::binary(*bwdBinaryEqual).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_SRC_1, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DstMemDesc, Device.engine, output.data()) }, { DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) } });
#endif
				Device.stream.wait();
			}

			
			const auto plain = IsPlainFormat();
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
					
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
					});
#ifdef DNN_STOCHASTIC
			}
#endif

			if constexpr (TestReduction)
			{
				const auto margin = Float(0.0005);

				for (auto i = 0ull; i < InputLayerBwd->NeuronsD1.size(); i++)
				{
					if (((output[i] - margin) > InputLayerBwd->NeuronsD1[i]) || ((output[i] + margin) < InputLayerBwd->NeuronsD1[i]))
					{
						cimg_library::cimg::dialog("Reduction Sanity Check", (std::string("Backward Check not passed: ") + Name).c_str(), "OK");
						break;
					}
				}
			}
		}

		void BackwardPropMax(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
					
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const bool padded = InputLayerBwd->PaddedC == InputLayerBwd->C;

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputOffset = OffsetPaddedMem(n, 0, 0, 0);

						if (padded)
							for (auto c = 0ull; c < InputLayerBwd->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayerBwd->OffsetPaddedMem(n, c, 0, 0);
								
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									auto inputNeurons = VecFloat().load_a(&InputLayer->Neurons[hw + inputOffset]);
									auto inputNeuronsD1 = VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
								
									if_add(inputNeurons == Neurons[hw + outputOffset], inputNeuronsD1, NeuronsD1[hw + outputOffset]).store_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
								}
							}
						else
							for (auto c = 0ull; c < InputLayerBwd->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropMinRef(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropMin(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const bool padded = InputLayerBwd->PaddedC == InputLayerBwd->C;

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputOffset = OffsetPaddedMem(n, 0, 0, 0);

						if (padded)
							for (auto c = 0ull; c < InputLayerBwd->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayerBwd->OffsetPaddedMem(n, c, 0, 0);
								
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									auto inputNeurons = VecFloat().load_a(&InputLayer->Neurons[hw + inputOffset]);
									auto inputNeuronsD1 = VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);

									if_add(inputNeurons == Neurons[hw + outputOffset], inputNeuronsD1, NeuronsD1[hw + outputOffset]).store_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
								}
							}
						else
							for (auto c = 0ull; c < InputLayerBwd->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayerBwd->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropSumRef(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)];
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += NeuronsD1[ hw];
				}
			}
			else
			{
#endif
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)];
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw];
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropSum(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)];
				}
				else
				{
					for (auto c = 0ull; c < InputLayerBwd->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayerBwd->NeuronsD1[(c * HW()) + hw] += NeuronsD1[hw];
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const bool padded = InputLayerBwd->PaddedC == InputLayerBwd->C;

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputOffset = OffsetPaddedMem(n, 0, 0, 0);

						if (padded)
							for (auto c = 0ull; c < InputLayerBwd->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayerBwd->OffsetPaddedMem(n, c, 0, 0);

								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									auto inputNeuronsD1 = VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);

									inputNeuronsD1 += NeuronsD1[hw + outputOffset];
									inputNeuronsD1.store_a(&InputLayerBwd->NeuronsD1[hw + inputOffset]);
								}
							}
						else
							for (auto c = 0ull; c < InputLayerBwd->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayerBwd->NeuronsD1[InputLayerBwd->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, c, h, w)];
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayerBwd->CDHW();
						for (auto c = 0ull; c < InputLayerBwd->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayerBwd->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw];
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			auto output = FloatArray();
			if constexpr (TestReduction)
			{
				output.resize(batchSize, InputLayerBwd->C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto i = 0ull; i < InputLayerBwd->NeuronsD1.size(); i++)
					output[i] = InputLayerBwd->NeuronsD1[i];
			}

			if constexpr ((Reference || ReferenceReduction) && !TestReduction)
			{
				switch (Op)
				{
				case ReduceOperations::Avg:
					BackwardPropAvgRef(batchSize);
					break;
				case ReduceOperations::Max:
					BackwardPropMaxRef(batchSize);
					break;
				case ReduceOperations::Min:
					BackwardPropMinRef(batchSize);
					break;
				case ReduceOperations::Sum:
					BackwardPropSumRef(batchSize);
					break;
				}
			}
			else
			{
				switch (Op)
				{
				case ReduceOperations::Avg:
					BackwardPropAvg(batchSize);
					break;
				case ReduceOperations::Max:
					BackwardPropMax(batchSize);
					break;
				case ReduceOperations::Min:
					BackwardPropMin(batchSize);
					break;
				case ReduceOperations::Sum:
					BackwardPropSum(batchSize);
					break;
				}
			}

			if constexpr (TestReduction)
			{
				auto input = FloatArray();
				input.resize(batchSize, InputLayerBwd->C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto i = 0ull; i < InputLayerBwd->NeuronsD1.size(); i++)
					input[i] = InputLayerBwd->NeuronsD1[i];

				for (auto i = 0ull; i < InputLayerBwd->NeuronsD1.size(); i++)
					InputLayerBwd->NeuronsD1[i] = output[i];

				switch (Op)
				{
				case ReduceOperations::Avg:
					BackwardPropAvgRef(batchSize);
					break;
				case ReduceOperations::Max:
					BackwardPropMaxRef(batchSize);
					break;
				case ReduceOperations::Min:
					BackwardPropMinRef(batchSize);
					break;
				case ReduceOperations::Sum:
					BackwardPropSumRef(batchSize);
					break;
				}

				const auto margin = Float(0.001);

				for (auto i = 0ull; i < InputLayerBwd->NeuronsD1.size(); i++)
				{
					auto ref = input[i];
					auto val = InputLayerBwd->NeuronsD1[i];
					if (((ref - margin) > val) || ((ref + margin) < val))
					{
						cimg_library::cimg::dialog("Reduction Sanity Check", (std::string("Backward Check not passed: ") + Name + nwl + std::to_string(ref) + nwl + std::to_string(val)).c_str(), "OK");
						break;
					}
				}
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}