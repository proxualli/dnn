#pragma once
#include "Layer.h"

namespace dnn
{
	class Reduction final : public Layer
	{
	private:
		std::unique_ptr<dnnl::reduction::primitive_desc> fwdDescReduction;
		std::unordered_map<int, dnnl::memory> fwdArgs;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::reduction> fwdReduction;
#endif
		dnnl::algorithm algorithm; 

	public:
		const ReduceOperations Op;
		const Float P;
		const Float Eps;

		Reduction(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const ReduceOperations op, const Float p = Float(0), const Float eps = Float(0)) :
			Layer(device, format, name, LayerTypes::Reduction, 0, 0, 1, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Op(op),
			Eps(eps),
			P(p),
			algorithm(dnnl::algorithm::reduction_mean)
		{
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

			fwdDescReduction = std::make_unique<dnnl::reduction::primitive_desc>(dnnl::reduction::primitive_desc(Device.engine, algorithm, *InputLayer->DstMemDesc, *DstMemDesc, P, Eps));
#ifdef DNN_CACHE_PRIMITIVES
			fwdReduction = std::make_unique<dnnl::reduction>(dnnl::reduction(*fwdDescReduction));
#endif

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescReduction->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescReduction->dst_desc());

			fwdArgs = std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC, dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data()) }, { DNNL_ARG_DST, dnnl::memory(*DstMemDesc, Device.engine, Neurons.data()) } };
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
#ifdef DNN_CACHE_PRIMITIVES
			fwdReduction->execute(Device.stream, fwdArgs);
#else
			dnnl::reduction(*fwdDescReduction).execute(Device.stream, fwdArgs);
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#endif
		}


		void BackwardPropAvgRef(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)] / Float(InputLayer->C);

				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += NeuronsD1[hw] / Float(InputLayer->C);
				}
			}
			else
			{
#endif
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayer->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)] / Float(InputLayerFwd->C);
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayer->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw] / Float(InputLayerFwd->C);
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropAvg(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)] / Float(InputLayer->C);
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += NeuronsD1[hw] / Float(InputLayer->C);
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const auto factor = Float(1) / Float(InputLayer->C);
				const bool padded = InputLayer->PaddedC == InputLayer->C;
				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						if (padded)
							for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
							{
								const auto inputOffset = n * InputLayer->PaddedCDHW() + c * HW();
								const auto outputOffset = n * HW();
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									auto inputD1 = VecFloat().load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
									auto neuronsD1 = VecFloat().load_a(&NeuronsD1[hw + outputOffset]);

									//inputD1 += neuronsD1 / Float(InputLayer->C);
									//inputD1.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);

									mul_add(neuronsD1, factor, inputD1).store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								}
							}
						else
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)] * factor;
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayer->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw] * factor;
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropMaxRef(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
					
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayer->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayer->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}

		void BackwardPropMax(const UInt batchSize)
		{
			const auto plain = IsPlainFormat();
			
#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
					
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const bool padded = InputLayer->PaddedC == InputLayer->C;

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputOffset = OffsetPaddedMem(n, 0, 0, 0);

						if (padded)
							for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayer->OffsetPaddedMem(n, c, 0, 0);
								
								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									auto inputNeurons = VecFloat().load_a(&InputLayer->Neurons[hw + inputOffset]);
									auto inputNeuronsD1 = VecFloat().load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								
									if_add(inputNeurons == Neurons[hw + outputOffset], inputNeuronsD1, NeuronsD1[hw + outputOffset]).store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								}
							}
						else
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayer->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
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
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
						});
				else
					for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * HW();
							const auto inStart = n * InputLayer->CDHW();
							for (auto c = 0ull; c < InputLayer->C; c++)
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
									InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
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
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(0, c, h, w)] == Neurons[OffsetPaddedMem(0, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(0, 0, h, w)] : Float(0));
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += (InputLayer->Neurons[(c * HW()) + hw] == Neurons[hw]) ? NeuronsD1[hw] : Float(0);
				}
			}
			else
			{
#endif
				const auto strideHW = HW() * VectorSize;
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const bool padded = InputLayer->PaddedC == InputLayer->C;

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto outputOffset = OffsetPaddedMem(n, 0, 0, 0);

						if (padded)
							for (auto c = 0ull; c < InputLayer->PaddedC; c += VectorSize)
							{
								const auto inputOffset = InputLayer->OffsetPaddedMem(n, c, 0, 0);

								for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
								{
									auto inputNeurons = VecFloat().load_a(&InputLayer->Neurons[hw + inputOffset]);
									auto inputNeuronsD1 = VecFloat().load_a(&InputLayer->NeuronsD1[hw + inputOffset]);

									if_add(inputNeurons == Neurons[hw + outputOffset], inputNeuronsD1, NeuronsD1[hw + outputOffset]).store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								}
							}
						else
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += (InputLayer->Neurons[InputLayer->OffsetPaddedMem(n, c, h, w)] == Neurons[OffsetPaddedMem(n, 0, h, w)] ? NeuronsD1[OffsetPaddedMem(n, 0, h, w)] : Float(0));
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayer->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += (InputLayer->Neurons[inStart + (c * HW()) + hw] == Neurons[start + hw]) ? NeuronsD1[start + hw] : Float(0);
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
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)];
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += NeuronsD1[start + hw];
				}
			}
			else
			{
#endif
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < InputLayer->C; c++)
								for (auto h = 0ull; h < H; h++)
									for (auto w = 0ull; w < W; w++)
										InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)];
						});
				else
					for_i(batchSize, threads, [=](UInt n)
						{
							const auto start = n * HW();
							const auto inStart = n * InputLayer->CDHW();
							for (auto c = 0ull; c < InputLayer->C; c++)
								PRAGMA_OMP_SIMD()
								for (auto hw = 0ull; hw < HW(); hw++)
									InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw];
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
					for (auto c = 0ull; c < InputLayer->C; c++)
						for (auto h = 0ull; h < H; h++)
							for (auto w = 0ull; w < W; w++)
								InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(0, c, h, w)] += NeuronsD1[OffsetPaddedMem(0, 0, h, w)];
				}
				else
				{
					for (auto c = 0ull; c < InputLayer->C; c++)
						PRAGMA_OMP_SIMD()
						for (auto hw = 0ull; hw < HW(); hw++)
							InputLayer->NeuronsD1[(c * HW()) + hw] += NeuronsD1[start + hw];
				}
			}
			else
			{
#endif
				const auto threads = batchSize == 1ull ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);

				if (!plain)
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < InputLayer->C; c++)
							for (auto h = 0ull; h < H; h++)
								for (auto w = 0ull; w < W; w++)
									InputLayer->NeuronsD1[InputLayer->OffsetPaddedMem(n, c, h, w)] += NeuronsD1[OffsetPaddedMem(n, 0, h, w)];
					});
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * HW();
						const auto inStart = n * InputLayer->CDHW();
						for (auto c = 0ull; c < InputLayer->C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0ull; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw];
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
				output.resize(batchSize, InputLayer->C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
					output[i] = InputLayer->NeuronsD1[i];
			}

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
		
			if constexpr (TestReduction)
			{
				auto input = FloatArray();
				input.resize(batchSize, InputLayer->C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
					input[i] = InputLayer->NeuronsD1[i];

				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
					InputLayer->NeuronsD1[i] = output[i];

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

				const auto margin = Float(0.0005);

				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
				{
					if (((input[i] - margin) > InputLayer->NeuronsD1[i]) || ((input[i] + margin) < InputLayer->NeuronsD1[i]))
					{
						cimg_library::cimg::dialog("Reduction Sanity Check", (std::string("Backward Check not passed: ") + Name).c_str(), "OK");
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