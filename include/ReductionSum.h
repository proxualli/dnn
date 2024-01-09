#pragma once
#include "Layer.h"

namespace dnn
{
	class ReductionSum final : public Layer
	{
	private:
		std::unique_ptr<dnnl::reduction::primitive_desc> fwdDescReduction;
		std::unordered_map<int, dnnl::memory> fwdArgs;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::reduction> fwdReduction;
#endif
		
	public:
		ReductionSum(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::ReductionSum, 0, 0, 1, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs)
		{
		}

		void UpdateResolution() final override
		{
			D = Inputs[0]->D;
			H = Inputs[0]->H;
			W = Inputs[0]->W;
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

			fwdDescReduction = std::make_unique<dnnl::reduction::primitive_desc>(dnnl::reduction::primitive_desc(Device.engine, dnnl::algorithm::reduction_sum, *InputLayer->DstMemDesc, *DstMemDesc, 0.f, 0.f));
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

			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradientMulti(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
			const auto threads = batchSize == 1 ? 1ull : GetThreads(elements, Float(10));
			const auto strideHW = HW() * VectorSize;

#ifdef DNN_STOCHASTIC
			if (batchSize == 1)
			{
				if (!plain)
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat InD1, OutD1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto inputOffset = c * HW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								InD1.load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								OutD1.load_a(&NeuronsD1[hw]);
								InD1 += OutD1;
								InD1.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
							}
						}
					});
				}
				else
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * CDHW();
						const auto inStart = n * InputLayer->CDHW();

						for (auto c = 0ull; c < C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0; hw < HW(); hw++)
								InputLayer->NeuronsD1[(c * HW()) + hw] += NeuronsD1[hw];
					});
				}
			}
			else
			{
#endif
				if (!plain)
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat InD1, OutD1;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto inputOffset = (n * InputLayer->PaddedCDHW()) + (c  * HW());
							const auto outputOffset = n * PaddedCDHW();
							for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
							{
								InD1.load_a(&InputLayer->NeuronsD1[hw + inputOffset]);
								OutD1.load_a(&NeuronsD1[hw + outputOffset]);
								InD1 += OutD1;
								InD1.store_a(&InputLayer->NeuronsD1[hw + inputOffset]);
							}
						}
					});
				}
				else
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						const auto start = n * CDHW();
						const auto inStart = n * InputLayer->CDHW();
						
						for (auto c = 0ull; c < C; c++)
							PRAGMA_OMP_SIMD()
							for (auto hw = 0; hw < HW(); hw++)
								InputLayer->NeuronsD1[inStart + (c * HW()) + hw] += NeuronsD1[start + hw];
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