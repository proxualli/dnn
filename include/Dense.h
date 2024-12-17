#pragma once
#include "Layer.h"

namespace dnn
{
	class Dense final : public Layer
	{
	private:
		std::unique_ptr<dnnl::inner_product_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::inner_product_backward_weights::primitive_desc> bwdWeightsDesc;
		std::unique_ptr<dnnl::inner_product_backward_data::primitive_desc> bwdDataDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::inner_product_forward> fwd;
		std::unique_ptr<dnnl::inner_product_backward_weights> bwdWeights;
		std::unique_ptr<dnnl::inner_product_backward_data> bwdData;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		bool reorderFwdSrc;
		bool reorderBwdWeightsSrc;
		bool reorderBwdWeightsDiffWeights;
		bool reorderBwdDataDiffSrc;
		bool reorderBwdDataWeights;
		bool reorderBwdDataDiffDst;
		
	public:
		Dense(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const UInt c, const std::vector<Layer*>& inputs, const bool hasBias) :
			Layer(device, format, name, LayerTypes::Dense, c * inputs[0]->CDHW(), c, c, 1, 1, 1, 0, 0, 0, inputs, hasBias),
			reorderFwdSrc(false),
			reorderBwdWeightsSrc(false),
			reorderBwdWeightsDiffWeights(false),
			reorderBwdDataDiffSrc(false),
			reorderBwdDataWeights(false),
			reorderBwdDataDiffDst(false)			
		{
			assert(Inputs.size() == 1);

			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw));
			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw));
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader() + GetWeightsDescription();

			description.append(nwl + std::string(" Connections:") + tab + std::to_string(CDHW() * (InputLayer->CDHW() + 1)));

			return description;
		}

		UInt FanIn() const final override
		{
			return InputLayer->CDHW();
		}

		UInt FanOut() const final override
		{
			return CDHW();
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			std::vector<dnnl::memory::desc> memDesc;
			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
				memDesc = std::vector<dnnl::memory::desc>({
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C) }), dnnl::memory::data_type::f32, NeuronsFormat),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C)}), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x) });
			}
			else
			{
				memDesc = std::vector<dnnl::memory::desc>({
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, NeuronsFormat),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x) });
			}

			fwdDesc = std::make_unique<dnnl::inner_product_forward::primitive_desc>(HasBias ? 
				dnnl::inner_product_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, memDesc[0], memDesc[2], memDesc[3], memDesc[1]) :
				dnnl::inner_product_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, memDesc[0], memDesc[2], memDesc[1]));

			bwdWeightsDesc = std::make_unique<dnnl::inner_product_backward_weights::primitive_desc>(HasBias ? 
				dnnl::inner_product_backward_weights::primitive_desc(Device.engine, memDesc[0], memDesc[2], memDesc[3], memDesc[1], *fwdDesc) :
				dnnl::inner_product_backward_weights::primitive_desc(Device.engine, memDesc[0], memDesc[2], memDesc[1], *fwdDesc));

			bwdDataDesc = std::make_unique<dnnl::inner_product_backward_data::primitive_desc>(dnnl::inner_product_backward_data::primitive_desc(Device.engine, memDesc[0], memDesc[2], memDesc[1], *fwdDesc));

			if (*WeightsMemDesc != fwdDesc->weights_desc())
			{
				auto weights = FloatVector(fwdDesc->weights_desc().get_size() / sizeof(Float), Float(0));
				auto memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
				auto weightsMem = dnnl::memory(fwdDesc->weights_desc(), Device.engine, weights.data());

				dnnl::reorder(memWeights, weightsMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
				Device.stream.wait();

				Weights = weights;
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->weights_desc());
			}

			WeightsFormat = GetMemoryFormat(*WeightsMemDesc);
			
			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(bwdWeightsDesc->diff_dst_desc());
			
			ChosenFormat = GetMemoryFormat(*DstMemDesc);
			
			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));
			
			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdWeightsSrc = bwdWeightsDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdWeightsDiffWeights = bwdWeightsDesc->diff_weights_desc() != *WeightsMemDesc;
			reorderBwdDataDiffSrc = bwdDataDesc->diff_src_desc() != *InputLayerBwd->DiffDstMemDesc;
			reorderBwdDataWeights = bwdDataDesc->weights_desc() != *WeightsMemDesc;
			reorderBwdDataDiffDst = bwdDataDesc->diff_dst_desc() != *DiffDstMemDesc;
			
#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::inner_product_forward>(dnnl::inner_product_forward(*fwdDesc));
			bwdWeights = std::make_unique<dnnl::inner_product_backward_weights>(dnnl::inner_product_backward_weights(*bwdWeightsDesc));
			bwdData = std::make_unique<dnnl::inner_product_backward_data>(dnnl::inner_product_backward_data(*bwdDataDesc));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto& memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			const auto& weightsMem = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());

			auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());
#ifdef DNN_CACHE_PRIMITIVES
			HasBias ?
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_BIAS, dnnl::memory(fwdDesc->bias_desc(), Device.engine, Biases.data()) }, { DNNL_ARG_DST, dstMem } }) :
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#else
			HasBias ?
				dnnl::inner_product_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_BIAS, dnnl::memory(fwdDesc->bias_desc(), Device.engine, Biases.data()) }, { DNNL_ARG_DST, dstMem } }) :
				dnnl::inner_product_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#else
			DNN_UNREF_PAR(batchSize);
#endif		
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN

			const auto& memDiffDst = dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data());
			
			const auto& memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderBwdWeightsSrc ? dnnl::memory(bwdWeightsDesc->src_desc(), Device.engine) : memSrc;
			if (reorderBwdWeightsSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto memDiffWeights = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsD1.data());
			auto diffWeightsMem = reorderBwdWeightsDiffWeights ? dnnl::memory(bwdWeightsDesc->diff_weights_desc(), Device.engine) : memDiffWeights;

#ifdef DNN_CACHE_PRIMITIVES
			HasBias ?
				bwdWeights->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, memDiffDst }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_BIAS, dnnl::memory(bwdWeightsDesc->diff_bias_desc(), Device.engine, BiasesD1.data()) } }) :
				bwdWeights->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, memDiffDst }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem } });
#else
			HasBias ?
				dnnl::inner_product_backward_weights(*bwdWeightsDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, memDiffDst }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_BIAS, dnnl::memory(bwdWeightsDesc->diff_bias_desc(), Device.engine, BiasesD1.data()) } }) :
				dnnl::inner_product_backward_weights(*bwdWeightsDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, memDiffDst }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem } });

#endif
			Device.stream.wait();

			if (reorderBwdWeightsDiffWeights)
			{
				dnnl::reorder(diffWeightsMem, memDiffWeights).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffWeightsMem}, { DNNL_ARG_TO, memDiffWeights } });
				Device.stream.wait();
			}

			const auto& memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
			auto weightsMem = reorderBwdDataWeights ? dnnl::memory(bwdDataDesc->weights_desc(), Device.engine) : memWeights;
			if (reorderBwdDataWeights)
			{
				dnnl::reorder(memWeights, weightsMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
				Device.stream.wait();
			}

			auto memDiffSrc = SharesInputInplace ? dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDataDiffSrc ? dnnl::memory(bwdDataDesc->diff_src_desc(), Device.engine) : memDiffSrc;

			auto diffDataDstMem = reorderBwdDataDiffDst ? dnnl::memory(bwdDataDesc->diff_dst_desc(), Device.engine) : memDiffDst;
			if (reorderBwdDataDiffDst)
			{
				dnnl::reorder(memDiffDst, diffDataDstMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memDiffDst}, { DNNL_ARG_TO, diffDataDstMem } });
				Device.stream.wait();
			}

#ifdef DNN_CACHE_PRIMITIVES
			bwdData->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDataDstMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
			dnnl::inner_product_backward_data(*bwdDataDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDataDstMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
			Device.stream.wait();

			if (reorderBwdDataDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
				Device.stream.wait();
			}

			if (SharesInputInplace)
			{
#ifdef DNN_CACHE_PRIMITIVES
				bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) } });
#else
				dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) } });
#endif
				Device.stream.wait();
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN		
		}

		ByteArray GetImage(const Byte fillColor) final override
		{
			if (BiasCount > 0)
			{
				const auto rangeWeights = GetColorRange<Float>(WeightsStats.Min, WeightsStats.Max);
				const auto rangeBiases = GetColorRange<Float>(BiasesStats.Min, BiasesStats.Max);

				FloatVector weights;
				if (*WeightsMemDesc != *PersistWeightsMemDesc)
				{
					weights = FloatVector(WeightsMemDesc->get_size() / sizeof(Float));

					auto memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
					auto weightsMem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weights.data());

					dnnl::reorder(memWeights, weightsMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
					Device.stream.wait();
				}
				else
					weights = Weights;

				const auto width = BiasCount;
				const auto height = WeightCount / BiasCount;
				const auto totalSize = width * (height + 3);

				auto image = ByteArray(totalSize, fillColor);

				for (auto y = 0ull; y < height; y++)
				{
					const auto start = y * width;
					const auto end = start + width;
					for (auto x = start; x < end; x++)
						image[x] = GetColorFromRange<Float>(rangeWeights, WeightsStats.Min, weights[x]);
				}

				if (HasBias)
				{
					const auto offset = (height + 1) * width;
					for (auto x = 0ull; x < width; x++)
						image[x + offset] = GetColorFromRange<Float>(rangeBiases, BiasesStats.Min, Biases[x]);
				}

				return image;
			}
			else
				return ByteArray();
		}
	};
}