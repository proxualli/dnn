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
		std::unique_ptr<dnnl::inner_product_forward> fwd;
		std::unique_ptr<dnnl::inner_product_backward_weights> bwdWeights;
		std::unique_ptr<dnnl::inner_product_backward_data> bwdData;
		std::unique_ptr<dnnl::binary> bwdAdd;

		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
		bool reorderBwdWeights;
		bool reorderBwdDiffWeights;

	public:
		Dense(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const size_t c, const std::vector<Layer*>& inputs, const bool hasBias) :
			Layer(device, format, name, LayerTypes::Dense, c* inputs[0]->CDHW, c, c, 1, 1, 1, 0, 0, 0, inputs, hasBias),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false),
			reorderBwdWeights(false),
			reorderBwdDiffWeights(false)
		{
			assert(Inputs.size() == 1);

			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oi));
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oi));
			}
			else
			{
				PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw));
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw));
			}
		}

		std::string GetDescription() const final override
		{
			std::string description = GetDescriptionHeader() + GetWeightsDescription(true);

			description.append(nwl + " Connections:" + tab + std::to_string(CDHW * (InputLayer->CDHW + 1)));

			return description;
		}

		size_t FanIn() const final override
		{
			return InputLayer->CDHW;
		}

		size_t FanOut() const final override
		{
			return CDHW;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			std::vector<dnnl::memory::desc> memDesc;
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				memDesc = std::vector<dnnl::memory::desc>({
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C)}), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x) });
			}
			else
			{
				memDesc = std::vector<dnnl::memory::desc>({
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, Format),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x) });
			}

			fwdDesc = std::make_unique<dnnl::inner_product_forward::primitive_desc>(dnnl::inner_product_forward::primitive_desc(HasBias ?
				dnnl::inner_product_forward::desc(dnnl::prop_kind::forward, memDesc[0], memDesc[2], memDesc[3], memDesc[1]) :
				dnnl::inner_product_forward::desc(dnnl::prop_kind::forward, memDesc[0], memDesc[2], memDesc[1]), Device.first));

			bwdWeightsDesc = std::make_unique<dnnl::inner_product_backward_weights::primitive_desc>(dnnl::inner_product_backward_weights::primitive_desc(HasBias ?
				dnnl::inner_product_backward_weights::desc(memDesc[0], memDesc[2], memDesc[3], memDesc[1]) :
				dnnl::inner_product_backward_weights::desc(memDesc[0], memDesc[2], memDesc[1]), Device.first, *fwdDesc));

			bwdDataDesc = std::make_unique<dnnl::inner_product_backward_data::primitive_desc>(dnnl::inner_product_backward_data::primitive_desc(dnnl::inner_product_backward_data::desc(memDesc[0], memDesc[2], memDesc[1]), Device.first, *fwdDesc));

			if (*WeightsMemDesc != fwdDesc->weights_desc())
			{
				auto weights = FloatVector(fwdDesc->weights_desc().get_size(), Float(0));
				auto memWeights = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());
				auto weightsMem = dnnl::memory(fwdDesc->weights_desc(), Device.first, weights.data());

				dnnl::reorder(memWeights, weightsMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
				Device.second.wait();

				Weights = weights;
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->weights_desc());
			}

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.first));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdWeightsDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDataDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;
			reorderBwdWeights = bwdDataDesc->weights_desc() != *WeightsMemDesc;
			reorderBwdDiffWeights = bwdWeightsDesc->diff_weights_desc() != *WeightsMemDesc;

			fwd = std::make_unique<dnnl::inner_product_forward>(dnnl::inner_product_forward(*fwdDesc));
			bwdWeights = std::make_unique<dnnl::inner_product_backward_weights>(dnnl::inner_product_backward_weights(*bwdWeightsDesc));
			bwdData = std::make_unique<dnnl::inner_product_backward_data>(dnnl::inner_product_backward_data(*bwdDataDesc));
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.second.wait();
			}

			auto weightsMem = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());

			auto dstMem = dnnl::memory(*DstMemDesc, Device.first, Neurons.data());

			HasBias ?
				fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_BIAS, dnnl::memory(fwdDesc->bias_desc(), Device.first, Biases.data()) }, { DNNL_ARG_DST, dstMem } }) :
				fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });

			Device.second.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
#endif		
		}

		void BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN

			auto diffDstMem = dnnl::memory(*DiffDstMemDesc, Device.first, NeuronsD1.data());

			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderBwdSrc ? dnnl::memory(bwdWeightsDesc->src_desc(), Device.first) : memSrc;
			if (reorderBwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.second.wait();
			}

			auto memDiffWeights = dnnl::memory(*WeightsMemDesc, Device.first, WeightsD1.data());
			auto diffWeightsMem = reorderBwdDiffWeights ? dnnl::memory(bwdWeightsDesc->diff_weights_desc(), Device.first) : memDiffWeights;

			HasBias ?
				bwdWeights->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_BIAS, dnnl::memory(bwdWeightsDesc->diff_bias_desc(), Device.first, BiasesD1.data()) } }) :
				bwdWeights->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem } });

			Device.second.wait();

			if (reorderBwdDiffWeights)
			{
				dnnl::reorder(diffWeightsMem, memDiffWeights).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffWeightsMem}, { DNNL_ARG_TO, memDiffWeights } });
				Device.second.wait();
			}

			auto memWeights = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());
			auto weightsMem = reorderBwdWeights ? dnnl::memory(bwdDataDesc->weights_desc(), Device.first) : memWeights;
			if (reorderBwdWeights)
			{
				dnnl::reorder(memWeights, weightsMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
				Device.second.wait();
			}

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDataDesc->diff_src_desc(), Device.first) : memDiffSrc;

			bwdData->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
			Device.second.wait();

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
				Device.second.wait();
			}

			if (SharesInput)
			{
				bwdAdd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) } });
				Device.second.wait();
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN		
		}

		ByteVector GetImage(const Byte fillColor) final override
		{
			const auto rangeWeights = GetColorRange(WeightsMin, WeightsMax);
			const auto rangeBiases = GetColorRange(BiasesMin, BiasesMax);

			FloatVector weights;
			if (*WeightsMemDesc != *PersistWeightsMemDesc)
			{
				weights = FloatVector(WeightsMemDesc->get_size());

				auto memWeights = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());
				auto weightsMem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weights.data());

				dnnl::reorder(memWeights, weightsMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
				Device.second.wait();
			}
			else
				weights = Weights;

			const auto width = BiasCount;
			const auto height = WeightCount / BiasCount;
			const auto totalSize = width * (height + 3);

			auto image = ByteVector(totalSize, fillColor);

			for (auto y = 0ull; y < height; y++)
			{
				const auto start = y * width;
				const auto end = start + width;
				for (auto x = start; x < end; x++)
					image[x] = GetColorFromRange(rangeWeights, WeightsMin, weights[x]);
			}

			if (HasBias)
			{
				const auto offset = (height + 1) * width;
				for (auto x = 0ull; x < width; x++)
					image[x + offset] = GetColorFromRange(rangeBiases, BiasesMin, Biases[x]);
			}

			return image;
		}
	};
}
