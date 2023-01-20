#pragma once
#include "Layer.h"

namespace dnn
{
	class PRelu final : public Layer
	{
	private:
		std::unique_ptr<dnnl::prelu_forward::primitive_desc> fwdDescPRelu;
		std::unique_ptr<dnnl::prelu_backward::primitive_desc> bwdDescPRelu;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::prelu_forward> fwdPRelu;
		std::unique_ptr<dnnl::prelu_backward> bwdPRelu;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
		bool reorderBwdDiffWeights;

	public:
		const Float Alpha;
		
		PRelu(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float alpha = Float(0.25)) :
			Layer(device, format, name, LayerTypes::PRelu, inputs[0]->C , inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, true),
			Alpha(alpha),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false),
			reorderBwdDiffWeights(false)
		{
			assert(Inputs.size() == 1);

			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 1, dnnl::memory::dim(C), 1, 1 }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw));
			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 1, dnnl::memory::dim(C), 1, 1 }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw));
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Alpha:") + dtab + FloatToString(Alpha));
		
			return description;
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
			if (InputLayer->DstMemDesc->get_ndims() == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
					ChosenFormat = LayerBeforeCost || IsPlainDataFmt(*InputLayer->DstMemDesc) ? PlainFmt : GetDataFmt(*InputLayer->DstMemDesc);
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));
#ifdef DNN_CACHE_PRIMITIVES
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
			
			auto memDesc = dnnl::memory::desc(dnnl::memory::dims({ 1, dnnl::memory::dim(C), 1, 1 }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

			fwdDescPRelu = std::make_unique<dnnl::prelu_forward::primitive_desc>(dnnl::prelu_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward,*InputLayer->DstMemDesc, memDesc, *DstMemDesc));
			bwdDescPRelu = std::make_unique<dnnl::prelu_backward::primitive_desc>(dnnl::prelu_backward::primitive_desc(Device.engine, *InputLayer->DstMemDesc, memDesc , *InputLayer->DiffDstMemDesc, memDesc, *DiffDstMemDesc, *fwdDescPRelu));

			if (*WeightsMemDesc != fwdDescPRelu->weights_desc())
			{
				auto memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Biases.data());
				
				auto weights = FloatVector(fwdDescPRelu->weights_desc().get_size() / sizeof(Float));
				auto weightsMem = dnnl::memory(fwdDescPRelu->weights_desc(), Device.engine, weights.data());

				dnnl::reorder(memWeights, weightsMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
				Device.stream.wait();

				Biases = weights;
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescPRelu->weights_desc());
			}

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescPRelu->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescPRelu->dst_desc());

			ChosenFormat = GetDataFmt(*DstMemDesc);

			reorderFwdSrc = fwdDescPRelu->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdDescPRelu->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDescPRelu->diff_src_desc() != *InputLayer->DiffDstMemDesc;
			reorderBwdDiffWeights = bwdDescPRelu->diff_weights_desc() != fwdDescPRelu->weights_desc();

#ifdef DNN_CACHE_PRIMITIVES
			fwdPRelu = std::make_unique<dnnl::prelu_forward>(dnnl::prelu_forward(*fwdDescPRelu));
			bwdPRelu = std::make_unique<dnnl::prelu_backward>(dnnl::prelu_backward(*bwdDescPRelu));
#endif
		}

		ByteArray GetImage(const Byte fillColor) final override
		{
			if (HasWeights)
			{
				const auto rangeWeights = GetColorRange<Float>(BiasesStats.Min, BiasesStats.Max);

				const auto width = BiasCount;
				const auto height = 1;
				const auto totalSize = width * (height + 3);

				auto image = ByteArray(totalSize, fillColor);

				for (auto y = 0ull; y < height; y++)
				{
					const auto start = y * width;
					const auto end = start + width;
					for (auto x = start; x < end; x++)
						image[x] = GetColorFromRange<Float>(rangeWeights, BiasesStats.Min, Biases[x]);
				}

				return image;
			}
			else
				return ByteArray();
		}

		void ResetWeights(const Fillers weightsFiller, const FillerModes weightsFillerMode, const Float weightsGain, const Float weightsFillerScale, const Fillers biasesFiller, const FillerModes biasesFillerMode, const Float biasesGain, const Float biasesFillerScale) override
		{
			if (HasWeights)
				Biases = FloatVector(PaddedC, Float(Alpha));

			DNN_UNREF_PAR(weightsFiller);
			DNN_UNREF_PAR(weightsFillerMode);
			DNN_UNREF_PAR(weightsGain);
			DNN_UNREF_PAR(weightsFillerScale);
			DNN_UNREF_PAR(biasesFiller);
			DNN_UNREF_PAR(biasesFillerMode);
			DNN_UNREF_PAR(biasesGain);
			DNN_UNREF_PAR(biasesFillerScale);
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDescPRelu->src_desc(), Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto weightsMem = dnnl::memory(fwdDescPRelu->weights_desc(), Device.engine, Biases.data());

			auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());
#ifdef DNN_CACHE_PRIMITIVES
			fwdPRelu->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#else
			dnnl::prelu_forward(*fwdDescPRelu).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#endif
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW() : batchSize * PaddedCDHW();
			const auto threads = GetThreads(elements);
			const auto strideHW = HW() * VectorSize;

			auto dstMem = dnnl::memory(bwdDescPRelu->dst_desc(), Device.engine, Neurons.data());
			auto diffDstMem = dnnl::memory(bwdDescPRelu->diff_dst_desc(), Device.engine, NeuronsD1.data());

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescPRelu->diff_src_desc(), Device.engine) : memDiffSrc;

			auto memSrc = dnnl::memory(*InputLayerFwd->DstMemDesc, Device.engine, InputLayerFwd->Neurons.data());
			auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDescPRelu->src_desc(), Device.engine) : memSrc;
			if (reorderBwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto memDiffWeights = dnnl::memory(*WeightsMemDesc, Device.engine, BiasesD1.data());
			auto diffWeightsMem = reorderBwdDiffWeights ? dnnl::memory(bwdDescPRelu->diff_weights_desc(), Device.engine) : memDiffWeights;

			auto weightsMem = dnnl::memory(bwdDescPRelu->weights_desc(), Device.engine, Biases.data());
#ifdef DNN_CACHE_PRIMITIVES
			bwdPRelu->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
			dnnl::prelu_backward(*bwdDescPRelu).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
			Device.stream.wait();

			if (reorderBwdDiffWeights)
			{
				dnnl::reorder(diffWeightsMem, memDiffWeights).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffWeightsMem}, { DNNL_ARG_TO, memDiffWeights } });
				Device.stream.wait();
			}

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
				Device.stream.wait();
			}

			if (SharesInput)
			{
#ifdef DNN_CACHE_PRIMITIVES
				bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#else
				dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#endif
				Device.stream.wait();
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}