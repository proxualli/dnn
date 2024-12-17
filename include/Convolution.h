#pragma once
#include "Layer.h"

namespace dnn
{
	class Convolution final : public Layer
	{
	private:
		std::unique_ptr<dnnl::convolution_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::convolution_backward_weights::primitive_desc> bwdWeightsDesc;
		std::unique_ptr<dnnl::convolution_backward_data::primitive_desc> bwdDataDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::convolution_forward> fwd;
		std::unique_ptr<dnnl::convolution_backward_weights> bwdWeights;
		std::unique_ptr<dnnl::convolution_backward_data> bwdData;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		bool reorderFwdSrc;
		bool reorderBwdWeightsSrc;
		bool reorderBwdWeightsDiffWeights;
		bool reorderBwdDataDiffSrc;
		bool reorderBwdDataWeights;
		bool reorderBwdDataDiffDst;
		
	public:
		const UInt Groups;
		const UInt KernelH;
		const UInt KernelW;
		const UInt StrideH;
		const UInt StrideW;
		const UInt DilationH;
		const UInt DilationW;
		const UInt DilationKernelH;
		const UInt DilationKernelW;
		const dnnl::memory::dims Strides;
		const dnnl::memory::dims Dilates;
		const dnnl::memory::dims Padding;

		Convolution(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const UInt c, const UInt kernelH, const UInt kernelW, const UInt strideH, const UInt strideW, const UInt dilationH, const UInt dilationW, const UInt padH, const UInt padW, const UInt groups, const bool hasBias) :
			Layer(device, format, name, LayerTypes::Convolution, groups * (inputs[0]->C / groups) * (c / groups) * kernelH * kernelW, c, c, inputs[0]->D, (((inputs[0]->H - (1 + (kernelH - 1) * dilationH)) + (padH * 2)) / strideH) + 1, (((inputs[0]->W - (1 + (kernelW - 1) * dilationW)) + (padW * 2)) / strideW) + 1, 0, padH, padW, inputs, hasBias),
			Groups(groups),
			KernelH(kernelH),
			KernelW(kernelW),
			StrideH(strideH),
			StrideW(strideW),
			DilationH(dilationH),
			DilationW(dilationW),
			DilationKernelH(1 + (kernelH - 1) * dilationH),
			DilationKernelW(1 + (kernelW - 1) * dilationW),
			Strides(dnnl::memory::dims({ dnnl::memory::dim(strideH) , dnnl::memory::dim(strideW) })),
			Dilates(dnnl::memory::dims({ dnnl::memory::dim(dilationH - 1), dnnl::memory::dim(dilationW - 1) })),
			Padding(dnnl::memory::dims({ dnnl::memory::dim(padH), dnnl::memory::dim(padW) })),
			reorderFwdSrc(false),
			reorderBwdWeightsSrc(false),
			reorderBwdWeightsDiffWeights(false),
			reorderBwdDataDiffSrc(false),
			reorderBwdDataWeights(false),
			reorderBwdDataDiffDst(false)			
		{
			assert(Inputs.size() == 1);

			assert(Groups > 0 && Groups <= InputLayer->C && Groups <= C);

			if (Groups > 1)
			{
				PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(Groups), dnnl::memory::dim(C / Groups), dnnl::memory::dim(InputLayer->C / Groups), dnnl::memory::dim(KernelH), dnnl::memory::dim(KernelW) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::goihw));
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(Groups), dnnl::memory::dim(C / Groups), dnnl::memory::dim(InputLayer->C / Groups), dnnl::memory::dim(KernelH), dnnl::memory::dim(KernelW) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::goihw));
			}
			else
			{
				PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(KernelH), dnnl::memory::dim(KernelW) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw));
				WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(KernelH), dnnl::memory::dim(KernelW) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw));
			}
		}

		void UpdateResolution() final override
		{
			H = (((InputLayer->H - DilationKernelH) + (PadH * 2)) / StrideH) + 1;
			W = (((InputLayer->W - DilationKernelW) + (PadW * 2)) / StrideW) + 1;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			if (Groups > 1)
				description.append(nwl + std::string(" Groups:     ") + tab + std::to_string(Groups));
			if (DilationH == 1 && DilationW == 1)
				description.append(nwl + std::string(" Kernel:     ") + tab + std::to_string(KernelH) + std::string("x") + std::to_string(KernelW));
			else
			{
				description.append(nwl + std::string(" Dilates:    ") + tab + std::to_string(DilationH) + std::string("x") + std::to_string(DilationW));
				description.append(nwl + std::string(" Kernel:     ") + tab + std::to_string(DilationKernelH) + std::string("x") + std::to_string(DilationKernelW));
			}
			if (StrideH * StrideW > 1)
				description.append(nwl + std::string(" Stride:     ") + tab + std::to_string(StrideH) + std::string("x") + std::to_string(StrideW));
			if (HasPadding)
				description.append(nwl + std::string(" Padding:    ") + tab + std::to_string(PadH) + std::string("x") + std::to_string(PadW));
			    
			description.append(GetWeightsDescription());

			description.append(nwl + std::string(" Connections:") + tab + std::to_string(Groups * (InputLayer->C / Groups) * (C / Groups) * (H / StrideW) * (W / StrideH) * (HasBias ? KernelH * KernelW + 1 : KernelH * KernelW)));

			return description;
		}

		UInt FanIn() const  final override
		{
			return InputLayer->C / Groups * KernelH * KernelW;
		}

		UInt FanOut() const final override
		{
			return C / Groups * KernelH * KernelW / StrideH * StrideW;
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			std::vector<dnnl::memory::desc> memDesc;

			if (Groups > 1)
			{
				memDesc = std::vector<dnnl::memory::desc>({
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, NeuronsFormat),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, NeuronsFormat),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(Groups), dnnl::memory::dim(C / Groups), dnnl::memory::dim(InputLayer->C / Groups), dnnl::memory::dim(KernelH), dnnl::memory::dim(KernelW) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any) });
			}
			else
			{
				memDesc = std::vector<dnnl::memory::desc>({
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, NeuronsFormat),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, NeuronsFormat),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(KernelH), dnnl::memory::dim(KernelW) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any),
					dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any) });
			}
						
			fwdDesc = std::make_unique<dnnl::convolution_forward::primitive_desc>(HasBias ? 
				dnnl::convolution_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, dnnl::algorithm::convolution_auto, memDesc[0], memDesc[2], memDesc[3], memDesc[1], Strides, Dilates, Padding, Padding) :
				dnnl::convolution_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, dnnl::algorithm::convolution_auto, memDesc[0], memDesc[2], memDesc[1], Strides, Dilates, Padding, Padding));
			
			bwdWeightsDesc = std::make_unique<dnnl::convolution_backward_weights::primitive_desc>(HasBias ? 
				dnnl::convolution_backward_weights::primitive_desc(Device.engine, dnnl::algorithm::convolution_auto, memDesc[0], memDesc[2], memDesc[3], memDesc[1], Strides, Dilates, Padding, Padding, *fwdDesc) :
				dnnl::convolution_backward_weights::primitive_desc(Device.engine, dnnl::algorithm::convolution_auto, memDesc[0], memDesc[2], memDesc[1], Strides, Dilates, Padding, Padding, *fwdDesc));

			bwdDataDesc = std::make_unique<dnnl::convolution_backward_data::primitive_desc>(dnnl::convolution_backward_data::primitive_desc(Device.engine, dnnl::algorithm::convolution_auto, memDesc[0], memDesc[2], memDesc[1], Strides, Dilates, Padding, Padding, *fwdDesc));
			
			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));

			if (*WeightsMemDesc != fwdDesc->weights_desc())
			{
				auto memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());

				auto weights = FloatVector(fwdDesc->weights_desc().get_size() / sizeof(Float));
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

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdWeightsSrc = bwdWeightsDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdWeightsDiffWeights = bwdWeightsDesc->diff_weights_desc() != *WeightsMemDesc;
			reorderBwdDataDiffSrc = bwdDataDesc->diff_src_desc() != *InputLayerBwd->DiffDstMemDesc;
			reorderBwdDataWeights = bwdDataDesc->weights_desc() != *WeightsMemDesc;
			reorderBwdDataDiffDst = bwdDataDesc->diff_dst_desc() != *DiffDstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::convolution_forward>(dnnl::convolution_forward(*fwdDesc));
			bwdWeights = std::make_unique<dnnl::convolution_backward_weights>(dnnl::convolution_backward_weights(*bwdWeightsDesc));
			bwdData = std::make_unique<dnnl::convolution_backward_data>(dnnl::convolution_backward_data(*bwdDataDesc));
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

			const auto& weightsMem = dnnl::memory(fwdDesc->weights_desc(), Device.engine, Weights.data());
			auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());

#ifdef DNN_CACHE_PRIMITIVES
			HasBias ? fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_BIAS, dnnl::memory(fwdDesc->bias_desc(), Device.engine, Biases.data()) }, { DNNL_ARG_DST, dstMem } }) :
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#else
			HasBias ? dnnl::convolution_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_BIAS, dnnl::memory(fwdDesc->bias_desc(), Device.engine, Biases.data()) }, { DNNL_ARG_DST, dstMem } }) :
				dnnl::convolution_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN			
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
				bwdWeights->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, memDiffDst}, { DNNL_ARG_SRC, srcMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_BIAS, dnnl::memory(bwdWeightsDesc->diff_bias_desc(), Device.engine, BiasesD1.data()) } }) :
				bwdWeights->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, memDiffDst}, { DNNL_ARG_SRC, srcMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem } });
#else
			HasBias ?
				dnnl::convolution_backward_weights(*bwdWeightsDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, memDiffDst}, { DNNL_ARG_SRC, srcMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_BIAS, dnnl::memory(bwdWeightsDesc->diff_bias_desc(), Device.engine, BiasesD1.data()) } }) :
				dnnl::convolution_backward_weights(*bwdWeightsDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, memDiffDst}, { DNNL_ARG_SRC, srcMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem } });
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
			dnnl::convolution_backward_data(*bwdDataDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDataDstMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
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
			const auto rangeWeights = GetColorRange<Float>(WeightsStats.Min, WeightsStats.Max);
			const auto rangeBiases = GetColorRange<Float>(BiasesStats.Min, BiasesStats.Max);

			auto weights = FloatVector();
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

			if (Groups > 1)
			{
				const auto border = 1ull;
				const auto pitchH = KernelH + border;
				const auto pitchW = KernelW + border;
				const auto width = C * pitchH + border;
				const auto height = (InputLayer->C / Groups) * pitchW + border;
				const auto biasOffset = height * width;
				auto image = ByteArray(biasOffset + width, fillColor);

				for (auto g = 0ull; g < Groups; g++)
				{
					for (auto c = 0ull; c < (C / Groups); c++)
					{
						const auto left = (g * (C / Groups) + c) * pitchH + border;
						for (auto r = 0ull; r < (InputLayer->C / Groups); r++)
						{
							const auto top = r * pitchW + border;
							const auto idx = ((g * (C / Groups) + c) * (InputLayer->C / Groups) + r) * KernelH * KernelW;
							for (auto y = 0ull; y < KernelH; y++)
								for (auto x = 0ull; x < KernelW; x++)
									image[((top + y) * width) + left + x] = GetColorFromRange<Float>(rangeWeights, WeightsStats.Min, weights[idx + (y * KernelW) + x]);
						}
						if (HasBias)
							image[left + biasOffset] = GetColorFromRange<Float>(rangeBiases, BiasesStats.Min, Biases[g * (C / Groups) + c]);
					}
				}

				return image;
			}
			else
			{
				if (InputLayer->C != 3)
				{
					const auto border = (KernelH == 1ull && KernelW == 1ull) ? 0ull : 1ull;
					const auto pitchH = KernelH + border;
					const auto pitchW = KernelW + border;
					const auto width = C * pitchH + border;
					const auto height = InputLayer->C * pitchW + border;
					const auto biasOffset = height * width;
					auto image = ByteArray(biasOffset + width, fillColor);

					for (auto c = 0ull; c < C; c++)
					{
						const auto left = c * pitchH + border;
						for (auto r = 0ull; r < InputLayer->C; r++)
						{
							const auto top = r * pitchW + border;
							const auto idx = (c * InputLayer->C + r) * KernelH * KernelW;
							for (auto y = 0ull; y < KernelH; y++)
								for (auto x = 0ull; x < KernelW; x++)
									image[((top + y) * width) + left + x] = GetColorFromRange<Float>(rangeWeights, WeightsStats.Min, weights[idx + (y * KernelW) + x]);
						}
						if (HasBias)
							image[left + biasOffset] = GetColorFromRange<Float>(rangeBiases, BiasesStats.Min, Biases[c]);
					}

					return image;
				}
				else
				{
					const auto border = 1ull;
					const auto pitchH = KernelH + border;
					const auto pitchW = KernelW + border;
					const auto width = C * pitchH + border;
					const auto height = pitchW + 3 * border;
					const auto size = width * height;
					const auto totalSize = 3 * size;
					auto image = ByteArray(totalSize, fillColor);

					auto mapping = 0ull;
					for (auto c = 0ull; c < C; c++)
					{
						const auto mapOffset = c * pitchH + border;
						for (auto inputC = 0ull; inputC < 3; inputC++)
						{
							const auto channelOffset = inputC * size;
							const auto mapIndex = mapping * KernelH * KernelW;

							for (auto y = 0ull; y < KernelW; y++)
								for (auto x = 0ull; x < KernelH; x++)
									image[x + mapOffset + ((border + y) * width) + channelOffset] = GetColorFromRange<Float>(rangeWeights, WeightsStats.Min, weights[x + (y * KernelW) + mapIndex]);
								
							if (HasBias)
								image[mapOffset + ((1ull + pitchW) * width) + channelOffset] = GetColorFromRange<Float>(rangeBiases, BiasesStats.Min, Biases[c]);

							mapping++;
						}
					}

					auto temp = ByteArray(totalSize);
					for (auto i = 0ull; i < size; i++)
					{
						temp[0 + (3 * i)] = image[i + (0 * size)];
						temp[1 + (3 * i)] = image[i + (1 * size)];
						temp[2 + (3 * i)] = image[i + (2 * size)];
					}
					
					image.release();

					return temp;
				}
			}
		}
	};
}