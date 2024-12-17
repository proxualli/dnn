#pragma once
#include "Layer.h"

namespace dnn
{
	class MaxPooling final : public Layer
	{
	private:
		std::unique_ptr<dnnl::pooling_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::pooling_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
		std::unique_ptr<dnnl::memory> workspaceMemory;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::pooling_forward> fwd;
		std::unique_ptr<dnnl::pooling_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		bool reorderFwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const UInt KernelH;
		const UInt KernelW;
		const UInt StrideH;
		const UInt StrideW;
		const UInt DilationH;
		const UInt DilationW;
		const UInt DilationKernelH;
		const UInt DilationKernelW;
		const dnnl::memory::dims Kernel;
		const dnnl::memory::dims Strides;
		const dnnl::memory::dims Dilation;
		const dnnl::memory::dims Padding;
				
		MaxPooling(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const UInt kernelH = 2, const UInt kernelW = 2, const UInt strideH = 2, const UInt strideW = 2, const UInt dilationH = 1, const UInt dilationW = 1, const UInt padH = 0, const UInt padW = 0) :
			Layer(device, format, name, LayerTypes::MaxPooling, 0, 0, inputs[0]->C, inputs[0]->D, (((inputs[0]->H - kernelH) + (padH * 2)) / strideH) + 1, (((inputs[0]->W - kernelW) + (padW * 2)) / strideW) + 1, 0, padH, padW, inputs),
			KernelH(kernelH),
			KernelW(kernelW),
			StrideH(strideH),
			StrideW(strideW),
			DilationH(dilationH),
			DilationW(dilationW),
			DilationKernelH(1 + (kernelH - 1) * dilationH),
			DilationKernelW(1 + (kernelW - 1) * dilationW),
			Kernel(dnnl::memory::dims({ dnnl::memory::dim(kernelH), dnnl::memory::dim(kernelW) })),
			Strides(dnnl::memory::dims({ dnnl::memory::dim(strideH), dnnl::memory::dim(strideW) })),
			Dilation(dnnl::memory::dims({ dnnl::memory::dim(dilationH - 1), dnnl::memory::dim(dilationW - 1) })),
			Padding(dnnl::memory::dims({ dnnl::memory::dim(padH), dnnl::memory::dim(padW) })),
			reorderFwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);
		}

		void UpdateResolution() final override
		{
			H = (((InputLayer->H - (1 + (KernelH - 1) * DilationH)) + (Padding[0] * 2)) / StrideH) + 1;
			W = (((InputLayer->W - (1 + (KernelW - 1) * DilationW)) + (Padding[1] * 2)) / StrideW) + 1;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			if (DilationH == 1 && DilationW == 1)
				description.append(nwl + std::string(" Kernel:") + tab + std::to_string(KernelH) + std::string("x") + std::to_string(KernelW));
			else
			{
				description.append(nwl + std::string(" Dilates:") + tab + std::to_string(DilationH) + std::string("x") + std::to_string(DilationW));
				description.append(nwl + std::string(" Kernel:") + tab + std::to_string(DilationKernelH) + std::string("x") + std::to_string(DilationKernelW));
			}

			description.append(nwl + std::string(" Stride:") + tab + std::to_string(StrideH) + std::string("x") + std::to_string(StrideW));
			if (HasPadding)
				description.append(nwl + std::string(" Padding:") + tab + std::to_string(PadH) + std::string("x") + std::to_string(PadW));

			return description;
		}

		UInt FanIn() const final override
		{
			return StrideH * StrideW;
		}

		UInt FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const UInt batchSize) final override
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
					ChosenFormat = GetMemoryFormat(*InputLayer->DstMemDesc);
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			
			fwdDesc = std::make_unique<dnnl::pooling_forward::primitive_desc>(dnnl::pooling_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, dnnl::algorithm::pooling_max, *InputLayer->DstMemDesc, *DstMemDesc, Strides, Kernel, Dilation, Padding, Padding));
			bwdDesc = std::make_unique<dnnl::pooling_backward::primitive_desc>(dnnl::pooling_backward::primitive_desc(Device.engine, dnnl::algorithm::pooling_max, *InputLayerBwd->DiffDstMemDesc, *DiffDstMemDesc, Strides, Kernel, Dilation, Padding, Padding, *fwdDesc));

			workspaceMemory = std::make_unique<dnnl::memory>(dnnl::memory(fwdDesc->workspace_desc(), Device.engine));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayerBwd->DiffDstMemDesc;

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayerBwd->DiffDstMemDesc, *InputLayerBwd->DiffDstMemDesc, *InputLayerBwd->DiffDstMemDesc));

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::pooling_forward>(dnnl::pooling_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::pooling_backward>(dnnl::pooling_backward(*bwdDesc));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training)  final override
		{
			const auto& memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem }, { DNNL_ARG_WORKSPACE, *workspaceMemory } });
#else
			dnnl::pooling_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem }, { DNNL_ARG_WORKSPACE, *workspaceMemory } });
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

			const auto& diffDstMem = dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data());

			auto memDiffSrc = SharesInputInplace ? dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

#ifdef DNN_CACHE_PRIMITIVES
			bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory> { {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_WORKSPACE, *workspaceMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
			dnnl::pooling_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory> { {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_WORKSPACE, *workspaceMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
			Device.stream.wait();

			if (reorderBwdDiffSrc)
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
	};
}