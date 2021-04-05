#pragma once
#include "Layer.h"

namespace dnn
{
	class AvgPooling final : public Layer
	{
	private:
		const dnnl::memory::dims Kernel;
		const dnnl::memory::dims Stride;
		const dnnl::memory::dims Padding;

		std::unique_ptr<dnnl::pooling_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::pooling_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
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
		const Float Scale;

		AvgPooling(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const UInt kernelH = 2, const UInt kernelW = 2, const UInt strideH = 2, const UInt strideW = 2, const UInt padH = 0, const UInt padW = 0) :
			Layer(device, format, name, LayerTypes::AvgPooling, 0, 0, inputs[0]->C, inputs[0]->D, (((inputs[0]->H - kernelH) + (padH * 2)) / strideH) + 1, (((inputs[0]->W - kernelW) + (padW * 2)) / strideW) + 1, 0, padH, padW, inputs),
			KernelH(kernelH),
			KernelW(kernelW),
			StrideH(strideH),
			StrideW(strideW),
			Scale(Float(1) / (kernelH * kernelW)),
			Kernel(dnnl::memory::dims({ dnnl::memory::dim(kernelH), dnnl::memory::dim(kernelW) })),
			Stride(dnnl::memory::dims({ dnnl::memory::dim(strideH) , dnnl::memory::dim(strideW) })),
			Padding(dnnl::memory::dims({ dnnl::memory::dim(padH), dnnl::memory::dim(padW) })),
			reorderFwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Kernel:") + tab + std::to_string(KernelH) + std::string("x") + std::to_string(KernelW));
			description.append(nwl + std::string(" Stride:") + tab + std::to_string(StrideH) + std::string("x") + std::to_string(StrideW));
			if (HasPadding)
				description.append(nwl + std::string(" Padding:") + tab + std::to_string(PadH) + std::string("x") + std::to_string(PadW));
			description.append(nwl + std::string(" Scale:") + dtab + FloatToString(Scale));

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
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				chosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
					chosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
				else
					chosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}

			if (HasPadding)
			{
				fwdDesc = std::make_unique<dnnl::pooling_forward::primitive_desc>(dnnl::pooling_forward::primitive_desc(dnnl::pooling_forward::desc(dnnl::prop_kind::forward, dnnl::algorithm::pooling_avg_include_padding, *InputLayer->DstMemDesc, *DstMemDesc, Stride, Kernel, Padding, Padding), Device.engine));
				bwdDesc = std::make_unique<dnnl::pooling_backward::primitive_desc>(dnnl::pooling_backward::primitive_desc(dnnl::pooling_backward::desc(dnnl::algorithm::pooling_avg_include_padding, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, Stride, Kernel, Padding, Padding), Device.engine, *fwdDesc));
			}
			else
			{
				fwdDesc = std::make_unique<dnnl::pooling_forward::primitive_desc>(dnnl::pooling_forward::primitive_desc(dnnl::pooling_forward::desc(dnnl::prop_kind::forward, dnnl::algorithm::pooling_avg_exclude_padding, *InputLayer->DstMemDesc, *DstMemDesc, Stride, Kernel, Padding, Padding), Device.engine));
				bwdDesc = std::make_unique<dnnl::pooling_backward::primitive_desc>(dnnl::pooling_backward::primitive_desc(dnnl::pooling_backward::desc(dnnl::algorithm::pooling_avg_exclude_padding, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, Stride, Kernel, Padding, Padding), Device.engine, *fwdDesc));
			}

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(bwdDesc->diff_dst_desc());

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.engine));

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::pooling_forward>(dnnl::pooling_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::pooling_backward>(dnnl::pooling_backward(*bwdDesc));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			const auto &srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());

#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, { {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_DST, dstMem} });
#else
			dnnl::pooling_forward(*fwdDesc).execute(Device.stream, { {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_DST, dstMem} });
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
			DNN_UNREF_PAR(training);
#endif
		}

		void BackwardProp(const UInt batchSize) final override 
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN

			auto diffDstMem = dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data());

			const auto &memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			const auto &diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

#ifdef DNN_CACHE_PRIMITIVES
			bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
			dnnl::pooling_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
			Device.stream.wait();

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