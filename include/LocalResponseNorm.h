#pragma once
#include "Layer.h"

namespace dnn
{
	class LocalResponseNorm final : public Layer
	{
	private:
		std::unique_ptr<dnnl::lrn_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::lrn_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
		std::unique_ptr<dnnl::memory> workspaceMemory;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::lrn_forward> fwd;
		std::unique_ptr<dnnl::lrn_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const bool AcrossChannels;
		const dnnl::algorithm Algorithm;
		const UInt LocalSize;
		const Float Alpha;
		const Float Beta;
		const Float K;

		LocalResponseNorm(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool acrossChannels = false, const UInt localSize = 5, const Float alpha = Float(1), const Float beta = Float(5), const Float k = Float(1)) :
			Layer(device, format, name, LayerTypes::LocalResponseNorm, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			AcrossChannels(acrossChannels),
			Algorithm(acrossChannels ? dnnl::algorithm::lrn_across_channels : dnnl::algorithm::lrn_within_channel),
			LocalSize(localSize),
			Alpha(alpha),
			Beta(beta),
			K(k),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);

			FwdZeroGradient = Float(1);
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" AcrossChannels:") + tab + (AcrossChannels ? std::string("Yes") : std::string("No")));
			description.append(nwl + std::string(" LocalSize:") + tab + std::to_string(LocalSize));
			description.append(nwl + std::string(" Alpha:") + dtab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:") + dtab + FloatToString(Beta));
			description.append(nwl + std::string(" K:") + dtab + FloatToString(K));

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
			
			fwdDesc = std::make_unique<dnnl::lrn_forward::primitive_desc>(dnnl::lrn_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, Algorithm, *InputLayer->DstMemDesc, *DstMemDesc, LocalSize, Alpha, Beta, K));
			bwdDesc = std::make_unique<dnnl::lrn_backward::primitive_desc>(dnnl::lrn_backward::primitive_desc(Device.engine, Algorithm, *InputLayerBwd->DiffDstMemDesc, *DiffDstMemDesc, *InputLayer->DstMemDesc, LocalSize, Alpha, Beta, K, *fwdDesc));
			workspaceMemory = std::make_unique<dnnl::memory>(dnnl::memory(fwdDesc->workspace_desc(), Device.engine));
			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayerBwd->DiffDstMemDesc, *InputLayerBwd->DiffDstMemDesc, *InputLayerBwd->DiffDstMemDesc));
						
			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayerBwd->DiffDstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::lrn_forward>(dnnl::lrn_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::lrn_backward>(dnnl::lrn_backward(*bwdDesc));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training)  final override
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory> { {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem }, { DNNL_ARG_WORKSPACE, *workspaceMemory } });
				Device.stream.wait();
			}

			auto dstMem = dnnl::memory(fwdDesc->dst_desc(), Device.engine, Neurons.data());
#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
#else
			dnnl::lrn_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
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

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

#ifdef DNN_CACHE_PRIMITIVES
			bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_WORKSPACE, *workspaceMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
			dnnl::lrn_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_WORKSPACE, *workspaceMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
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