#pragma once
#include "Layer.h"

namespace dnn
{
	class LocalResponseNormalization final : public Layer
	{
	private:
		std::unique_ptr<dnnl::lrn_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::lrn_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
		std::unique_ptr<dnnl::lrn_forward> fwd;
		std::unique_ptr<dnnl::lrn_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
		std::unique_ptr<dnnl::memory> WorkspaceMemory;
		dnnl::algorithm algorithm;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const bool AcrossChannels;
		const size_t LocalSize;
		const Float Alpha;
		const Float Beta;
		const Float K;

		LocalResponseNormalization(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool acrossChannels = false, const size_t localSize = 5, const Float alpha = Float(1), const Float beta = Float(5), const Float k = Float(1)) :
			Layer(device, format, name, LayerTypes::LocalResponseNormalization, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			AcrossChannels(acrossChannels),
			LocalSize(localSize),
			Alpha(alpha),
			Beta(beta),
			K(k),
			algorithm(dnnl::algorithm::lrn_within_channel),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);
		}

		std::string GetDescription() const final override
		{
			std::string description = GetDescriptionHeader();

			description.append(nwl + std::string(" AcrossChannels:") + tab + (AcrossChannels ? std::string("Yes") : std::string("No")));
			description.append(nwl + std::string(" LocalSize:") + tab + std::to_string(LocalSize));
			description.append(nwl + std::string(" Alpha:") + dtab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:") + dtab + FloatToString(Beta));
			description.append(nwl + std::string(" K:") + dtab + FloatToString(K));

			return description;
		}

		size_t FanIn() const final override
		{
			return 1;
		}

		size_t FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
			}
			else
			{
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, GetDataFmt(*InputLayer->DstMemDesc)));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, GetDataFmt(*InputLayer->DiffDstMemDesc)));
			}

			algorithm = AcrossChannels ? dnnl::algorithm::lrn_across_channels : dnnl::algorithm::lrn_within_channel;

			fwdDesc = std::make_unique<dnnl::lrn_forward::primitive_desc>(dnnl::lrn_forward::primitive_desc(dnnl::lrn_forward::desc(dnnl::prop_kind::forward, algorithm, *InputLayer->DstMemDesc, LocalSize, Alpha, Beta, K), Device.first));
			bwdDesc = std::make_unique<dnnl::lrn_backward::primitive_desc>(dnnl::lrn_backward::primitive_desc(dnnl::lrn_backward::desc(algorithm, *DiffDstMemDesc, *DstMemDesc, LocalSize, Alpha, Beta, K), Device.first, *fwdDesc));

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.first));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));

			WorkspaceMemory = std::make_unique<dnnl::memory>(dnnl::memory(fwdDesc->workspace_desc(), Device.first));

			fwd = std::make_unique<dnnl::lrn_forward>(dnnl::lrn_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::lrn_backward>(dnnl::lrn_backward(*bwdDesc));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;
		}

		void ForwardProp(const size_t batchSize, const bool training)  final override
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory> { {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory } });
				Device.second.wait();
			}

			auto dstMem = dnnl::memory(fwdDesc->dst_desc(), Device.first, Neurons.data());

			fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
			Device.second.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
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

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.first) : memDiffSrc;

			bwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_WORKSPACE, *WorkspaceMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
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
	};
}