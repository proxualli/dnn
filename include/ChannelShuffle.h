#pragma once
#include "Layer.h"

namespace dnn
{
	class ChannelShuffle final : public Layer
	{
	private:
		std::unique_ptr<dnnl::shuffle_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::shuffle_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::shuffle_forward> fwd;
		std::unique_ptr<dnnl::shuffle_backward> bwd;

	public:
	    const size_t Groups;
		const size_t GroupSize;

		ChannelShuffle(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const size_t groups) :
			Layer(device, format, name, LayerTypes::ChannelShuffle, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs),
			Groups(groups),
			GroupSize(inputs[0]->C / groups)
		{
			assert(Inputs.size() == 1);

			assert(Groups > 0 && Groups <= C);
		}

		std::string ChannelShuffle::GetDescription() const final override
		{
			std::string description = GetDescriptionHeader();

			description.append(nwl + " Groups:" + tab + std::to_string(Groups));
			description.append(nwl + " GroupSize:" + tab + std::to_string(GroupSize));
			description.append(nwl + " Connections:" + tab + std::to_string(InputLayer->C / Groups));

			return description;
		}

		size_t ChannelShuffle::FanIn() const final override
		{
			return 1;
		}

		size_t ChannelShuffle::FanOut() const final override
		{
			return 1;
		}

		void ChannelShuffle::InitializeDescriptors(const size_t batchSize) final override
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

			fwdDesc = std::make_unique<dnnl::shuffle_forward::primitive_desc>(dnnl::shuffle_forward::primitive_desc(dnnl::shuffle_forward::desc(dnnl::prop_kind::forward_training, *DstMemDesc, 1, int(GroupSize)), Device.first));
			bwdDesc = std::make_unique<dnnl::shuffle_backward::primitive_desc>(dnnl::shuffle_backward::primitive_desc(dnnl::shuffle_backward::desc(*DiffDstMemDesc, 1, int(GroupSize)), Device.first, *fwdDesc));

			fwd = std::make_unique<dnnl::shuffle_forward>(dnnl::shuffle_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::shuffle_backward>(dnnl::shuffle_backward(*bwdDesc));
		}

		void ChannelShuffle::ForwardProp(const size_t batchSize, const bool training) final override
		{
			auto srcMem = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto dstMem = dnnl::memory(*DstMemDesc, Device.first, Neurons.data());

			fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
			Device.second.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN
		}

		void ChannelShuffle::BackwardProp(const size_t batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN

			auto diffDstMem = dnnl::memory(*DiffDstMemDesc, Device.first, NeuronsD1.data());
			auto diffSrcMem = dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());

			bwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
			Device.second.wait();

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}