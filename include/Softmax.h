#pragma once
#include "Layer.h"

namespace dnn
{
	class Softmax final : public Layer
	{
	private:
		std::unique_ptr<dnnl::softmax_v2_forward::primitive_desc> fwdDescSoftmax;
		std::unique_ptr<dnnl::softmax_v2_backward::primitive_desc> bwdDescSoftmax;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::softmax_v2_forward> fwdSoftmax;
		std::unique_ptr<dnnl::softmax_v2_backward> bwdSoftmax;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		Softmax(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs) :
			Layer(device, format, name, LayerTypes::Softmax, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);
		}

		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			return GetDescriptionHeader();
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
			if (InputLayer->DstMemDesc->data.ndims == 2)
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

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.engine));
#ifdef DNN_CACHE_PRIMITIVES
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif

			const auto axis = (H == 1 && W == 1) ? 1 : 3;
			fwdDescSoftmax = std::make_unique<dnnl::softmax_v2_forward::primitive_desc>(dnnl::softmax_v2_forward::primitive_desc(dnnl::softmax_v2_forward::desc(dnnl::prop_kind::forward, dnnl::algorithm::softmax_accurate, *InputLayer->DstMemDesc, *DstMemDesc, axis), Device.engine));
			bwdDescSoftmax = std::make_unique<dnnl::softmax_v2_backward::primitive_desc>(dnnl::softmax_v2_backward::primitive_desc(dnnl::softmax_v2_backward::desc(dnnl::algorithm::softmax_accurate, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, *DstMemDesc, axis), Device.engine, *fwdDescSoftmax));
			reorderBwdSrc = bwdDescSoftmax->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDescSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
#ifdef DNN_CACHE_PRIMITIVES
			fwdSoftmax = std::make_unique<dnnl::softmax_v2_forward>(dnnl::softmax_v2_forward(*fwdDescSoftmax));
			bwdSoftmax = std::make_unique<dnnl::softmax_v2_backward>(dnnl::softmax_v2_backward(*bwdDescSoftmax));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW() : batchSize * PaddedCDHW();
			const auto threads = GetThreads(elements);
			const auto strideHW = HW() * VectorSize;
			const auto vecZero = VecFloat(0);

			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			
			auto dstMem = dnnl::memory(fwdDescSoftmax->dst_desc(), Device.engine, Neurons.data());
#ifdef DNN_CACHE_PRIMITIVES
			fwdSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, memSrc}, { DNNL_ARG_DST, dstMem } });
#else
			dnnl::softmax_v2_forward(*fwdDescSoftmax).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, memSrc}, { DNNL_ARG_DST, dstMem } });
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

			auto dstMem = dnnl::memory(bwdDescSoftmax->dst_desc(), Device.engine, Neurons.data());
			auto diffDstMem = dnnl::memory(bwdDescSoftmax->diff_dst_desc(), Device.engine, NeuronsD1.data());

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescSoftmax->diff_src_desc(), Device.engine) : memDiffSrc;

#ifdef DNN_CACHE_PRIMITIVES
			bwdSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
			dnnl::softmax_v2_backward(*bwdDescSoftmax).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, {DNNL_ARG_DIFF_DST, diffDstMem}, {DNNL_ARG_DIFF_SRC, diffSrcMem} });
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