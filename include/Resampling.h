#pragma once
#include "Layer.h"

namespace dnn
{
	enum class Algorithms
	{
		Linear = 0,
		Nearest = 1
	};

	class Resampling final : public Layer
	{
	private:
		std::unique_ptr<dnnl::resampling_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::resampling_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::resampling_forward> fwd;
		std::unique_ptr<dnnl::resampling_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif

	public:
		const Algorithms Algorithm;
		const Float FactorH;
		const Float FactorW;

		Resampling(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Algorithms algorithm, const Float factorH, const Float factorW) :
			Layer(device, format, name, LayerTypes::Resampling, 0, 0, inputs[0]->C, inputs[0]->D, static_cast<UInt>(inputs[0]->H * double(factorH)), static_cast<UInt>(inputs[0]->W * double(factorW)), 0, 0, 0, inputs),
			Algorithm(algorithm),
			FactorH(factorH),
			FactorW(factorW)
		{
			assert(Inputs.size() == 1);
		}

		void UpdateResolution() final override
		{
			H = static_cast<UInt>(std::trunc(double(InputLayer->H) * double(FactorH)));
			W = static_cast<UInt>(std::trunc(double(InputLayer->W) * double(FactorW)));
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Scaling:") + dtab + FloatToString(FactorH, 4) + std::string("x") + FloatToString(FactorW, 4));
			description.append(nwl + std::string(" Algorithm:  ") + tab + std::string(magic_enum::enum_name<Algorithms>(Algorithm)));
			
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
			dnnl::algorithm algorithm;
			switch (Algorithm)
			{
			case Algorithms::Linear:
				algorithm = dnnl::algorithm::resampling_linear;
				break;
			case Algorithms::Nearest:
				algorithm = dnnl::algorithm::resampling_nearest;
				break;
			default:
				algorithm = dnnl::algorithm::resampling_linear;
			}

			const auto factor = std::vector<float>({ FactorH, FactorW });

			auto memDesc = std::vector<dnnl::memory::desc>({
				dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(InputLayer->C), dnnl::memory::dim(InputLayer->H), dnnl::memory::dim(InputLayer->W) }), dnnl::memory::data_type::f32, NeuronsFormat),
				dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, NeuronsFormat) });

			fwdDesc = std::make_unique<dnnl::resampling_forward::primitive_desc>(dnnl::resampling_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, algorithm, factor, *InputLayer->DstMemDesc, memDesc[1]));
			
			DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDesc->dst_desc());

			if (NeuronsFormat == dnnl::memory::format_tag::any)
				ChosenFormat = GetMemoryFormat(*DstMemDesc);
			else
				ChosenFormat = PlainFmt;

			bwdDesc = std::make_unique<dnnl::resampling_backward::primitive_desc>(dnnl::resampling_backward::primitive_desc(Device.engine, algorithm, factor, memDesc[0], *DiffDstMemDesc, *fwdDesc));
			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::resampling_forward>(dnnl::resampling_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::resampling_backward>(dnnl::resampling_backward(*bwdDesc));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto& memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());

#ifdef DNN_CACHE_PRIMITIVES
			fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, memSrc}, { DNNL_ARG_DST, dstMem } });
#else
			dnnl::resampling_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, memSrc}, { DNNL_ARG_DST, dstMem } });
#endif
			Device.stream.wait();

#ifndef DNN_LEAN
			if (training)
				InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
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

			const auto& diffDstMem = dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data());
			auto memDiffSrc = SharesInputInplace ? dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data());

#ifdef DNN_CACHE_PRIMITIVES
			bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_DIFF_SRC, memDiffSrc } });
#else
			dnnl::resampling_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DIFF_DST, diffDstMem}, { DNNL_ARG_DIFF_SRC, memDiffSrc } });
#endif
			Device.stream.wait();

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