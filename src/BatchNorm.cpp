#include "Model.h"

namespace dnn
{
	BatchNorm::BatchNorm(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool scaling, const Float momentum, const Float eps, const bool hasBias) :
		Layer(device, format, name, LayerTypes::BatchNorm, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias),
		Scaling(scaling),
		Eps(eps),
		Momentum(momentum),
		OneMinusMomentum(Float(1) - momentum),
		Flags(static_cast<dnnl::normalization_flags>(0U)),
		inference(false),
		reorderFwdSrc(false),
		reorderBwdSrc(false),
		reorderBwdDiffSrc(false)
	{
		assert(Inputs.size() == 1);

		Mean = FloatVector(PaddedC, Float(0));
		RunningMean = FloatVector(PaddedC, Float(0));
		Variance = FloatVector(PaddedC, Float(1));
		RunningVariance = FloatVector(PaddedC, Float(1));
		InvStdDev = FloatVector(PaddedC);

		if (Scaling)
		{
			ScaleShift = FloatVector(2 * PaddedC, Float(1));
			for (auto c = 0ull; c < PaddedC; c++)
				ScaleShift[PaddedC + c] = Float(0);

			DiffScaleShift = FloatVector(2 * PaddedC, Float(0));
		}

		WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 2, dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
		PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 2, dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
	}

	std::string BatchNorm::GetDescription() const
	{
		std::string description = GetDescriptionHeader() + GetWeightsDescription(Scaling);

		description.append(nwl + " Momentum:" + tab + FloatToString(Momentum));
		description.append(nwl + " Eps:" + dtab + FloatToStringScientific(Eps));
		
		auto mean = Float(0);
		auto variance = Float(0);
		for (auto c = 0ull; c < C; c++)
		{
			mean += RunningMean[c];
			variance += RunningVariance[c];
		}
		mean /= C;
		variance /= C;

		description.append(nwl + " Mean:" + dtab + FloatToStringFixed(mean));
		description.append(nwl + " Variance:" + tab + FloatToStringFixed(variance));

		return description;
	}

	size_t BatchNorm::FanIn() const
	{
		return 1;
	}

	size_t BatchNorm::FanOut() const
	{
		return 1;
	}

	void BatchNorm::InitializeDescriptors(const size_t batchSize)
	{
		if (InputLayer->DstMemDesc->data.ndims == 2)
		{
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));

			plainFormat = true;
		}
		else
		{
			if (Format == dnnl::memory::format_tag::any)
			{
				Format = GetDataFmt(*InputLayer->DstMemDesc);
				if (Format != GetDataFmt(*InputLayer->DiffDstMemDesc))
					throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
			}

			plainFormat = (Format == dnnl::memory::format_tag::nchw);

			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
		}

		if (inference)
			Flags = Scaling ? dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift : dnnl::normalization_flags::use_global_stats;
		else
			Flags = Scaling ? dnnl::normalization_flags::use_scale_shift : static_cast<dnnl::normalization_flags>(0U);
		
		fwdDesc = std::make_unique<dnnl::batch_normalization_forward::primitive_desc>(dnnl::batch_normalization_forward::primitive_desc(dnnl::batch_normalization_forward::desc(inference ? dnnl::prop_kind::forward_inference : dnnl::prop_kind::forward_training, *DstMemDesc, Eps, Flags), Device.first));
		
		reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
		
		fwd = std::make_unique<dnnl::batch_normalization_forward>(dnnl::batch_normalization_forward(*fwdDesc));

		if (!inference)
		{
			bwdDesc = std::make_unique<dnnl::batch_normalization_backward::primitive_desc>(dnnl::batch_normalization_backward::primitive_desc(dnnl::batch_normalization_backward::desc(Scaling ? dnnl::prop_kind::backward : dnnl::prop_kind::backward_data, *DiffDstMemDesc, *DstMemDesc, Eps, Flags), Device.first, *fwdDesc));
		
			reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

			bwd = std::make_unique<dnnl::batch_normalization_backward>(dnnl::batch_normalization_backward(*bwdDesc));
		}

		bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.first));
		bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
	}

	void BatchNorm::ForwardProp(const size_t batchSize, const bool training)
	{
		if (!training)
		{
			if (!inference)
			{
				inference = true;
				InitializeDescriptors(batchSize);
			}

			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
				Device.second.wait();
			}

			auto memMean = dnnl::memory(fwdDesc->mean_desc(), Device.first, RunningMean.data());
			auto memVariance = dnnl::memory(fwdDesc->variance_desc(), Device.first, RunningVariance.data());

			auto dstMem = dnnl::memory(*DstMemDesc, Device.first, Neurons.data());
			
			if (Scaling)
			{
				for (auto c = 0ull; c < C; c++)
				{
					ScaleShift[c] = Weights[c];
					ScaleShift[PaddedC + c] = Biases[c];
				}
				auto memScaleShift = dnnl::memory(*WeightsMemDesc, Device.first, ScaleShift.data());
				fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_MEAN, memMean}, {DNNL_ARG_VARIANCE, memVariance}, {DNNL_ARG_SCALE_SHIFT, memScaleShift}, {DNNL_ARG_DST, dstMem} });
			}
			else
				fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_MEAN, memMean}, {DNNL_ARG_VARIANCE, memVariance}, {DNNL_ARG_DST, dstMem} });

			Device.second.wait();
		}
		else
		{
			if (inference)
			{
				inference = false;
				InitializeDescriptors(batchSize);
			}

			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
				Device.second.wait();
			}

			auto memMean = dnnl::memory(fwdDesc->mean_desc(), Device.first, Mean.data());
			auto memVariance = dnnl::memory(fwdDesc->variance_desc(), Device.first, Variance.data());

			auto dstMem = dnnl::memory(*DstMemDesc, Device.first, Neurons.data());

			if (Scaling)
			{
				for (auto c = 0ull; c < C; c++)
				{
					ScaleShift[c] = Weights[c];
					ScaleShift[PaddedC + c] = Biases[c];
				}
				auto memScaleShift = dnnl::memory(*WeightsMemDesc, Device.first, ScaleShift.data());

				fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_MEAN, memMean}, {DNNL_ARG_VARIANCE, memVariance}, {DNNL_ARG_SCALE_SHIFT, memScaleShift}, {DNNL_ARG_DST, dstMem} });
			}
			else

			fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_MEAN, memMean}, {DNNL_ARG_VARIANCE, memVariance}, {DNNL_ARG_DST, dstMem} });
			Device.second.wait();

			const Float unbiasedFactor = Float(batchSize * HW) / Float(batchSize * HW - 1);
			for (size_t c = 0; c < C; c++)
			{
				RunningMean[c] = (Momentum * RunningMean[c]) + (OneMinusMomentum * Mean[c]);
				RunningVariance[c] = (Momentum * RunningVariance[c]) + (OneMinusMomentum * Variance[c] * unbiasedFactor);
			}

#ifndef DNN_LEAN
			ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN
			}
	}

	void BatchNorm::BackwardProp(const size_t batchSize)
	{
#ifdef DNN_LEAN
		ZeroGradient(batchSize);
#else
		DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN

		auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
		auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.first) : memSrc;
		if (reorderBwdSrc)
		{
			dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
			Device.second.wait();
		}
				
		auto diffDstMem = dnnl::memory(*DiffDstMemDesc, Device.first, NeuronsD1.data());
		
		auto memMean = dnnl::memory(bwdDesc->mean_desc(), Device.first, Mean.data());
		auto memVariance = dnnl::memory(bwdDesc->variance_desc(), Device.first, Variance.data());
		
		auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());
		auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.first) : memDiffSrc;

		if (Scaling)
		{
			for (auto c = 0ull; c < 2 * PaddedC; c++)
				DiffScaleShift[c] = Float(0);

			auto scaleShiftMemory = dnnl::memory(*WeightsMemDesc, Device.first, ScaleShift.data());
			auto diffScaleShiftMemory = dnnl::memory(*WeightsMemDesc, Device.first, DiffScaleShift.data());

			bwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_DIFF_DST, diffDstMem}, {DNNL_ARG_MEAN, memMean}, {DNNL_ARG_VARIANCE, memVariance}, {DNNL_ARG_SCALE_SHIFT, scaleShiftMemory},  {DNNL_ARG_DIFF_SRC, diffSrcMem}, {DNNL_ARG_DIFF_SCALE_SHIFT, diffScaleShiftMemory} });
			
			for (auto c = 0ull; c < C; c++)
			{
				WeightsD1[c] += DiffScaleShift[c];
				BiasesD1[c] += DiffScaleShift[PaddedC + c];
			}
		}
		else
			bwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_DIFF_DST, diffDstMem}, {DNNL_ARG_MEAN, memMean}, {DNNL_ARG_VARIANCE, memVariance}, {DNNL_ARG_DIFF_SRC, diffSrcMem} });

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

	ByteVector BatchNorm::GetImage(const Byte fillColor) 
	{
		if (Scaling)
		{
			const auto rangeWeights = GetColorRange(WeightsMin, WeightsMax);
			const auto rangeBiases = GetColorRange(BiasesMin, BiasesMax);

			const auto width = BiasCount;
			const auto height = WeightCount / BiasCount;
			const auto totalSize = width * (height + 3);

			auto image = ByteVector(totalSize, fillColor);

			for (auto y = 0ull; y < height; y++)
			{
				const auto start = y * width;
				const auto end = start + width;
				for (auto x = start; x < end; x++)
					image[x] = GetColorFromRange(rangeWeights, WeightsMin, Weights[x]);
			}

			if (HasBias)
			{
				const auto offset = (height + 1) * width;
				for (auto x = 0ull; x < width; x++)
					image[x + offset] = GetColorFromRange(rangeBiases, BiasesMin, Biases[x]);
			}

			return image;
		}
		else
			return ByteVector();
	}
}
