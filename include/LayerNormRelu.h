#pragma once
#include "Layer.h"

namespace dnn
{
	class LayerNormRelu final : public Layer
	{
	private:
		dnnl::normalization_flags Flags;
		std::unique_ptr<dnnl::layer_normalization_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::layer_normalization_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
		std::unique_ptr<dnnl::memory::desc> StatsDesc;
		std::unique_ptr<dnnl::memory> WorkspaceMemory;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::layer_normalization_forward> fwd;
		std::unique_ptr<dnnl::layer_normalization_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		FloatVector ScaleShift;
		FloatVector DiffScaleShift;

		bool inference;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const Float Eps;
		FloatVector Mean;
		FloatVector Variance;

		LayerNormRelu(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool scaling = true, const Float eps = Float(1e-06), const bool hasBias = true) :
			Layer(device, format, name, LayerTypes::LayerNormRelu, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias, scaling),
			Eps(eps),
			Flags(dnnl::normalization_flags::fuse_norm_relu),
			inference(false),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);

			Mean = FloatVector(1, Float(0));
			Variance = FloatVector(1, Float(1));

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


		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader() + GetWeightsDescription(Scaling);

			description.append(nwl + std::string(" Eps:") + dtab + FloatToStringScientific(Eps));

			auto mean = Float(0);
			auto variance = Float(0);
			PRAGMA_OMP_SIMD()
				for (auto e = 0ull; e < Mean.size(); e++)
				{
					mean += Mean[e];
					variance += Variance[e];
				}
			mean /= Mean.size();
			variance /= Mean.size();

			description.append(nwl + std::string(" Mean:") + dtab + FloatToStringFixed(mean));
			description.append(nwl + std::string(" Variance:") + tab + FloatToStringFixed(variance));

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
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::ab;

				StatsDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::a));
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				ChosenFormat = PlainFmt;

				StatsDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc));
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			if (inference)
				Flags = Scaling ? dnnl::normalization_flags::fuse_norm_relu | dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift : dnnl::normalization_flags::fuse_norm_relu | dnnl::normalization_flags::use_global_stats;
			else
				Flags = Scaling ? dnnl::normalization_flags::fuse_norm_relu | dnnl::normalization_flags::use_scale_shift : dnnl::normalization_flags::fuse_norm_relu;

			fwdDesc = std::make_unique<dnnl::layer_normalization_forward::primitive_desc>(dnnl::layer_normalization_forward::primitive_desc(dnnl::layer_normalization_forward::desc(inference ? dnnl::prop_kind::forward_inference : dnnl::prop_kind::forward_training, *DstMemDesc, *StatsDesc, Eps, Flags), Device.engine));

			Mean.resize(fwdDesc->mean_desc().get_size() / sizeof(Float), Float(0));
			Variance.resize(fwdDesc->variance_desc().get_size() / sizeof(Float), Float(1));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::layer_normalization_forward>(dnnl::layer_normalization_forward(*fwdDesc));
#endif
			if (!inference)
			{
				WorkspaceMemory = std::make_unique<dnnl::memory>(dnnl::memory(fwdDesc->workspace_desc(), Device.engine));

				bwdDesc = std::make_unique<dnnl::layer_normalization_backward::primitive_desc>(dnnl::layer_normalization_backward::primitive_desc(dnnl::layer_normalization_backward::desc(Scaling ? dnnl::prop_kind::backward : dnnl::prop_kind::backward_data, *DiffDstMemDesc, *DstMemDesc, *StatsDesc, Eps, Flags), Device.engine, *fwdDesc));

				reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

				bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.engine));

#ifdef DNN_CACHE_PRIMITIVES
				bwd = std::make_unique<dnnl::layer_normalization_backward>(dnnl::layer_normalization_backward(*bwdDesc));
				bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
			}
		}

		bool Lockable() const final override
		{
			return WeightCount > 0 && Scaling;
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			if (!training)
			{
				if (!inference)
				{
					inference = true;
					InitializeDescriptors(batchSize);
				}

				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				const auto& srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto memMean = dnnl::memory(fwdDesc->mean_desc(), Device.engine, Mean.data());
				auto memVariance = dnnl::memory(fwdDesc->variance_desc(), Device.engine, Variance.data());
				auto dstMem = dnnl::memory(fwdDesc->dst_desc(), Device.engine, Neurons.data());

				if (Scaling)
				{
					for (auto c = 0ull; c < C; c++)
					{
						ScaleShift[c] = Weights[c];
						ScaleShift[PaddedC + c] = Biases[c];
					}
					auto memScaleShift = dnnl::memory(*WeightsMemDesc, Device.engine, ScaleShift.data());
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE_SHIFT, memScaleShift }, { DNNL_ARG_DST, dstMem } });
#endif
					dnnl::layer_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE_SHIFT, memScaleShift }, { DNNL_ARG_DST, dstMem } });
				}
				else
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem } });
#else
					dnnl::layer_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem } });
#endif

				Device.stream.wait();
			}
			else
			{
				if (inference)
				{
					inference = false;
					InitializeDescriptors(batchSize);
				}

				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				const auto& srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto memMean = dnnl::memory(fwdDesc->mean_desc(), Device.engine, Mean.data());
				auto memVariance = dnnl::memory(fwdDesc->variance_desc(), Device.engine, Variance.data());
				auto dstMem = dnnl::memory(fwdDesc->dst_desc(), Device.engine, Neurons.data());

				if (Scaling)
				{
					for (auto c = 0ull; c < C; c++)
					{
						ScaleShift[c] = Weights[c];
						ScaleShift[PaddedC + c] = Biases[c];
					}
					auto memScaleShift = dnnl::memory(*WeightsMemDesc, Device.engine, ScaleShift.data());
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE_SHIFT, memScaleShift }, { DNNL_ARG_DST, dstMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory }});
#else
					dnnl::layer_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE_SHIFT, memScaleShift }, { DNNL_ARG_DST, dstMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory }});
#endif
				}
				else
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory }});
#else
					dnnl::layer_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory } });
#endif
				Device.stream.wait();

#ifndef DNN_LEAN
				if (!InplaceBwd)
					InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW);
#else
				DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN

			auto memSrc = dnnl::memory(*InputsOriginal[0]->DstMemDesc, Device.engine, InputsOriginal[0]->Neurons.data());
			const auto& srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.engine) : memSrc;
			if (reorderBwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto memMean = dnnl::memory(bwdDesc->mean_desc(), Device.engine, Mean.data());
			auto memVariance = dnnl::memory(bwdDesc->variance_desc(), Device.engine, Variance.data());

			const auto& memDiffSrc = SharesInput && !InplaceBwd ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			const auto& diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;
			if (Scaling)
			{
				for (auto c = 0ull; c < 2 * PaddedC; c++)
					DiffScaleShift[c] = Float(0);

				auto scaleShiftMemory = dnnl::memory(*WeightsMemDesc, Device.engine, ScaleShift.data());
				auto diffScaleShiftMemory = dnnl::memory(*WeightsMemDesc, Device.engine, DiffScaleShift.data());
#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory> { {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE_SHIFT, scaleShiftMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_DIFF_SCALE_SHIFT, diffScaleShiftMemory }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory } });
#else
				dnnl::layer_normalization_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory> { {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE_SHIFT, scaleShiftMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_DIFF_SCALE_SHIFT, diffScaleShiftMemory }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory } });
#endif

				for (auto c = 0ull; c < C; c++)
				{
					WeightsD1[c] += DiffScaleShift[c];
					BiasesD1[c] += DiffScaleShift[PaddedC + c];
				}
			}
			else
#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory } });
#else
				dnnl::layer_normalization_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_WORKSPACE, *WorkspaceMemory } });
#endif
			Device.stream.wait();

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
				Device.stream.wait();
			}

			if (SharesInput && !InplaceBwd)
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

		ByteArray GetImage(const Byte fillColor) final override
		{
			if (Scaling)
			{
				const auto rangeWeights = GetColorRange(WeightsStats.Min, WeightsStats.Max);
				const auto rangeBiases = GetColorRange(BiasesStats.Min, BiasesStats.Max);

				const auto width = BiasCount;
				const auto height = WeightCount / BiasCount;
				const auto totalSize = width * (height + 3);

				auto image = ByteArray(totalSize, fillColor);

				for (auto y = 0ull; y < height; y++)
				{
					const auto start = y * width;
					const auto end = start + width;
					for (auto x = start; x < end; x++)
						image[x] = GetColorFromRange(rangeWeights, WeightsStats.Min, Weights[x]);
				}

				if (HasBias)
				{
					const auto offset = (height + 1) * width;
					for (auto x = 0ull; x < width; x++)
						image[x + offset] = GetColorFromRange(rangeBiases, BiasesStats.Min, Biases[x]);
				}

				return image;
			}
			else
				return ByteArray();
		}

		void ResetWeights(const Fillers weightFiller, const Float weightFillerScale, const Fillers biasFiller, const Float biasFillerScale) override
		{
			Weights.resize(PaddedC); std::fill(Weights.begin(), Weights.end(), Float(1));
			Biases.resize(PaddedC); std::fill(Biases.begin(), Biases.end(), Float(0));

			DNN_UNREF_PAR(weightFiller);
			DNN_UNREF_PAR(weightFillerScale);
			DNN_UNREF_PAR(biasFiller);
			DNN_UNREF_PAR(biasFillerScale);
		}

		void Save(std::ostream& os, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			if (Scaling)
			{
				for (auto c = 0ull; c < C; c++)
				{
					Weights[c] = ScaleShift[c];
					Biases[c] = ScaleShift[C + c];
				}
			}

			Layer::Save(os, persistOptimizer, optimizer);
		}

		void Load(std::istream& is, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			Layer::Load(is, persistOptimizer, optimizer);

			if (Scaling)
			{
				for (auto c = 0ull; c < C; c++)
				{
					ScaleShift[c] = Weights[c];
					ScaleShift[C + c] = Biases[c];
				}
			}
		}
	};
}