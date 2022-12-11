#pragma once
#include "Activation.h"

namespace dnn
{
	template <typename Activation = HardSwish, typename dnn::LayerTypes T = LayerTypes::BatchNormHardSwish>
	class BatchNormActivation final : public Layer
	{
	private:
		dnnl::normalization_flags flags;
		std::unique_ptr<dnnl::batch_normalization_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::batch_normalization_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::batch_normalization_forward> fwd;
		std::unique_ptr<dnnl::batch_normalization_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		FloatVector scale;
		FloatVector shift;
		FloatVector diffScale;
		FloatVector diffShift;
		bool inference;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;
		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;
		FloatArray InputNeurons;

		BatchNormActivation<Activation,T>(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool scaling = true, const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true) :
			Layer(device, format, name, T, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias, scaling),
			Eps(eps),
			Momentum(momentum),
			OneMinusMomentum(Float(1) - momentum),
			Mean(FloatVector(PaddedC, Float(0))),
			RunningMean(FloatVector(PaddedC, Float(0))),
			Variance(FloatVector(PaddedC, Float(1))),
			RunningVariance(FloatVector(PaddedC, Float(1))),
			InvStdDev(FloatVector(PaddedC)),
			InputNeurons(FloatArray())
		{
			assert(Inputs.size() == 1);

			if (Scaling)
			{
				scale = FloatVector(PaddedC, Float(1));
				shift = FloatVector(PaddedC, Float(0));
				diffScale = FloatVector(PaddedC, Float(0));
				diffShift = FloatVector(PaddedC, Float(0));
			}

			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x));
			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x));
		}
	
		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader() + GetWeightsDescription(Scaling);

			description.append(nwl + std::string(" Momentum:") + tab + FloatToString(Momentum));
			description.append(nwl + std::string(" Eps:") + dtab + FloatToStringScientific(Eps));

			auto mean = Float(0);
			auto variance = Float(0);
			for (auto c = 0ull; c < C; c++)
			{
				mean += RunningMean[c];
				variance += RunningVariance[c];
			}
			mean /= C;
			variance /= C;

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

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			if constexpr (Reference)
				InputNeurons.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			if (InputLayer->DstMemDesc->get_ndims() == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					ChosenFormat = GetDataFmt(*InputLayer->DstMemDesc);
					if (ChosenFormat != GetDataFmt(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument(std::string("Src and Diff format are different in ") + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + std::string(" layer ") + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));

				if (inference)
					flags = Scaling ? dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift : dnnl::normalization_flags::use_global_stats;
				else
					flags = Scaling ? dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift : static_cast<dnnl::normalization_flags>(0U);

				/*dnnl::post_ops batchnorm_ops;
				const auto alpha = 3.f;
				const auto beta = 0.1666666667f;
				batchnorm_ops.append_eltwise(dnnl::algorithm::eltwise_hardswish, alpha, beta);
				dnnl::primitive_attr batchnorm_attr;
				batchnorm_attr.set_post_ops(batchnorm_ops);*/

				fwdDesc = std::make_unique<dnnl::batch_normalization_forward::primitive_desc>(dnnl::batch_normalization_forward::primitive_desc(Device.engine, inference ? dnnl::prop_kind::forward_inference : dnnl::prop_kind::forward_training, *InputLayer->DstMemDesc, *DstMemDesc, Eps, flags));

				reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
				fwd = std::make_unique<dnnl::batch_normalization_forward>(dnnl::batch_normalization_forward(*fwdDesc));
#endif
				if (!inference)
				{
					bwdDesc = std::make_unique<dnnl::batch_normalization_backward::primitive_desc>(dnnl::batch_normalization_backward::primitive_desc(Device.engine, Scaling ? dnnl::prop_kind::backward : dnnl::prop_kind::backward_data, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, *DstMemDesc, Eps, flags, *fwdDesc));

					reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
					reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

					bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));

#ifdef DNN_CACHE_PRIMITIVES
					bwd = std::make_unique<dnnl::batch_normalization_backward>(dnnl::batch_normalization_backward(*bwdDesc));
					bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
				}
			}
		}

		bool Lockable() const final override
		{
			return WeightCount > 0 && Scaling;
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{				
			if constexpr (Reference)
				ForwardPropRef(batchSize, training);
			else
			{
				const auto strideH = W * VectorSize;
				const auto plain = IsPlainFormat();
				const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());

				if (!training)
				{
					const auto threads = GetThreads(elements, Float(0.2));

					if (plain) // nchw
					{
						const auto partialHW = GetVectorPart(HW());

						for_i(C, threads, [=](UInt c)
							{
								const auto invStdDev = Float(1) / std::sqrt(RunningVariance[c] + Eps);
						const auto weightedInvStdDev = Scaling ? invStdDev * Weights[c] : invStdDev;
						const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW() + (n * CDHW());
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
								Activation::fVec((VecFloat().load_a(&InputLayer->Neurons[hw]) - RunningMean[c]) * weightedInvStdDev + biases).store_a(&Neurons[hw]);
							for (auto hw = part; hw < start + HW(); hw++)
								Neurons[hw] = Activation::f(((InputLayer->Neurons[hw]) - RunningMean[c]) * weightedInvStdDev + biases);
						}
							});
					}
					else
					{
						for_i(PaddedC / VectorSize, threads, [=](UInt c)
						{
							const auto channelOffset = c * VectorSize;
							const auto mapOffset = channelOffset * HW();

							const auto runningMean = VecFloat().load_a(&RunningMean[channelOffset]);
							const auto invStdDev = VecFloat(Float(1)) / sqrt(VecFloat().load_a(&RunningVariance[channelOffset]) + Eps);

							const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]) : invStdDev;
							const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto offsetC = n * PaddedCDHW() + mapOffset;
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - runningMean, weightedInvStdDev, biases)).store_a(&Neurons[w]);
								}
							}
						});
					}
				}
				else
				{
					const auto threads = GetThreads(elements, Float(0.1));

#ifndef DNN_LEAN
					const auto vecZero = VecFloat(0);
#endif
					if (plain)
					{
						const auto partialHW = GetVectorPart(HW());

						for_i(C, threads, [=](UInt c)
							{
								auto vecMean = VecFloat(0);
						auto mean = Float(0);
						auto vecVariance = VecFloat(0);
						auto variance = Float(0);
						{

							auto correction0 = VecFloat(0);
							auto correction1 = VecFloat(0);
							auto correction0Float = Float(0);
							auto correction1Float = Float(0);
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto start = c * HW() + (n * CDHW());
								const auto part = start + partialHW;
								for (auto hw = start; hw < part; hw += VectorSize)
								{
									KahanSum<VecFloat>(VecFloat().load_a(&InputLayer->Neurons[hw]), vecMean, correction0);
									KahanSum<VecFloat>(square(VecFloat().load_a(&InputLayer->Neurons[hw])), vecVariance, correction1);
								}
								const auto end = start + HW();
								for (auto hw = part; hw < end; hw++)
								{

									KahanSum<Float>(InputLayer->Neurons[hw], mean, correction0Float);
									KahanSum<Float>(Square(InputLayer->Neurons[hw]), variance, correction1Float);
								}
							}
						}
						mean += horizontal_add(vecMean);
						mean /= Float(batchSize * HW());
						Mean[c] = mean;
						variance += horizontal_add(vecVariance);
						const auto unbiasedVariance = std::max(0.f, variance / Float(batchSize * HW() - 1));
						variance /= Float(batchSize * HW());
						variance = std::max(0.f, variance);
						Variance[c] = variance;

						RunningMean[c] = RunningMean[c] * Momentum + OneMinusMomentum * mean;
						RunningVariance[c] = RunningVariance[c] * Momentum + OneMinusMomentum * unbiasedVariance;

						const auto invStdDev = Float(1) / std::sqrt(variance + Eps);
						const auto weightedInvStdDev = Scaling ? invStdDev * Weights[c] : invStdDev;
						const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

						if (InplaceBwd)
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto start = c * HW() + (n * CDHW());
								const auto part = start + partialHW;
								for (auto hw = start; hw < part; hw += VectorSize)
									Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases)).store_a(&Neurons[hw]);

								for (auto hw = part; hw < start + HW(); hw++)
									Neurons[hw] = Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases);
							}
						else
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto start = c * HW() + (n * CDHW());
								const auto part = start + partialHW;
								for (auto hw = start; hw < part; hw += VectorSize)
								{
									Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases)).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[hw]);
#endif
								}
								for (auto hw = part; hw < start + HW(); hw++)
								{
									Neurons[hw] = Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases);
#ifndef DNN_LEAN
									NeuronsD1[hw] = Float(0);
#endif
								}
							}
							});
					}
					else
					{
						for_i(PaddedC / VectorSize, threads, [=](UInt c)
							{
								const auto channelOffset = c * VectorSize;
						const auto mapOffset = channelOffset * HW();

						auto variance = VecFloat(0);
						auto mean = VecFloat(0);
						{
							auto correction0 = VecFloat(0);
							auto correction1 = VecFloat(0);
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto offsetC = n * PaddedCDHW() + mapOffset;
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									{
										KahanSum<VecFloat>(VecFloat().load_a(&InputLayer->Neurons[w]), mean, correction0);
										KahanSum<VecFloat>(square(VecFloat().load_a(&InputLayer->Neurons[w])), variance, correction1);
									}
								}
							}
						}
						mean /= Float(batchSize * HW());
						mean.store_a(&Mean[channelOffset]);
						const auto unbiasedVariance = max(0.f, (variance / Float(batchSize * HW() - 1)) - square(mean));
						variance = max(0.f, (variance / Float(batchSize * HW())) - square(mean));
						variance.store_a(&Variance[channelOffset]);

						mul_add(VecFloat().load_a(&RunningMean[channelOffset]), Momentum, OneMinusMomentum * mean).store_a(&RunningMean[channelOffset]);
						mul_add(VecFloat().load_a(&RunningVariance[channelOffset]), Momentum, OneMinusMomentum * unbiasedVariance).store_a(&RunningVariance[channelOffset]);

						const auto invStdDev = VecFloat(Float(1)) / sqrt(variance + Eps);
						invStdDev.store_a(&InvStdDev[channelOffset]);

						const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]) : invStdDev;
						const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);
						//VecFloat().load_a(&Weights[channelOffset]).store_a(&scale[channelOffset]);
						//biases.store_a(&shift[channelOffset]);

						if (InplaceBwd)
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto offsetC = n * PaddedCDHW() + mapOffset;
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases)).store_a(&Neurons[w]);
								}
							}
						else
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto offsetC = n * PaddedCDHW() + mapOffset;
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									{
										Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases)).store_a(&Neurons[w]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[w]);
#endif
									}
								}
							}
							});
					}
				}
			}

		}

		void BackwardProp(const UInt batchSize) final override
		{
			if constexpr (Reference)
				BackwardPropRef(batchSize);
			else
			{
#ifdef DNN_LEAN
				ZeroGradient(batchSize);
#endif // DNN_LEAN

				const auto strideH = W * VectorSize;
				const auto plain = IsPlainFormat();
				const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
				const auto threads = GetThreads(elements, Float(0.1));

				if (plain)
				{
					const auto partialHW = GetVectorPart(HW());

					for_i(C, threads, [=](UInt c)
						{
							const auto weightedInvStdDev = Scaling ? InvStdDev[c] * Weights[c] : InvStdDev[c];
					const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

					auto diffGammaFloat = Float(0);
					auto diffBetaFloat = Float(0);
					auto diffSrcFloat = Float(0);

					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);

					auto inputNeurons = VecFloat(0);

					const FloatArray& layerD1 = InplaceBwd ? InputLayer->NeuronsD1 : NeuronsD1;

					{
						auto correction0Float = Float(0);
						auto correction1Float = Float(0);

						auto correction0 = VecFloat(0);
						auto correction1 = VecFloat(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW() + (n * CDHW());
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								inputNeurons.load_a(&InputLayerFwd->Neurons[hw]);
								inputNeurons -= Mean[c];
								diffSrc = Activation::dfVec(inputNeurons * weightedInvStdDev + biases) * VecFloat().load_a(&layerD1[hw]);
								KahanSum<VecFloat>(diffSrc * inputNeurons, diffGamma, correction0);
								KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * layerD1[hw];
								KahanSum<Float>(diffSrcFloat * (InputLayerFwd->Neurons[hw] - Mean[c]), diffGammaFloat, correction0Float);
								KahanSum<Float>(diffSrcFloat, diffBetaFloat, correction1Float);
							}
						}
					}

					diffGammaFloat += horizontal_add(diffGamma);
					diffGammaFloat *= InvStdDev[c];

					diffBetaFloat += horizontal_add(diffBeta);

					if (Scaling)
					{
						WeightsD1[c] += diffGammaFloat;
						BiasesD1[c] += diffBetaFloat;
					}

					diffGammaFloat *= InvStdDev[c] / Float(batchSize * HW());
					diffBetaFloat /= Float(batchSize * HW());

					const auto gamma = Scaling ? Weights[c] * InvStdDev[c] : InvStdDev[c];
					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW() + (n * CDHW());
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&layerD1[hw]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{

								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * layerD1[hw];

								// if not using global stats!
								diffSrcFloat -= (InputLayerFwd->Neurons[hw] - Mean[c]) * diffGammaFloat + diffBetaFloat;

								//diffSrc *= gamma;
								InputLayer->NeuronsD1[hw] = diffSrcFloat * gamma;
							}
						}
					else
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW() + (n * CDHW());
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&layerD1[hw]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{

								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * layerD1[hw];

								// if not using global stats!
								diffSrcFloat -= (InputLayerFwd->Neurons[hw] - Mean[c]) * diffGammaFloat + diffBetaFloat;

								//diffSrc *= gamma;
								InputLayer->NeuronsD1[hw] += diffSrcFloat * gamma;
							}
						}
						});
				}
				else
				{
					for_i(PaddedC / VectorSize, threads, [=](UInt c)
						{
							const auto channelOffset = c * VectorSize;
					const auto mapOffset = channelOffset * HW();

					const auto mean = VecFloat().load_a(&Mean[channelOffset]);
					const auto invStdDev = VecFloat().load_a(&InvStdDev[channelOffset]);
					const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]) : invStdDev;
					const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);
					auto inputNeurons = VecFloat(0);
					const FloatArray& layerD1 = InplaceBwd ? InputLayer->NeuronsD1 : NeuronsD1;
					{
						auto correction0 = VecFloat(0);
						auto correction1 = VecFloat(0);

						for (auto n = 0ull; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW() + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc.load_a(&layerD1[w]);
									inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
									inputNeurons -= mean;
									diffSrc *= Activation::dfVec(mul_add(inputNeurons, weightedInvStdDev, biases));
									KahanSum<VecFloat>(diffSrc * inputNeurons, diffGamma, correction0);
									KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
								}
							}
						}
					}

					diffGamma *= invStdDev;

					if (Scaling)
					{
						(VecFloat().load_a(&WeightsD1[channelOffset]) + diffGamma).store_a(&WeightsD1[channelOffset]);
						(VecFloat().load_a(&BiasesD1[channelOffset]) + diffBeta).store_a(&BiasesD1[channelOffset]);
					}

					diffGamma *= invStdDev / Float(batchSize * HW());
					diffBeta /= Float(batchSize * HW());

					const auto gamma = Scaling ? VecFloat().load_a(&Weights[channelOffset]) * invStdDev : invStdDev;

					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW() + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc.load_a(&layerD1[w]);
									inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
									inputNeurons -= mean;
									diffSrc = mul_add(Activation::dfVec(mul_add(inputNeurons, weightedInvStdDev, biases)), diffSrc, -mul_add(inputNeurons, diffGamma, diffBeta));
									(diffSrc * gamma).store_a(&InputLayer->NeuronsD1[w]);
								}
							}
						}
					else
						for (auto n = 0ull; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW() + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc.load_a(&layerD1[w]);
									inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
									inputNeurons -= mean;
									diffSrc = mul_add(Activation::dfVec(mul_add(inputNeurons, weightedInvStdDev, biases)), diffSrc, -mul_add(inputNeurons, diffGamma, diffBeta));
									mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
								}
							}
						}
						});
				}

#ifdef DNN_LEAN
				ReleaseGradient();
#endif // DNN_LEAN	
			}
		}

		void ForwardPropRef (const UInt batchSize, const bool training)
		{
			const auto plain = IsPlainFormat();
			const auto threads = GetThreads(batchSize * (plain ? CDHW() : PaddedCDHW()), Float(0.1));
			const auto strideHW = HW() * VectorSize;

			if (!training)
			{
				if (!inference)
				{
					inference = true;
					InitializeDescriptors(batchSize);
				}

				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto memMean = dnnl::memory(fwdDesc->mean_desc(), Device.engine, RunningMean.data());
				auto memVariance = dnnl::memory(fwdDesc->variance_desc(), Device.engine, RunningVariance.data());

				auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());

				if (Scaling)
				{
					for (auto c = 0ull; c < C; c++)
					{
						scale[c] = Weights[c];
						shift[c] = Biases[c];
					}
					auto memScale = dnnl::memory(*WeightsMemDesc, Device.engine, scale.data());
					auto memShift = dnnl::memory(*WeightsMemDesc, Device.engine, shift.data());
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, memScale }, { DNNL_ARG_SHIFT, memShift }, { DNNL_ARG_DST, dstMem } });
#endif
					dnnl::batch_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, memScale }, { DNNL_ARG_SHIFT, memShift }, { DNNL_ARG_DST, dstMem } });
				}
				else
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem } });
#else
					dnnl::batch_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem } });
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
				auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto memMean = dnnl::memory(fwdDesc->mean_desc(), Device.engine, Mean.data());
				auto memVariance = dnnl::memory(fwdDesc->variance_desc(), Device.engine, Variance.data());

				auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, InputNeurons.data());

				if (Scaling)
				{
					for (auto c = 0ull; c < C; c++)
					{
						scale[c] = Weights[c];
						shift[c] = Biases[c];
					}
					
					auto memScale = dnnl::memory(*WeightsMemDesc, Device.engine, scale.data());
					auto memShift = dnnl::memory(*WeightsMemDesc, Device.engine, shift.data());
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, memScale }, { DNNL_ARG_SHIFT, memShift }, { DNNL_ARG_DST, dstMem } });
#else
					dnnl::batch_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, memScale }, { DNNL_ARG_SHIFT, memShift }, { DNNL_ARG_DST, dstMem } });
#endif
				}
				else
#ifdef DNN_CACHE_PRIMITIVES
					fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem } });
#else
					dnnl::batch_normalization_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DST, dstMem } });
#endif
				Device.stream.wait();

				const Float unbiasedFactor = Float(batchSize * HW()) / Float(batchSize * HW() - 1);
				for (auto c = 0ull; c < C; c++)
				{
					RunningMean[c] = (Momentum * RunningMean[c]) + (OneMinusMomentum * Mean[c]);
					RunningVariance[c] = (Momentum * RunningVariance[c]) + (OneMinusMomentum * Variance[c] * unbiasedFactor);
				}

#ifndef DNN_LEAN

#else
				DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN
			}
						
			if (training)
			{
				if (!plain)
				{
					if (!InplaceBwd)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto vecZero = VecFloat(0);
							VecFloat neurons;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = n * PaddedCDHW() + c * HW();
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
								{
									neurons.load_a(&InputNeurons[hw]);
									Activation::fVec(neurons).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[hw]);
#endif // DNN_LEAN
								}
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							const auto vecZero = VecFloat(0);
							VecFloat neurons;
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = n * PaddedCDHW() + c * HW();
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
								{
									neurons.load_a(&InputNeurons[hw]);
									Activation::fVec(neurons).store_a(&Neurons[hw]);
								}
							}
						});
					}
				}
				else
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto offset = n * CDHW() + c * HW();
							for (auto hw = offset; hw < offset + HW(); hw++)
							{
								Neurons[hw] = Activation::f(InputNeurons[hw]);
#ifndef DNN_LEAN
								if (!InplaceBwd)
									NeuronsD1[hw] = Float(0);
#endif // DNN_LEAN
							}
						}
					});
			}
			else
			{
				if (!plain)
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						VecFloat neurons;

						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto offset = n * PaddedCDHW() + c * HW();
							for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
							{
								neurons.load_a(&Neurons[hw]);
								Activation::fVec(neurons).store_a(&Neurons[hw]);
							}
						}
					});
				}
				else
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < C; c++)
						{
							const auto offset = n * CDHW() + c * HW();
							for (auto hw = offset; hw < offset + HW(); hw++)
								Neurons[hw] = Activation::f(Neurons[hw]);
						}
					});
				}
			}
		}

		void BackwardPropRef(const UInt batchSize)
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#else
			DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN
					
			const auto plain = IsPlainFormat();
			const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
			const auto threads = GetThreads(elements, Float(0.1));
			const auto strideHW = HW() * VectorSize;

			if (InputLayer->DstMemDesc->get_ndims() == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (InplaceBwd)
					{
						if (!plain)
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
								(Activation::dfVec(VecFloat().load_a(&InputNeurons[c])), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
						else
						{
							PRAGMA_OMP_SIMD()
								for (auto c = 0ull; c < C; c++)
									InputLayer->NeuronsD1[c] = Activation::df(InputNeurons[c]) * InputLayer->NeuronsD1[c];
						}
					}
					else
					{
						if (!plain)
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
								(Activation::dfVec(VecFloat().load_a(&InputNeurons[c])) * VecFloat().load_a(&NeuronsD1[c])).store_a(NeuronsD1[c]);
						else
						{
							PRAGMA_OMP_SIMD()
							for (auto c = 0ull; c < C; c++)
								NeuronsD1[c] = Activation::df(InputNeurons[c]) * NeuronsD1[c];
						}
					}
				else
				{
#endif
					if (InplaceBwd)
					{
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * PaddedC;
								for (auto c = offset; c < offset + PaddedC; c += VectorSize)
									(Activation::dfVec(VecFloat().load_a(&InputNeurons[c])) * VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * C;
								PRAGMA_OMP_SIMD()
								for (auto c = offset; c < offset + C; c++)
									InputLayer->NeuronsD1[c] = Activation::df(InputNeurons[c]) * InputLayer->NeuronsD1[c];
							});
					}
					else
					{
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * PaddedC;
								for (auto c = offset; c < offset + PaddedC; c += VectorSize)
									(Activation::dfVec(VecFloat().load_a(&InputNeurons[c])) * VecFloat().load_a(&NeuronsD1[c])).store_a(&NeuronsD1[c]);
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * C;
								PRAGMA_OMP_SIMD()
								for (auto c = offset; c < offset + C; c++)
									NeuronsD1[c] = Activation::df(InputNeurons[c]) * NeuronsD1[c];
							});
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (InplaceBwd)
					{
						if (!plain)
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = c * HW();
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
									(Activation::dfVec(VecFloat().load_a(&InputNeurons[hw])) * VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = c * HW();
								PRAGMA_OMP_SIMD()
								for (auto hw = offset; hw < offset + HW(); hw++)
									InputLayer->NeuronsD1[hw] = Activation::df(InputNeurons[hw]) * InputLayer->NeuronsD1[hw];
							}
						}
					}
					else
					{
						if (!plain)
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = c * HW();
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
									(Activation::dfVec(VecFloat().load_a(&InputNeurons[hw])) * VecFloat().load_a(&NeuronsD1[hw])).store_a(&NeuronsD1[hw]);
							}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = c * HW();
								PRAGMA_OMP_SIMD()
								for (auto hw = offset; hw < offset + HW(); hw++)
									NeuronsD1[hw] = Activation::df(InputNeurons[hw]) * NeuronsD1[hw];
							}
						}
					}
				}
				else
				{
#endif
					if (InplaceBwd)
					{
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = n * PaddedCDHW() + c * HW();
									for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
										(Activation::dfVec(VecFloat().load_a(&InputNeurons[hw])) * VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = n * CDHW() + c * HW();
									PRAGMA_OMP_SIMD()
									for (auto hw = offset; hw < offset + HW(); hw++)
										InputLayer->NeuronsD1[hw] *= Activation::df(InputNeurons[hw]);
								}
							});
					}
					else
					{
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = n * PaddedCDHW() + c * HW();
									for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
										(Activation::dfVec(VecFloat().load_a(&InputNeurons[hw])) * VecFloat().load_a(&NeuronsD1[hw])).store_a(&NeuronsD1[hw]);
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = n * CDHW() + c * HW();
									PRAGMA_OMP_SIMD()
									for (auto hw = offset; hw < offset + HW(); hw++)
										NeuronsD1[hw] *= Activation::df(InputNeurons[hw]);
								}
							});
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}


			auto memSrc = dnnl::memory(*InputLayerFwd->DstMemDesc, Device.engine, InputLayerFwd->Neurons.data());
			auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.engine) : memSrc;
			if (reorderBwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto memMean = dnnl::memory(bwdDesc->mean_desc(), Device.engine, Mean.data());
			auto memVariance = dnnl::memory(bwdDesc->variance_desc(), Device.engine, Variance.data());

			auto memDiffSrc = SharesInput && !InplaceBwd ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

			if (Scaling)
			{
				for (auto c = 0ull; c < PaddedC; c++)
				{
					diffScale[c] = Float(0);
					diffShift[c] = Float(0);
				}

				auto scaleMemory = dnnl::memory(*WeightsMemDesc, Device.engine, scale.data());
				auto shiftMemory = dnnl::memory(*WeightsMemDesc, Device.engine, shift.data());
				auto diffScaleMemory = dnnl::memory(*WeightsMemDesc, Device.engine, diffScale.data());
				auto diffShiftMemory = dnnl::memory(*WeightsMemDesc, Device.engine, diffShift.data());
#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, scaleMemory }, { DNNL_ARG_SHIFT, shiftMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_DIFF_SCALE, diffScaleMemory }, { DNNL_ARG_DIFF_SHIFT, diffShiftMemory } });
#else
				dnnl::batch_normalization_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, scaleMemory }, { DNNL_ARG_SHIFT, shiftMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_DIFF_SCALE, diffScaleMemory }, { DNNL_ARG_DIFF_SHIFT, diffShiftMemory } });
#endif

				for (auto c = 0ull; c < C; c++)
				{
					WeightsD1[c] += diffScale[c];
					BiasesD1[c] += diffShift[c];
				}
			}
			else
#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::batch_normalization_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
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
				const auto rangeWeights = GetColorRange<Float>(WeightsStats.Min, WeightsStats.Max);
				const auto rangeBiases = GetColorRange<Float>(BiasesStats.Min, BiasesStats.Max);

				const auto width = BiasCount;
				const auto height = WeightCount / BiasCount;
				const auto totalSize = width * (height + 3);

				auto image = ByteArray(totalSize, fillColor);

				for (auto y = 0ull; y < height; y++)
				{
					const auto start = y * width;
					const auto end = start + width;
					for (auto x = start; x < end; x++)
						image[x] = GetColorFromRange<Float>(rangeWeights, WeightsStats.Min, Weights[x]);
				}

				if (HasBias)
				{
					const auto offset = (height + 1) * width;
					for (auto x = 0ull; x < width; x++)
						image[x + offset] = GetColorFromRange<Float>(rangeBiases, BiasesStats.Min, Biases[x]);
				}

				return image;
			}
			else
				return ByteArray();
		}

		void ResetWeights(const Fillers weightsFiller, const FillerModes weightsFillerMode, const Float weightsGain, const Float weightsFillerScale, const Fillers biasesFiller, const FillerModes biasesFillerMode, const Float biasesGain, const Float biasesFillerScale) override
		{
			Weights.resize(PaddedC); std::fill(Weights.begin(), Weights.end(), Float(1));
			Biases.resize(PaddedC); std::fill(Biases.begin(), Biases.end(), Float(0));

			if (Scaling)
			{
				for (auto c = 0ull; c < PaddedC; c++)
				{
					scale[c] = Weights[c];
					shift[c] = Biases[c];
				}
			}

			RunningMean.resize(PaddedC); std::fill(RunningMean.begin(), RunningMean.end(), Float(0));
			RunningVariance.resize(PaddedC); std::fill(RunningVariance.begin(), RunningVariance.end(), Float(1));

			DNN_UNREF_PAR(weightsFiller);
			DNN_UNREF_PAR(weightsFillerMode);
			DNN_UNREF_PAR(weightsGain);
			DNN_UNREF_PAR(weightsFillerScale);
			DNN_UNREF_PAR(biasesFiller);
			DNN_UNREF_PAR(biasesFillerMode);
			DNN_UNREF_PAR(biasesGain);
			DNN_UNREF_PAR(biasesFillerScale);
		}

		void Save(std::ostream& os, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			os.write(reinterpret_cast<const char*>(RunningMean.data()), std::streamsize(C * sizeof(Float)));
			os.write(reinterpret_cast<const char*>(RunningVariance.data()), std::streamsize(C * sizeof(Float)));

			if (Scaling)
			{
				for (auto c = 0ull; c < C; c++)
				{
					Weights[c] = scale[c];
					Biases[c] = shift[c];
				}
			}

			Layer::Save(os, persistOptimizer, optimizer);
		}

		void Load(std::istream& is, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			is.read(reinterpret_cast<char*>(RunningMean.data()), std::streamsize(C * sizeof(Float)));
			is.read(reinterpret_cast<char*>(RunningVariance.data()), std::streamsize(C * sizeof(Float)));

			Layer::Load(is, persistOptimizer, optimizer);

			if (Scaling)
			{
				for (auto c = 0ull; c < C; c++)
				{
					scale[c] = Weights[c];
					shift[c] = Biases[c];
				}
			}
		}

		std::streamsize GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const override
		{
			return (2 * C * sizeof(Float)) + Layer::GetWeightsSize(persistOptimizer, optimizer);
		}
	};
}