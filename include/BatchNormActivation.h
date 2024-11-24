#pragma once
#include "Activation.h"

namespace dnn
{
	class BatchNormActivation final : public Layer
	{
	private:
		std::unique_ptr<dnnl::batch_normalization_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::batch_normalization_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::batch_normalization_forward> fwd;
		std::unique_ptr<dnnl::batch_normalization_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		dnnl::normalization_flags flags;
		bool inference;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
		bool reorderBwdDiffDst;

	public:
		const Activations ActivationFunction;
		const Float Alpha;
		const Float Beta;
		const Act Func;
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;
		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;
		FloatArray InputNeurons;

		BatchNormActivation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector<Layer*>& inputs, const bool scaling = true, const Float alpha = Float(0), const Float beta = Float(0), const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true) :
			Layer(device, format, name, LayerTypes::BatchNormActivation, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias, scaling),
			ActivationFunction(activation),
			Alpha(Activation::GetAlpha(activation, alpha, beta)),
			Beta(Activation::GetBeta(activation, alpha, beta)),
			Func(Activation::GetActivation(activation)),
			Eps(eps),
			Momentum(momentum),
			OneMinusMomentum(Float(1) - momentum),
			Mean(FloatVector(PaddedC, Float(0))),
			RunningMean(FloatVector(PaddedC, Float(0))),
			Variance(FloatVector(PaddedC, Float(1))),
			RunningVariance(FloatVector(PaddedC, Float(1))),
			InvStdDev(FloatVector(PaddedC)),
			InputNeurons(FloatArray()),
			flags(static_cast<dnnl::normalization_flags>(0U)),
			inference(false),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false),
			reorderBwdDiffDst(false)
		{
			assert(Inputs.size() == 1);

			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::a));
			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::a));
			WeightsFormat = GetMemoryFormat(*WeightsMemDesc);

			FwdInferenceWeight = Float(5);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
		}
	
		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Activation: ") + tab + std::string(magic_enum::enum_name<Activations>(ActivationFunction)));
			description.append(nwl + std::string(" Alpha:      ") + tab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:       ") + tab + FloatToString(Beta));
			description += GetWeightsDescription(Scaling);
			description.append(nwl + std::string(" Momentum:   ") + tab + FloatToString(Momentum));
			description.append(nwl + std::string(" Eps:        ") + tab + FloatToStringScientific(Eps));

			auto mean = Float(0);
			auto variance = Float(0);
			for (auto c = 0ull; c < C; c++)
			{
				mean += RunningMean[c];
				variance += RunningVariance[c];
			}
			mean /= C;
			variance /= C;

			description.append(nwl + std::string(" Mean:       ") + tab + FloatToStringFixed(mean));
			description.append(nwl + std::string(" Variance:   ") + tab + FloatToStringFixed(variance));
						
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

			if constexpr (Reference || TestBatchNormalization || ReferenceBatchNormalization)
				InputNeurons.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
		}

		void InitializeDescriptorsFwd(const UInt batchSize) final override
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
						throw std::invalid_argument(std::string("Src and Diff format are different in ") + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + std::string(" layer ") + Name);
				}
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			
			if constexpr (Reference || TestBatchNormalization || ReferenceBatchNormalization)
			{
				if (inference)
					flags = Scaling ?
						dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift
						: dnnl::normalization_flags::use_global_stats;
				else
					flags = Scaling ?
						dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift
						: static_cast<dnnl::normalization_flags>(0U);

				fwdDesc = std::make_unique<dnnl::batch_normalization_forward::primitive_desc>(dnnl::batch_normalization_forward::primitive_desc(Device.engine, inference ? dnnl::prop_kind::forward_inference : dnnl::prop_kind::forward_training, *DstMemDesc, *DstMemDesc, Eps, flags));

				reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
				fwd = std::make_unique<dnnl::batch_normalization_forward>(dnnl::batch_normalization_forward(*fwdDesc));
#endif
				if (!inference)
				{
					bwdDesc = std::make_unique<dnnl::batch_normalization_backward::primitive_desc>(dnnl::batch_normalization_backward::primitive_desc(Device.engine, Scaling ? dnnl::prop_kind::backward : dnnl::prop_kind::backward_data, *DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *DstMemDesc, Eps, flags, *fwdDesc));

					reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
					reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;
					reorderBwdDiffDst = bwdDesc->diff_dst_desc() != (!InplaceBwd ? *DiffDstMemDesc : *InputLayer->DiffDstMemDesc);

					bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));

#ifdef DNN_CACHE_PRIMITIVES
					bwd = std::make_unique<dnnl::batch_normalization_backward>(dnnl::batch_normalization_backward(*bwdDesc));
					bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
				}
			}
		}

		void InitializeDescriptorsBwd(const UInt batchSize) final override
		{
		}

		bool Lockable() const final override
		{
			return WeightCount > 0 && Scaling;
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{			
			if constexpr ((Reference || ReferenceBatchNormalization) && !TestBatchNormalization)
				ForwardPropRef(batchSize, training);
			else
			{
				const auto strideH = W * VectorSize;
				const auto plain = IsPlainFormat();
				const auto padded = C == PaddedC;
				const auto part = padded ? PaddedC : (PaddedC - VectorSize);

				if (!training)
				{
					const auto maxThreads = GetThreads(batchSize * GetElementsCount(), FwdInferenceWeight);
										
					if (plain) // nchw
					{
						const auto partialHW = GetVectorPart(HW());
						const auto threads = std::min<UInt>(maxThreads, C);

						for_i(C, threads, [=](UInt c)
						{
							const auto invStddev = Float(1) / std::sqrt(RunningVariance[c] + Eps);
							const auto weightedInvStdDev = Scaling ? (Weights[c] * invStddev) : invStddev;
							const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto start = c * HW() + (n * CDHW());
								const auto part = start + partialHW;
								for (auto hw = start; hw < part; hw += VectorSize)
									Func.fVec((VecFloat().load_a(&InputLayer->Neurons[hw]) - RunningMean[c]) * weightedInvStdDev + biases, Alpha, Beta).store_a(&Neurons[hw]);
								for (auto hw = part; hw < start + HW(); hw++)
									Neurons[hw] = Func.f((InputLayer->Neurons[hw] - RunningMean[c]) * weightedInvStdDev + biases, Alpha, Beta );
							}
						});
					}
					else
					{
						const auto threads = std::min<UInt>(maxThreads, PaddedC / VectorSize);

						for_i(PaddedC / VectorSize, threads, [=](UInt c)
						{
							//const auto overflow = ((!padded) && (c >= (part / VectorSize)));
							//const auto cutoff = overflow ? int(VectorSize - (PaddedC - C)) : int(VectorSize);

							const auto channelOffset = c * VectorSize;
							const auto mapOffset = channelOffset * HW();

							const auto runningMean = VecFloat().load_a(&RunningMean[channelOffset]);
							const auto invStddev = (VecFloat(1) / sqrt(VecFloat().load_a(&RunningVariance[channelOffset]) + Eps));
							const auto weightedInvStdDev = Scaling ? (VecFloat().load_a(&Weights[channelOffset]) * invStddev) : invStddev;
							const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto offsetC = n * PaddedCDHW() + mapOffset;
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										Func.fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - runningMean, weightedInvStdDev, biases), Alpha, Beta).store_a(&Neurons[w]);
								}
							}
						});
					}
				}
				else
				{
					const auto maxThreads = GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

					if (plain)
					{
						const auto partialHW = GetVectorPart(HW());
						const auto threads = std::min<UInt>(maxThreads, C);

						for_i(C, threads, [=](UInt c)
						{
							auto mean = Float(0);
							auto variance = Float(0);
							auto unbiasedVariance = Float(0);
						
							auto vecMean = VecFloat(0);
							auto vecVariance = VecFloat(0);
							auto correction0 = VecFloat(0);
							auto correction1 = VecFloat(0);
							auto correction0Float = Float(0);
							auto correction1Float = Float(0);
															
							if constexpr (SingleMeanVariancePass)
							{
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

								mean += horizontal_add(vecMean);
								mean /= Float(batchSize * HW());
								Mean[c] = mean;

								variance += horizontal_add(vecVariance);
								unbiasedVariance = std::max(Float(0), (variance / Float(batchSize * HW() - 1)) - Square<Float>(mean));
								variance /= Float(batchSize * HW());
								variance -= Square<Float>(mean);
								variance = std::max(Float(0), variance);
								Variance[c] = variance;
							}
							else
							{
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
										KahanSum<VecFloat>(VecFloat().load_a(&InputLayer->Neurons[hw]), vecMean, correction0);
									const auto end = start + HW();
									for (auto hw = part; hw < end; hw++)
										KahanSum<Float>(InputLayer->Neurons[hw], mean, correction0Float);
								}

								mean += horizontal_add(vecMean);
								mean /= Float(batchSize * HW());
								Mean[c] = mean;

								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
										KahanSum<VecFloat>(square(VecFloat().load_a(&InputLayer->Neurons[hw]) - mean), vecVariance, correction1);
									const auto end = start + HW();
									for (auto hw = part; hw < end; hw++)
										KahanSum<Float>(Square(InputLayer->Neurons[hw] - mean), variance, correction1Float);
								}

								variance += horizontal_add(vecVariance);
								unbiasedVariance = std::max(Float(0), variance / Float(batchSize * HW() - 1));
								variance /= Float(batchSize * HW());
								variance = std::max(Float(0), variance);
								Variance[c] = variance;
							}
							
							RunningMean[c] = RunningMean[c] * Momentum + OneMinusMomentum * mean;
							RunningVariance[c] = RunningVariance[c] * Momentum + OneMinusMomentum * unbiasedVariance;

							const auto invStddev = Float(1) / std::sqrt(variance + Eps);
							const auto weightedInvStdDev = Scaling ? (Weights[c] * invStddev) : invStddev;
							const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

							InvStdDev[c] = invStddev;

							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
										Func.fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases), Alpha, Beta).store_a(&Neurons[hw]);

									for (auto hw = part; hw < start + HW(); hw++)
										Neurons[hw] = Func.f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases, Alpha, Beta);
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										Func.fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases), Alpha, Beta).store_a(&Neurons[hw]);
	#ifndef DNN_LEAN
										VecZero.store_nt(&NeuronsD1[hw]);
	#endif
									}
									for (auto hw = part; hw < start + HW(); hw++)
									{
										Neurons[hw] = Func.f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases, Alpha, Beta);
	#ifndef DNN_LEAN
										NeuronsD1[hw] = Float(0);
	#endif
									}
								}
						});
					}
					else
					{
						const auto threads = std::min<UInt>(maxThreads, PaddedC / VectorSize);

						for_i(PaddedC / VectorSize, threads, [=](UInt c)
						{		
							const auto overflow = ((!padded) && (c >= (part / VectorSize)));
							const auto cutoff = overflow ? int(VectorSize - (PaddedC - C)) : int(VectorSize);

							const auto channelOffset = c * VectorSize;
							const auto mapOffset = channelOffset * HW();

							auto mean = VecFloat(0);
							auto variance = VecFloat(0);
							auto unbiasedVariance = VecFloat(0);

							if constexpr (SingleMeanVariancePass)
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

								mean /= Float(batchSize * HW());
								mean.store_a(&Mean[channelOffset]);
								
								unbiasedVariance = max(VecFloat(0), (variance / Float(batchSize * HW() - 1)) - square(mean));
								variance /= Float(batchSize * HW());
								variance -= square(mean);

								mean = mean.cutoff(cutoff);
								unbiasedVariance = unbiasedVariance.cutoff(cutoff);
								variance = variance.cutoff(cutoff);
							}
							else
							{
								auto correction0 = VecFloat(0);
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW() + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
											KahanSum<VecFloat>(VecFloat().load_a(&InputLayer->Neurons[w]), mean, correction0);
									}
								}

								mean /= Float(batchSize * HW());
								mean.store_a(&Mean[channelOffset]);
								
								auto correction1 = VecFloat(0);
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW() + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
											KahanSum<VecFloat>(square(VecFloat().load_a(&InputLayer->Neurons[w]) - mean), variance, correction1);
									}
								}

								unbiasedVariance = max(VecFloat(0), (variance / Float(batchSize * HW() - 1)));
								variance /= Float(batchSize * HW());

								mean = mean.cutoff(cutoff);
								unbiasedVariance = unbiasedVariance.cutoff(cutoff);
								variance = variance.cutoff(cutoff);
							}

							variance = max(VecFloat(0), variance);
							variance.store_a(&Variance[channelOffset]);
							variance = variance.cutoff(cutoff);

							mul_add(VecFloat().load_a(&RunningMean[channelOffset]), Momentum, OneMinusMomentum * mean).store_a(&RunningMean[channelOffset]);
							mul_add(VecFloat().load_a(&RunningVariance[channelOffset]), Momentum, OneMinusMomentum * unbiasedVariance).store_a(&RunningVariance[channelOffset]);

							const auto invStddev = (VecFloat(Float(1)) / sqrt(variance + Eps)).cutoff(cutoff);
							const auto weightedInvStdDev = Scaling ? (VecFloat().load_a(&Weights[channelOffset]).cutoff(cutoff) * invStddev) : invStddev;
							const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]).cutoff(cutoff) : VecFloat(0);


							invStddev.store_a(&InvStdDev[channelOffset]);

							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW() + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
											Func.fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]).cutoff(cutoff) - mean, weightedInvStdDev, biases), Alpha, Beta).store_a(&Neurons[w]);
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
											Func.fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]).cutoff(cutoff) - mean, weightedInvStdDev, biases), Alpha, Beta).store_a(&Neurons[w]);
#ifndef DNN_LEAN
											VecZero.store_nt(&NeuronsD1[w]);
#endif
										}
									}
								}
						});
					}
				}
			}

			if constexpr (TestBatchNormalization)
			{
				auto output = FloatArray();
				output.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto i = 0ull; i < Neurons.size(); i++)
					output[i] = Neurons[i];
				
				// check has equal neurons
				ForwardPropRef(batchSize, training);

				const auto margin = Float(0.0005);

				for (auto i = 0ull; i < Neurons.size(); i++)
				{
					if (((output[i] - margin) > Neurons[i]) || ((output[i] + margin) < Neurons[i]))
					{
						cimg_library::cimg::dialog("BatchNormActivation Sanity Check", (std::string("Forward Check not passed: ") + Name).c_str(), "OK");
						break;
					}
				}
			}
		}

	    void BackwardProp(const UInt batchSize) final override
		{
			auto output = FloatArray();
			if constexpr (TestBatchNormalization)
			{
				output.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
					output[i] = InputLayer->NeuronsD1[i];
			}

			if constexpr ((Reference || ReferenceBatchNormalization ) && !TestBatchNormalization)
				BackwardPropRef(batchSize);
			else
			{
#ifdef DNN_LEAN
				ZeroGradient(batchSize);
#endif // DNN_LEAN

				const auto strideH = W * VectorSize;
				const auto plain = IsPlainFormat();
				const auto maxThreads = GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const auto padded = C == PaddedC;
				const auto part = padded ? PaddedC : (PaddedC - VectorSize);
				
				if (plain)
				{
					const auto partialHW = GetVectorPart(HW());
					const auto threads = std::min<UInt>(maxThreads, C);

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
								diffSrc = Func.dfVec(inputNeurons * weightedInvStdDev + biases, Alpha, Beta) * VecFloat().load_a(&layerD1[hw]);
								KahanSum<VecFloat>(diffSrc * inputNeurons, diffGamma, correction0);
								KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
								diffSrcFloat = Func.df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases, Alpha, Beta) * layerD1[hw];
								KahanSum<Float>(diffSrcFloat * (InputLayerFwd->Neurons[hw] - Mean[c]), diffGammaFloat, correction0Float);
								KahanSum<Float>(diffSrcFloat, diffBetaFloat, correction1Float);
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
									diffSrc = Func.dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases, Alpha, Beta) * VecFloat().load_a(&layerD1[hw]);

									// if not using global stats!
									diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

									//diffSrc *= gamma;
									mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[hw]);
								}
								for (auto hw = part; hw < start + HW(); hw++)
								{

									diffSrcFloat = Func.df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases, Alpha, Beta) * layerD1[hw];

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
									diffSrc = Func.dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases, Alpha, Beta) * VecFloat().load_a(&layerD1[hw]);

									// if not using global stats!
									diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

									//diffSrc *= gamma;
									mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
								for (auto hw = part; hw < start + HW(); hw++)
								{
									diffSrcFloat = Func.df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases, Alpha, Beta) * layerD1[hw];

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
					const auto threads = std::min<UInt>(maxThreads, PaddedC / VectorSize);

					for_i(PaddedC / VectorSize, threads, [=](UInt c)
					{
						const auto overflow = ((!padded) && (c >= (part / VectorSize)));
						const auto cutoff = overflow ? int(VectorSize - (PaddedC - C)) : int(VectorSize);

						const auto channelOffset = c * VectorSize;
						const auto mapOffset = channelOffset * HW();

						const auto mean = VecFloat().load_a(&Mean[channelOffset]).cutoff(cutoff);
						const auto invStdDev = VecFloat().load_a(&InvStdDev[channelOffset]).cutoff(cutoff);
						const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]).cutoff(cutoff) : invStdDev;
						const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]).cutoff(cutoff) : VecFloat(0);
						auto diffGamma = VecFloat(0);
						auto diffBeta = VecFloat(0);
						auto diffSrc = VecFloat(0);
						auto inputNeurons = VecFloat(0);
						const FloatArray& layerD1 = InplaceBwd ? InputLayer->NeuronsD1 : NeuronsD1;
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
									inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
									inputNeurons -= mean;
									inputNeurons = inputNeurons.cutoff(cutoff);
									diffSrc = Func.dfVec(mul_add(inputNeurons, weightedInvStdDev, biases), Alpha, Beta) * VecFloat().load_a(&layerD1[w]);
									KahanSum<VecFloat>(diffSrc * inputNeurons, diffGamma, correction0);
									KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
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

						const auto gamma = Scaling ? VecFloat().load_a(&Weights[channelOffset]).cutoff(cutoff) * invStdDev : invStdDev;

						if (InplaceBwd)
							for (auto n = 0ull; n < batchSize; n++)
							{
								const auto offsetC = n * PaddedCDHW() + mapOffset;
								for (auto h = 0ull; h < H; h++)
								{
									const auto offsetH = offsetC + h * strideH;
									for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									{
										inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
										inputNeurons -= mean;
										inputNeurons = inputNeurons.cutoff(cutoff);
									
										diffSrc = Func.dfVec(mul_add(inputNeurons, weightedInvStdDev, biases), Alpha, Beta) * VecFloat().load_a(&layerD1[w]);

										// if not using global stats!
										diffSrc -= mul_add(inputNeurons, diffGamma, diffBeta);

										//diffSrc *= gamma;
										mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[w]);
									}
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
										inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
										inputNeurons -= mean;
										inputNeurons = inputNeurons.cutoff(cutoff);

										diffSrc = Func.dfVec(mul_add(inputNeurons, weightedInvStdDev, biases), Alpha, Beta) * VecFloat().load_a(&layerD1[w]);

										// if not using global stats!
										diffSrc -= mul_add(inputNeurons, diffGamma, diffBeta);

										//diffSrc *= gamma;
										mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[w]);
									}
								}
							}
					});
				}
#ifdef DNN_LEAN
				ReleaseGradient();
#endif // DNN_LEAN	
			}

			if constexpr (TestBatchNormalization)
			{
				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
					std::swap(output[i], InputLayer->NeuronsD1[i]);

				// check has equal neurons
				BackwardPropRef(batchSize);

				const auto margin = Float(0.0025);
				
				for (auto i = 0ull; i < InputLayer->NeuronsD1.size(); i++)
				{
					if (((output[i] - margin) > InputLayer->NeuronsD1[i]) || ((output[i] + margin) < InputLayer->NeuronsD1[i]))
					{
						auto msg = std::string("");
						msg += std::to_string(InputLayer->NeuronsD1[i]) + nwl;
						msg += std::to_string(output[i]) + nwl;;

						cimg_library::cimg::dialog("BatchNormActivation Sanity Check", (std::string("Backward Check not passed: ") + Name + nwl + msg).c_str(), "OK");
						break;
					}
				}
			}
		}
		
		void ForwardPropRef (const UInt batchSize, const bool training)
		{
			if (!training)
			{
				if (!inference)
				{
					inference = true;
					InitializeDescriptorsFwd(batchSize);
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
					auto memScale = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
					auto memShift = dnnl::memory(*WeightsMemDesc, Device.engine, Biases.data());
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
					InitializeDescriptorsFwd(batchSize);
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
					auto memScale = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
					auto memShift = dnnl::memory(*WeightsMemDesc, Device.engine, Biases.data());

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
			}
				
			const auto plain = IsPlainFormat();
			const auto maxThreads = GetThreads(batchSize * (plain ? CDHW() : PaddedCDHW()), Float(5));
			const auto threads = std::min<UInt>(maxThreads, batchSize);
			const auto strideHW = HW() * VectorSize;

			if (training)
			{
				if (!plain)
				{
					if (!InplaceBwd)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = n * PaddedCDHW() + c * HW();
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
								{
									Func.fVec(VecFloat().load_a(&InputNeurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
									VecZero.store_nt(&NeuronsD1[hw]);
#endif // DNN_LEAN
								}
							}
						});
					}
					else
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = n * PaddedCDHW() + c * HW();
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
									Func.fVec(VecFloat().load_a(&InputNeurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
							}
						});
					}
				}
				else
				{
					if (!InplaceBwd)
					{
						for_i(batchSize, threads, [=](UInt n)
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = n * CDHW() + c * HW();
								for (auto hw = offset; hw < offset + HW(); hw++)
								{
									Neurons[hw] = Func.f(InputNeurons[hw], Alpha, Beta);
#ifndef DNN_LEAN
									NeuronsD1[hw] = Float(0);
#endif // DNN_LEAN
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
									Neurons[hw] = Func.f(InputNeurons[hw], Alpha, Beta);
							}
						});
					}
				}
			}
			else
			{
				if (!plain)
				{
					for_i(batchSize, threads, [=](UInt n)
					{
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							const auto offset = n * PaddedCDHW() + c * HW();
							for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
								Func.fVec(VecFloat().load_a(&Neurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
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
								Neurons[hw] = Func.f(Neurons[hw], Alpha, Beta);
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
			const auto maxThreads = GetThreads(elements, Float(5));
			const auto threads = std::min<UInt>(maxThreads, batchSize);
			const auto strideHW = HW() * VectorSize;

			if (InputLayer->DstMemDesc->get_ndims() == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					if (InplaceBwd)
					{
						if (!plain)
						{
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
								(Func.dfVec(VecFloat().load_a(&InputNeurons[c]), Alpha, Beta) * VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
						}
						else
						{
							for (auto c = 0ull; c < C; c++)
								InputLayer->NeuronsD1[c] = Func.df(InputNeurons[c], Alpha, Beta) * InputLayer->NeuronsD1[c];
						}
					}
					else
					{
						if (!plain)
						{
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
								(Func.dfVec(VecFloat().load_a(&InputNeurons[c]), Alpha, Beta) * VecFloat().load_a(&NeuronsD1[c])).store_a(&NeuronsD1[c]);
						}
						else
						{
							for (auto c = 0ull; c < C; c++)
								NeuronsD1[c] = Func.df(InputNeurons[c], Alpha, Beta) * NeuronsD1[c];
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
								const auto offset = n * PaddedC;
								for (auto c = offset; c < offset + PaddedC; c += VectorSize)
									(Func.dfVec(VecFloat().load_a(&InputNeurons[c]), Alpha, Beta) * VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * C;
								for (auto c = offset; c < offset + C; c++)
									InputLayer->NeuronsD1[c] = Func.df(InputNeurons[c], Alpha, Beta) * InputLayer->NeuronsD1[c];
							});
					}
					else
					{
						if (!plain)
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * PaddedC;
								for (auto c = offset; c < offset + PaddedC; c += VectorSize)
									(Func.dfVec(VecFloat().load_a(&InputNeurons[c]), Alpha, Beta) * VecFloat().load_a(&NeuronsD1[c])).store_a(&NeuronsD1[c]);
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								const auto offset = n * C;
								for (auto c = offset; c < offset + C; c++)
									NeuronsD1[c] = Func.df(InputNeurons[c], Alpha, Beta) * NeuronsD1[c];
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
									(Func.dfVec(VecFloat().load_a(&InputNeurons[hw]), Alpha, Beta) * VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = c * HW();
								for (auto hw = offset; hw < offset + HW(); hw++)
									InputLayer->NeuronsD1[hw] = Func.df(InputNeurons[hw], Alpha, Beta) * InputLayer->NeuronsD1[hw];
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
									(Func.dfVec(VecFloat().load_a(&InputNeurons[hw]), Alpha, Beta) * VecFloat().load_a(&NeuronsD1[hw])).store_a(&NeuronsD1[hw]);
							}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = c * HW();
								for (auto hw = offset; hw < offset + HW(); hw++)
									NeuronsD1[hw] = Func.df(InputNeurons[hw], Alpha, Beta) * NeuronsD1[hw];
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
										(Func.dfVec(VecFloat().load_a(&InputNeurons[hw]), Alpha, Beta) * VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = n * CDHW() + c * HW();
									for (auto hw = offset; hw < offset + HW(); hw++)
										InputLayer->NeuronsD1[hw] *= Func.df(InputNeurons[hw], Alpha, Beta);
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
										(Func.dfVec(VecFloat().load_a(&InputNeurons[hw]), Alpha, Beta) * VecFloat().load_a(&NeuronsD1[hw])).store_a(&NeuronsD1[hw]);
								}
							});
						else
							for_i(batchSize, threads, [=](UInt n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = n * CDHW() + c * HW();
									for (auto hw = offset; hw < offset + HW(); hw++)
										NeuronsD1[hw] *= Func.df(InputNeurons[hw], Alpha, Beta);
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

			const auto& memDiffDst = !InplaceBwd ? dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			auto diffDstMem = reorderBwdDiffDst ? dnnl::memory(bwdDesc->diff_dst_desc(), Device.engine) : memDiffDst;
			if (reorderBwdDiffDst)
			{
				dnnl::reorder(memDiffDst, diffDstMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memDiffDst}, { DNNL_ARG_TO, diffDstMem } });
				Device.stream.wait();
			}

			auto memMean = dnnl::memory(bwdDesc->mean_desc(), Device.engine, Mean.data());
			auto memVariance = dnnl::memory(bwdDesc->variance_desc(), Device.engine, Variance.data());
			auto memDiffSrc = SharesInput ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

			if (Scaling)
			{
				auto scaleMemory = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
				auto shiftMemory = dnnl::memory(*WeightsMemDesc, Device.engine, Biases.data());
				auto diffScaleMemory = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsD1.data());
				auto diffShiftMemory = dnnl::memory(*WeightsMemDesc, Device.engine, BiasesD1.data());

#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, scaleMemory }, { DNNL_ARG_SHIFT, shiftMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_DIFF_SCALE, diffScaleMemory }, { DNNL_ARG_DIFF_SHIFT, diffShiftMemory } });
#else
				dnnl::batch_normalization_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_SCALE, scaleMemory }, { DNNL_ARG_SHIFT, shiftMemory }, { DNNL_ARG_DIFF_SRC, diffSrcMem }, { DNNL_ARG_DIFF_SCALE, diffScaleMemory }, { DNNL_ARG_DIFF_SHIFT, diffShiftMemory } });
#endif
			}
			else
#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::batch_normalization_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_MEAN, memMean }, { DNNL_ARG_VARIANCE, memVariance }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
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
		
		ByteArray GetImage(const Byte fillColor) final override
		{
			if (Scaling && BiasCount > 0)
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

			Layer::Save(os, persistOptimizer, optimizer);
		}

		void Load(std::istream& is, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			is.read(reinterpret_cast<char*>(RunningMean.data()), std::streamsize(C * sizeof(Float)));
			is.read(reinterpret_cast<char*>(RunningVariance.data()), std::streamsize(C * sizeof(Float)));

			Layer::Load(is, persistOptimizer, optimizer);
		}

		std::streamsize GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const override
		{
			return (2 * C * sizeof(Float)) + Layer::GetWeightsSize(persistOptimizer, optimizer);
		}
		
		UInt GetNeuronsSize(const UInt batchSize) const override
		{
			if constexpr (ReferenceBatchNormalization || Reference)
				return Layer::GetNeuronsSize(batchSize) + (batchSize * PaddedCDHW() * sizeof(Float));
			else
				return Layer::GetNeuronsSize(batchSize);
		}
	};
}