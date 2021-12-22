#pragma once
#include "Layer.h"
#include "Activation.h"

namespace dnn
{
	template <typename Activation = HardSwish, typename dnn::LayerTypes T = LayerTypes::BatchNormHardSwishDropout>
	class BatchNormActivationDropout final : public Layer
	{
	public:
		const bool LocalValue;
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;
		Float Keep;
		Float Scale;
		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;
		FloatArray NeuronsActive;

		BatchNormActivationDropout<Activation, T>(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.5), const bool localValue = false, const bool scaling = true, const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true) :
			Layer(device, format, name, T, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias, scaling, dropout > 0),
			LocalValue(localValue),
			Eps(eps),
			Momentum(momentum),
			OneMinusMomentum(Float(1) - momentum),
			Keep(Float(1) - dropout),
			Scale(Float(1) / (Float(1) - dropout)),
			Mean(FloatVector(PaddedC, Float(0))),
			RunningMean(FloatVector(PaddedC, Float(0))),
			Variance(FloatVector(PaddedC, Float(1))),
			RunningVariance(FloatVector(PaddedC, Float(1))),
			InvStdDev(FloatVector(PaddedC))
		{
			assert(Inputs.size() == 1);			

			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 2, dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 2, dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
		}
				
		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
			Layer::UpdateResolution();
		}

		void UpdateDropout(const Float dropout)
		{
			if (!LocalValue)
			{
				Enabled = dropout > 0;
				Keep = Float(1) - dropout;
				Scale = Float(1) / Keep;
			}
		}

		bool Lockable() const final override
		{
			return WeightCount > 0 && Scaling;
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

			description.append(nwl + std::string(" Dropout:") + tab + FloatToString(Float(1) - Keep));
			description.append(nwl + std::string(" Scale:") + dtab + FloatToString(Scale));

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
			}
		}

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			if (Enabled)
			{
				NeuronsActive.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto n = 0ull; n < batchSize; n++)
					for (auto i = 0ull; i < CDHW; i++)
						NeuronsActive[n * PaddedCDHW + i] = Float(1);
			}
			else
				NeuronsActive.release();
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto strideH = W * VectorSize;

			if (!training)
			{
				if (IsPlainFormat()) // nchw
				{
					const auto partialHW = GetVectorPart(HW);

					for_i(C, [=](UInt c)
					{
						const auto runningMean = RunningMean[c];
						const auto invStdDev = Float(1) / std::sqrt(RunningVariance[c] + Eps);

						const auto weightedInvStdDev = Scaling ? invStdDev * Weights[c] : invStdDev;
						const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
								Activation::fVec((VecFloat().load_a(&InputLayer->Neurons[hw]) - runningMean) * weightedInvStdDev + biases).store_a(&Neurons[hw]);
							for (auto hw = part; hw < start + HW; hw++)
								Neurons[hw] = Activation::f(((InputLayer->Neurons[hw]) - runningMean) * weightedInvStdDev + biases);
						}
					});
				}
				else
				{
					for_i(PaddedC / VectorSize, [=](UInt c)
					{
						const auto channelOffset = c * VectorSize;
						const auto mapOffset = channelOffset * HW;

						const auto runningMean = VecFloat().load_a(&RunningMean[channelOffset]);
						const auto invStdDev = VecFloat(1) / sqrt(VecFloat().load_a(&RunningVariance[channelOffset]) + Eps);

						const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]) : invStdDev;
						const auto biases = Scaling &&  HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
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
#ifndef DNN_LEAN
				const auto vecZero = VecFloat(0);
#endif
				if (IsPlainFormat())
				{
					const auto partialHW = GetVectorPart(HW);

					for_i(C, [=](UInt c)
					{
						auto vecMean = VecFloat(0);
						auto mean = Float(0);
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
								vecMean += VecFloat().load_a(&InputLayer->Neurons[hw]);
							const auto end = start + HW;
							PRAGMA_OMP_SIMD()
							for (auto hw = part; hw < end; hw++)
								mean += InputLayer->Neurons[hw];
						}
						mean += horizontal_add(vecMean);
						mean /= Float(batchSize * HW);

						auto vecVariance = VecFloat(0);
						auto variance = Float(0);
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
								vecVariance += square(VecFloat().load_a(&InputLayer->Neurons[hw]) - mean);
							const auto end = start + HW;
							PRAGMA_OMP_SIMD()
							for (auto hw = part; hw < end; hw++)
								variance += FloatSquare(InputLayer->Neurons[hw] - mean);
						}
						variance += horizontal_add(vecVariance);
						const auto unbiasedVariance = variance / Float(batchSize * HW - 1);
						variance /= Float(batchSize * HW);

						RunningMean[c] = RunningMean[c] * Momentum + OneMinusMomentum * mean;
						RunningVariance[c] = RunningVariance[c] * Momentum + OneMinusMomentum * unbiasedVariance;

						const auto invStdDev = Float(1) / std::sqrt(variance + Eps);
						const auto weightedInvStdDev = Scaling ? invStdDev * Weights[c] : invStdDev;
						const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

						if (Enabled)
						{
							VecFloat mask;
							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW + (n * CDHW);
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										mask = BernoulliVecFloat(Keep);
										mask.store_a(&NeuronsActive[hw]);
										(mask * Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases))).store_a(&Neurons[hw]);
									}
									const auto end = start + HW;
									for (auto hw = part; hw < end; hw++)
									{
										NeuronsActive[hw] = Bernoulli<Float>(Keep);
										Neurons[hw] = NeuronsActive[hw] * Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases);
									}
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW + (n * CDHW);
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										mask = BernoulliVecFloat(Keep);
										mask.store_a(&NeuronsActive[hw]);
										(mask * Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases))).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[hw]);
#endif
									}
									const auto end = start + HW;
									for (auto hw = part; hw < end; hw++)
									{
										NeuronsActive[hw] = Bernoulli<Float>(Keep);
										Neurons[hw] = NeuronsActive[hw] * Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases);
#ifndef DNN_LEAN
										NeuronsD1[hw] = Float(0);
#endif
									}
								}
						}
						else
						{
							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW + (n * CDHW);
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										(Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases))).store_a(&Neurons[hw]);
									}
									const auto end = start + HW;
									for (auto hw = part; hw < end; hw++)
										Neurons[hw] = Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases);
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW + (n * CDHW);
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{	
										(Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases))).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[hw]);
#endif
									}
									const auto end = start + HW;
									for (auto hw = part; hw < end; hw++)
									{
										Neurons[hw] = Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases);
#ifndef DNN_LEAN
										NeuronsD1[hw] = Float(0);
#endif
									}
								}
						}
					});
				}
				else 
				{
					for_i(PaddedC / VectorSize, [=](UInt c)
					{
						const auto channelOffset = c * VectorSize;
						const auto mapOffset = channelOffset * HW;

						auto mean = VecFloat(0);
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; h++)
							{
								const auto offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									mean += VecFloat().load_a(&InputLayer->Neurons[w]);
							}
						}
						mean /= Float(batchSize * HW);
						mean.store_a(&Mean[channelOffset]);

						auto variance = VecFloat(0);
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; h++)
							{
								const auto offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									variance += square(VecFloat().load_a(&InputLayer->Neurons[w]) - mean);
							}
						}
						const auto unbiasedVariance = variance / Float(batchSize * HW - 1);
						variance /= Float(batchSize * HW);
						variance.store_a(&Variance[channelOffset]);

						mul_add(VecFloat().load_a(&RunningMean[channelOffset]), Momentum, OneMinusMomentum * mean).store_a(&RunningMean[channelOffset]);
						mul_add(VecFloat().load_a(&RunningVariance[channelOffset]), Momentum, OneMinusMomentum * unbiasedVariance).store_a(&RunningVariance[channelOffset]);

						const auto invStdDev = VecFloat(1) / sqrt(variance + Eps);
						invStdDev.store_a(&InvStdDev[channelOffset]);

						const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]) : invStdDev;
						const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

						if (Enabled)
						{
							VecFloat mask;
							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										{
											mask = BernoulliVecFloat(Keep);
											mask.store_a(&NeuronsActive[w]);
											(mask * Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases))).store_a(&Neurons[w]);
										}
									}
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										{
											mask = BernoulliVecFloat(Keep);
											mask.store_a(&NeuronsActive[w]);
											(mask * Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases))).store_a(&Neurons[w]);
#ifndef DNN_LEAN
											vecZero.store_nt(&NeuronsD1[w]);
#endif
										}
									}
								}
						}
						else
						{
							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
											(Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases))).store_a(&Neurons[w]);
									}
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										{
											(Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases))).store_a(&Neurons[w]);
#ifndef DNN_LEAN
											vecZero.store_nt(&NeuronsD1[w]);
#endif
										}
									}
								}
						}
					});
				}
			}
		}

		void BackwardProp(const UInt batchSize)  final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto strideH = W * VectorSize;
			const auto enabled = Enabled;

			if (IsPlainFormat())
			{
				const auto partialHW = GetVectorPart(HW);
				
				for_i(C, [=](UInt c)
				{
					const auto weightedInvStdDev = Scaling ? InvStdDev[c] * Weights[c] : InvStdDev[c];
					const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);

					auto diffGammaFloat = Float(0);
					auto diffBetaFloat = Float(0);
					auto diffSrcFloat = Float(0);
					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec((VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&InputLayer->NeuronsD1[hw]);

								diffGamma = mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c], diffSrc, diffGamma);
								diffBeta += diffSrc;
							}
							for (auto hw = part; hw < start + HW; hw++)
							{
								diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df((InputLayerOriginal->Neurons[hw] - Mean[c]) * weightedInvStdDev + biases) * InputLayer->NeuronsD1[hw];

								diffGammaFloat += (InputLayerOriginal->Neurons[hw] - Mean[c]) * diffSrcFloat;
								diffBetaFloat += diffSrcFloat;
							}
						}
					else
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec((VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&NeuronsD1[hw]);

								diffGamma = mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c], diffSrc, diffGamma);
								diffBeta += diffSrc;
							}
							for (auto hw = part; hw < start + HW; hw++)
							{
								diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df((InputLayerOriginal->Neurons[hw] - Mean[c]) * weightedInvStdDev + biases) * NeuronsD1[hw];

								diffGammaFloat += (InputLayerOriginal->Neurons[hw] - Mean[c]) * diffSrcFloat;
								diffBetaFloat += diffSrcFloat;
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

					diffGammaFloat *= InvStdDev[c] / Float(batchSize * HW);
					diffBetaFloat /= Float(batchSize * HW);

					const auto gamma = Scaling ? Weights[c] * InvStdDev[c] : InvStdDev[c];
					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec((VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * (InplaceBwd ? VecFloat().load_a(&InputLayer->NeuronsD1[hw]) : VecFloat().load_a(&NeuronsD1[hw]));

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW; hw++)
							{
								diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df((InputLayerOriginal->Neurons[hw] - Mean[c]) * weightedInvStdDev + biases) * InputLayer->NeuronsD1[hw];

								// if not using global stats!
								diffSrcFloat -= (InputLayerOriginal->Neurons[hw] - Mean[c]) * diffGammaFloat + diffBetaFloat;

								//diffSrc *= gamma;
								InputLayer->NeuronsD1[hw] = diffSrcFloat * gamma;
							}
						}
					else
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec((VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&NeuronsD1[hw]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW; hw++)
							{
								diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df((InputLayerOriginal->Neurons[hw] - Mean[c]) * weightedInvStdDev + biases) * NeuronsD1[hw];

								// if not using global stats!
								diffSrcFloat -= (InputLayerOriginal->Neurons[hw] - Mean[c]) * diffGammaFloat + diffBetaFloat;

								//diffSrc *= gamma;
								InputLayer->NeuronsD1[hw] += diffSrcFloat * gamma;
							}
						}
				});
			}
			else
			{
				for_i(PaddedC / VectorSize, [=](UInt c)
				{
					const auto channelOffset = c * VectorSize;
					const auto mapOffset = channelOffset * HW;

					const auto mean = VecFloat().load_a(&Mean[channelOffset]);
					const auto invStdDev = VecFloat().load_a(&InvStdDev[channelOffset]);
					const auto weightedInvStdDev = Scaling ? invStdDev * VecFloat().load_a(&Weights[channelOffset]) : invStdDev;
					const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);

					if (InplaceBwd)
						for (UInt n = 0; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, weightedInvStdDev, biases)) * VecFloat().load_a(&InputLayer->NeuronsD1[w]);

									diffGamma = mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, diffSrc, diffGamma);
									diffBeta += diffSrc;
								}
							}
						}
					else
						for (UInt n = 0; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, weightedInvStdDev, biases)) * VecFloat().load_a(&NeuronsD1[w]);

									diffGamma = mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, diffSrc, diffGamma);
									diffBeta += diffSrc;
								}
							}
						}

					diffGamma *= invStdDev;

					if (Scaling)
					{
						(VecFloat().load_a(&WeightsD1[channelOffset]) += diffGamma).store_a(&WeightsD1[channelOffset]);
						(VecFloat().load_a(&BiasesD1[channelOffset]) += diffBeta).store_a(&BiasesD1[channelOffset]);
					}

					diffGamma *= invStdDev / Float(batchSize * HW);
					diffBeta /= Float(batchSize * HW);

					const auto gamma = Scaling ? VecFloat().load_a(&Weights[channelOffset]) * invStdDev : invStdDev;

					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, weightedInvStdDev, biases)) * VecFloat().load_a(&InputLayer->NeuronsD1[w]);

									// if not using global stats!
									diffSrc -= mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, diffGamma, diffBeta);

									//diffSrc *= gamma;
									mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[w]);
								}
							}
						}
					else
						for (auto n = 0ull; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, weightedInvStdDev, biases)) * VecFloat().load_a(&NeuronsD1[w]);

									// if not using global stats!
									diffSrc -= mul_add(VecFloat().load_a(&InputLayerOriginal->Neurons[w]) - mean, diffGamma, diffBeta);

									//diffSrc *= gamma;
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

		ByteArray GetImage(const Byte fillColor)  final override
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
#ifndef DNN_LEAN
			return batchSize * PaddedCDHW * sizeof(Float) * (InplaceBwd ? 2 : 3);
#else
			return batchSize * PaddedCDHW * sizeof(Float) * (InplaceBwd ? 1 : 2);
#endif // DNN_LEAN
		}
	};
}