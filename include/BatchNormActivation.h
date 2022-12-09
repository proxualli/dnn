#pragma once
#include "Activation.h"

namespace dnn
{
	template <typename Activation = HardSwish, typename dnn::LayerTypes T = LayerTypes::BatchNormHardSwish>
	class BatchNormActivation final : public Layer
	{
	public:
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;
		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;

		BatchNormActivation<Activation,T>(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool scaling = true, const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true) :
			Layer(device, format, name, T, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias, scaling),
			Eps(eps),
			Momentum(momentum),
			OneMinusMomentum(Float(1) - momentum),
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
			}
		}

		bool Lockable() const final override
		{
			return WeightCount > 0 && Scaling;
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
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

		void BackwardProp(const UInt batchSize) final override
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
				
					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);
					auto correction0 = VecFloat(0);
					auto correction1 = VecFloat(0);

					auto diffGammaFloat = Float(0);
					auto diffBetaFloat = Float(0);
					auto diffSrcFloat = Float(0);
					auto correction0Float = Float(0);
					auto correction1Float = Float(0);
					
					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW() + (n * CDHW());
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&InputLayer->NeuronsD1[hw]);

								//diffGamma = mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffSrc, diffGamma);
								KahanSum<VecFloat>(diffSrc * (VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]), diffGamma, correction0);

								// diffBeta += diffSrc;
								KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * InputLayer->NeuronsD1[hw];

								//diffGamma = mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffSrc, diffGamma);
								KahanSum<Float>(diffSrcFloat * (InputLayerFwd->Neurons[hw] - Mean[c]), diffGammaFloat, correction0Float);

								// diffBeta += diffSrc;
								KahanSum<Float>(diffSrcFloat, diffBetaFloat, correction1Float);
							}
						}
					else
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW() + (n * CDHW());
							const auto part = start + partialHW;
							for (auto hw = start; hw < part; hw += VectorSize)
							{
								diffSrc = Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&NeuronsD1[hw]);

								//diffGamma = mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffSrc, diffGamma);
								KahanSum<VecFloat>(diffSrc * (VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]), diffGamma, correction0);

								// diffBeta += diffSrc;
								KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * NeuronsD1[hw];

								//diffGamma = mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffSrc, diffGamma);
								KahanSum<Float>(diffSrcFloat * (InputLayerFwd->Neurons[hw] - Mean[c]), diffGammaFloat, correction0Float);

								// diffBeta += diffSrc;
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
								diffSrc = Activation::dfVec((VecFloat().load_a(&InputLayer->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&InputLayer->NeuronsD1[hw]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayer->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{

								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * InputLayer->NeuronsD1[hw];

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
								diffSrc = Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases) * VecFloat().load_a(&NeuronsD1[hw]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
							
								diffSrcFloat = Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases) * NeuronsD1[hw];

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
					auto correction0 = VecFloat(0);
					auto correction1 = VecFloat(0);
					if (InplaceBwd)
						for (auto n = 0ull; n < batchSize; ++n)
						{
							const auto offsetC = n * PaddedCDHW() + mapOffset;
							for (auto h = 0ull; h < H; ++h)
							{
								const auto offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									diffSrc = Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, weightedInvStdDev, biases)) * VecFloat().load_a(&InputLayer->NeuronsD1[w]);

									// diffGamma = mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, diffSrc, diffGamma);
									KahanSum<VecFloat>(diffSrc * (VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean), diffGamma, correction0);

									// diffBeta += diffSrc;
									KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
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
									diffSrc = Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, weightedInvStdDev, biases)) * VecFloat().load_a(&NeuronsD1[w]);

									// diffGamma = mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, diffSrc, diffGamma);
									KahanSum<VecFloat>(diffSrc * (VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean), diffGamma, correction0);

									// diffBeta += diffSrc;
									KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
								}
							}
						}

					diffGamma *= invStdDev;

					if (Scaling)
					{
						(VecFloat().load_a(&WeightsD1[channelOffset]) += diffGamma).store_a(&WeightsD1[channelOffset]);
						(VecFloat().load_a(&BiasesD1[channelOffset]) += diffBeta).store_a(&BiasesD1[channelOffset]);
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
									diffSrc = mul_add(Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, weightedInvStdDev, biases)), VecFloat().load_a(&InputLayer->NeuronsD1[w]), -mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, diffGamma, diffBeta));

									//diffSrc *= gamma;
									mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[w]);
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
									diffSrc = mul_add(Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, weightedInvStdDev, biases)), VecFloat().load_a(&NeuronsD1[w]), -mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, diffGamma, diffBeta));

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
	};
}