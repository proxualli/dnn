#pragma once
#include "Layer.h"
#include "Activation.h"

namespace dnn
{
	template <typename Activation = HardSwish, typename dnn::LayerTypes T = LayerTypes::BatchNormHardSwishDropout>
	class BatchNormActivationDropout final : public Layer
	{
	private:
		auto GetAlpha(const Activations activation, const Float alpha, const Float beta) const
		{
			switch (activation)
			{
			case Activations::Abs:
			case Activations::ASinh:
			case Activations::Clip:
			case Activations::ClipV2:
			case Activations::Exp:
			case Activations::Gelu:
			case Activations::GeluErf:
			case Activations::Log:
			case Activations::Logistic:
			case Activations::LogLogistic:
			case Activations::Mish:
			case Activations::Pow:
			case Activations::Relu:
			case Activations::Round:
			case Activations::SoftRelu:
			case Activations::SoftSign:
			case Activations::Sqrt:
			case Activations::Square:
			case Activations::Tanh:
			case Activations::TanhExp:
				break;
			case Activations::BoundedRelu:
				return alpha == Float(0) ? Float(6) : alpha;
			case Activations::Elu:
			case Activations::Linear:
			case Activations::Swish:
				return alpha == Float(0) ? Float(1) : alpha;
			case Activations::SoftPlus:
				return alpha == Float(0) ? Float(20) : alpha;
			case Activations::HardLogistic:
				return alpha == Float(0) ? Float(0.2) : alpha;
			case Activations::HardSwish:
				return alpha == Float(0) ? Float(3) : alpha;
			}

			return alpha;
		}

		auto GetBeta(const Activations activation, const Float alpha, const Float beta) const
		{
			switch (activation)
			{
			case Activations::Abs:
			case Activations::ASinh:
			case Activations::Clip:
			case Activations::ClipV2:
			case Activations::Elu:
			case Activations::Exp:
			case Activations::Gelu:
			case Activations::GeluErf:
			case Activations::Linear:
			case Activations::Log:
			case Activations::LogLogistic:
			case Activations::Logistic:
			case Activations::Mish:
			case Activations::Pow:
			case Activations::Relu:
			case Activations::Round:
			case Activations::SoftRelu:
			case Activations::SoftSign:
			case Activations::Sqrt:
			case Activations::Square:
			case Activations::Swish:
			case Activations::Tanh:
			case Activations::TanhExp:
				break;
			case Activations::BoundedRelu:
				return Float(0);
			case Activations::HardLogistic:
				return beta == Float(0) ? Float(0.5) : beta;
			case Activations::HardSwish:
				return beta == Float(0) ? Float(6) : beta;
			case Activations::SoftPlus:
				return beta == Float(0) ? Float(1) : beta;
			}

			return beta;
		}

	public:
		const bool LocalValue;
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;
		const Float Alpha;
		const Float Beta;
		Float Keep;
		Float Scale;
		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;
		FloatArray NeuronsActive;

		BatchNormActivationDropout<Activation, T>(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.5), const bool localValue = false, const bool scaling = true, const Float alpha = Float(0), const Float beta = Float(0), const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true) :
			Layer(device, format, name, T, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias, scaling, dropout > 0),
			LocalValue(localValue),
			Alpha(GetAlpha(Activation::Enum(), alpha, beta)),
			Beta(GetBeta(Activation::Enum(), alpha, beta)),
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

			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x));
			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x));
		}
				
		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
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
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Alpha:") + dtab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:") + dtab + FloatToString(Beta));
			description += GetWeightsDescription(Scaling);
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

		void SetBatchSize(const UInt batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			if (Enabled)
			{
				NeuronsActive.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
				for (auto n = 0ull; n < batchSize; n++)
					for (auto i = 0ull; i < CDHW(); i++)
						NeuronsActive[n * PaddedCDHW() + i] = Float(1);
			}
			else
				NeuronsActive.release();
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto strideH = W * VectorSize;
			const auto plain = IsPlainFormat();
			const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());

			if (!training)
			{
				const auto maxTreads = GetThreads(elements, Float(5));

				if (plain) // nchw
				{
					const auto partialHW = GetVectorPart(HW());
					const auto threads = std::min<UInt>(maxTreads, C);

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
								Activation::fVec((VecFloat().load_a(&InputLayer->Neurons[hw]) - RunningMean[c]) * weightedInvStdDev + biases, Alpha, Beta).store_a(&Neurons[hw]);
							for (auto hw = part; hw < start + HW(); hw++)
								Neurons[hw] = Activation::f((InputLayer->Neurons[hw] - RunningMean[c]) * weightedInvStdDev + biases, Alpha, Beta);
						}
					});
				}
				else
				{
					const auto threads = std::min<UInt>(maxTreads, PaddedC / VectorSize);

					for_i(PaddedC / VectorSize, threads, [=](UInt c)
					{
						const auto channelOffset = c * VectorSize;
						const auto mapOffset = channelOffset * HW();

						const auto runningMean = VecFloat().load_a(&RunningMean[channelOffset]);
						const auto invStddev = VecFloat(1) / sqrt(VecFloat().load_a(&RunningVariance[channelOffset]) + Eps);

						const auto weightedInvStdDev = Scaling ? (VecFloat().load_a(&Weights[channelOffset]) * invStddev): invStddev;
						const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto offsetC = n * PaddedCDHW() + mapOffset;
							for (auto h = 0ull; h < H; h++)
							{
								const auto offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - runningMean, weightedInvStdDev, biases), Alpha, Beta).store_a(&Neurons[w]);
							}
						}
					});
				}
			}
			else
			{
				const auto maxTreads = GetThreads(elements, Float(10));

				if (plain)
				{
					const auto partialHW = GetVectorPart(HW());
					const auto threads = std::min<UInt>(maxTreads, C);

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
							unbiasedVariance = std::max(0.f, variance / Float(batchSize * HW() - 1));
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

						if (Enabled)
						{
							VecFloat mask;
							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										mask = BernoulliVecFloat(Keep);
										mask.store_a(&NeuronsActive[hw]);
										(mask * Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases), Alpha, Beta)).store_a(&Neurons[hw]);
									}
									const auto end = start + HW();
									for (auto hw = part; hw < end; hw++)
									{
										NeuronsActive[hw] = Bernoulli<Float>(Keep);
										Neurons[hw] = NeuronsActive[hw] * Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases, Alpha, Beta);
									}
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										mask = BernoulliVecFloat(Keep);
										mask.store_a(&NeuronsActive[hw]);
										(mask * Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases), Alpha, Beta)).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
										VecFloat(0).store_nt(&NeuronsD1[hw]);
#endif
									}
									const auto end = start + HW();
									for (auto hw = part; hw < end; hw++)
									{
										NeuronsActive[hw] = Bernoulli<Float>(Keep);
										Neurons[hw] = NeuronsActive[hw] * Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases, Alpha, Beta);
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
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										(Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases), Alpha, Beta)).store_a(&Neurons[hw]);
									}
									const auto end = start + HW();
									for (auto hw = part; hw < end; hw++)
										Neurons[hw] = Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases, Alpha, Beta);
								}
							else
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto start = c * HW() + (n * CDHW());
									const auto part = start + partialHW;
									for (auto hw = start; hw < part; hw += VectorSize)
									{
										(Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[hw]) - mean) * weightedInvStdDev + biases), Alpha, Beta)).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
										VecFloat(0).store_nt(&NeuronsD1[hw]);
#endif
									}
									const auto end = start + HW();
									for (auto hw = part; hw < end; hw++)
									{
										Neurons[hw] = Scale * Activation::f((InputLayer->Neurons[hw] - mean) * weightedInvStdDev + biases, Alpha, Beta);
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
					const auto threads = std::min<UInt>(maxTreads, PaddedC / VectorSize);

					for_i(PaddedC / VectorSize, threads, [=](UInt c)
					{
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
						}

						variance = max(VecFloat(0), variance);
						variance.store_a(&Variance[channelOffset]);

						mul_add(VecFloat().load_a(&RunningMean[channelOffset]), Momentum, OneMinusMomentum * mean).store_a(&RunningMean[channelOffset]);
						mul_add(VecFloat().load_a(&RunningVariance[channelOffset]), Momentum, OneMinusMomentum * unbiasedVariance).store_a(&RunningVariance[channelOffset]);

						const auto invStddev = VecFloat(1) / sqrt(variance + Eps);
						const auto weightedInvStdDev = Scaling ? (VecFloat().load_a(&Weights[channelOffset]) * invStddev) : invStddev;
						const auto biases = Scaling && HasBias ? VecFloat().load_a(&Biases[channelOffset]) : VecFloat(0);

						invStddev.store_a(&InvStdDev[channelOffset]);

						if (Enabled)
						{
							VecFloat mask;
							if (InplaceBwd)
								for (auto n = 0ull; n < batchSize; n++)
								{
									const auto offsetC = n * PaddedCDHW() + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
										{
											mask = BernoulliVecFloat(Keep);
											mask.store_a(&NeuronsActive[w]);
											(mask * Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases), Alpha, Beta)).store_a(&Neurons[w]);
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
											mask = BernoulliVecFloat(Keep);
											mask.store_a(&NeuronsActive[w]);
											(mask * Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases), Alpha, Beta)).store_a(&Neurons[w]);
#ifndef DNN_LEAN
											VecFloat(0).store_nt(&NeuronsD1[w]);
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
									const auto offsetC = n * PaddedCDHW() + mapOffset;
									for (auto h = 0ull; h < H; h++)
									{
										const auto offsetH = offsetC + h * strideH;
										for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
											(Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases), Alpha, Beta)).store_a(&Neurons[w]);
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
											(Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases), Alpha, Beta)).store_a(&Neurons[w]);
#ifndef DNN_LEAN
											VecFloat(0).store_nt(&NeuronsD1[w]);
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
			const auto plain = IsPlainFormat();
			const auto elements = batchSize * (plain ? CDHW() : PaddedCDHW());
			const auto threads = GetThreads(elements, Float(10));

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
							diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec(inputNeurons * weightedInvStdDev + biases, Alpha, Beta) * VecFloat().load_a(&layerD1[hw]);
							KahanSum<VecFloat>(diffSrc * inputNeurons, diffGamma, correction0);
							KahanSum<VecFloat>(diffSrc, diffBeta, correction1);
						}
						for (auto hw = part; hw < start + HW(); hw++)
						{
							diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df(((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev) + biases, Alpha, Beta) * layerD1[hw];
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
								diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases, Alpha, Beta) * (InplaceBwd ? VecFloat().load_a(&InputLayer->NeuronsD1[hw]) : VecFloat().load_a(&NeuronsD1[hw]));

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat(0)).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
								diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev + biases, Alpha, Beta) * InputLayer->NeuronsD1[hw];

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
								diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[hw]) : VecFloat(1)) * Activation::dfVec((VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c]) * weightedInvStdDev + biases, Alpha, Beta) * VecFloat().load_a(&NeuronsD1[hw]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[hw]) - Mean[c], diffGammaFloat, diffBetaFloat);

								//diffSrc *= gamma;
								mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
							for (auto hw = part; hw < start + HW(); hw++)
							{
								diffSrcFloat = (enabled ? NeuronsActive[hw] : Float(1)) * Activation::df((InputLayerFwd->Neurons[hw] - Mean[c]) * weightedInvStdDev + biases, Alpha, Beta) * NeuronsD1[hw];

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
								diffSrc.load_a(&layerD1[w]);
								inputNeurons.load_a(&InputLayerFwd->Neurons[w]);
								inputNeurons -= mean;
								diffSrc *= (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(inputNeurons, weightedInvStdDev, biases), Alpha, Beta);
								KahanSum<VecFloat>(diffSrc * inputNeurons, diffGamma, correction0);
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
									diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, weightedInvStdDev, biases), Alpha, Beta) * VecFloat().load_a(&InputLayer->NeuronsD1[w]);

									// if not using global stats!
									diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, diffGamma, diffBeta);

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
									diffSrc = (enabled ? VecFloat().load_a(&NeuronsActive[w]) : VecFloat(1)) * Activation::dfVec(mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, weightedInvStdDev, biases), Alpha, Beta) * VecFloat().load_a(&NeuronsD1[w]);

									// if not using global stats!
									diffSrc -= mul_add(VecFloat().load_a(&InputLayerFwd->Neurons[w]) - mean, diffGamma, diffBeta);

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
#ifndef DNN_LEAN
			return batchSize * PaddedCDHW() * sizeof(Float) * (InplaceBwd ? 2 : 3);
#else
			return batchSize * PaddedCDHW() * sizeof(Float) * (InplaceBwd ? 1 : 2);
#endif // DNN_LEAN
		}
	};
}