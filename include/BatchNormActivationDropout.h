#pragma once
#include "Layer.h"
#include "Activation.h"

namespace dnn
{
	template <typename Activation = HardSwish, typename LayerTypes T = LayerTypes::BatchNormHardSwishDropout>
	class BatchNormActivationDropout final : public Layer
	{
	public:
		const bool Scaling;
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;
		const Float Keep;
		const Float Scale;

		FloatVector NeuronsActive;
		std::bernoulli_distribution DropoutDistribution;

		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;
	
		BatchNormActivationDropout<Activation, T>(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const Float dropout = Float(0.5), const bool scaling = true, const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true) :
			Layer(device, format, name, T, inputs[0]->C, inputs[0]->C, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, hasBias),
			Scaling(scaling),
			Eps(eps),
			Momentum(momentum),
			OneMinusMomentum(Float(1) - momentum),
			Keep(Float(1) - dropout),
			Scale(Float(1) / (Float(1) - dropout))
		{
			assert(Inputs.size() == 1);

			Mean = FloatVector(PaddedC, Float(0));
			RunningMean = FloatVector(PaddedC, Float(0));
			Variance = FloatVector(PaddedC, Float(1));
			RunningVariance = FloatVector(PaddedC, Float(1));
			InvStdDev = FloatVector(PaddedC);

			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 2, dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));
			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 2, dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc));

			DropoutDistribution = std::bernoulli_distribution(Keep);
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

		size_t FanIn() const final override
		{
			return 1;
		}

		size_t FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				Format = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, Format));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, Format));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
				{
					Format = GetDataFmt(*InputLayer->DstMemDesc);
					if (Format != GetDataFmt(*InputLayer->DiffDstMemDesc))
						throw std::invalid_argument("Src and Diff format are different in " + std::string(magic_enum::enum_name<LayerTypes>(LayerType)) + " layer " + Name);
				}

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
			}
		}

		void SetBatchSize(const size_t batchSize) final override
		{
			Layer::SetBatchSize(batchSize);

			ZeroFloatVectorAllocate(NeuronsActive, batchSize * PaddedCDHW);
			for (auto n = 0ull; n < batchSize; n++)
				for (auto i = 0ull; i < CDHW; i++)
					NeuronsActive[n * PaddedCDHW + i] = Float(1);
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto strideH = W * VectorSize;

			if (!training)
			{
				if (IsPlainFormat()) // nchw
				{
					const auto partialHW = (HW / VectorSize) * VectorSize;

					for_i(C, [=](size_t c)
					{
						const auto runningMean = RunningMean[c];
						const auto invStdDev = Float(1) / std::sqrt(RunningVariance[c] + Eps);

						const auto weightedInvStdDev = Scaling ? invStdDev * Weights[c] : invStdDev;
						const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto i = start; i < part; i += VectorSize)
								Activation::fVec((VecFloat().load_a(&InputLayer->Neurons[i]) - runningMean) * weightedInvStdDev + biases).store_a(&Neurons[i]);
							const auto end = start + HW;
							for (auto i = part; i < end; i++)
								Neurons[i] = Activation::f(((InputLayer->Neurons[i]) - runningMean) * weightedInvStdDev + biases);
						}
					});
				}
				else
				{
					for_i(PaddedC / VectorSize, [=](size_t c)
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
					const auto partialHW = (HW / VectorSize) * VectorSize;

					for_i(C, [=](size_t c)
					{
						auto vecMean = VecFloat(0);
						auto mean = Float(0);
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto i = start; i < part; i += VectorSize)
								vecMean += VecFloat().load_a(&InputLayer->Neurons[i]);
							const auto end = start + HW;
							for (auto i = part; i < end; i++)
								mean += InputLayer->Neurons[i];
						}
						mean += horizontal_add(vecMean);
						mean /= Float(batchSize * HW);

						auto vecVariance = VecFloat(0);
						auto variance = Float(0);
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto i = start; i < part; i += VectorSize)
								vecVariance += square(VecFloat().load_a(&InputLayer->Neurons[i]) - mean);
							const auto end = start + HW;
							for (auto i = part; i < end; i++)
								variance += FloatSquare(InputLayer->Neurons[i] - mean);
						}
						variance += horizontal_add(vecVariance);
						const auto unbiasedVariance = variance / Float(batchSize * HW - 1);
						variance /= Float(batchSize * HW);

						RunningMean[c] = RunningMean[c] * Momentum + OneMinusMomentum * mean;
						RunningVariance[c] = RunningVariance[c] * Momentum + OneMinusMomentum * unbiasedVariance;

						const auto invStdDev = Float(1) / std::sqrt(variance + Eps);
						const auto weightedInvStdDev = Scaling ? invStdDev * Weights[c] : invStdDev;
						const auto biases = Scaling && HasBias ? Biases[c] : Float(0);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto start = c * HW + (n * CDHW);
							const auto part = start + partialHW;
							for (auto i = start; i < part; i += VectorSize)
							{
								const auto neuronsActive = BernoulliVecFloat(Keep);
								neuronsActive.store_a(&NeuronsActive[i]);
								(neuronsActive * Scale * Activation::fVec(((VecFloat().load_a(&InputLayer->Neurons[i]) - mean) * weightedInvStdDev + biases))).store_a(&Neurons[i]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[i]);
#endif
							}
							const auto end = start + HW;
							for (auto i = part; i < end; i++)
							{
								NeuronsActive[i] = Bernoulli<Float>(Keep);
								Neurons[i] = NeuronsActive[i] * Scale * Activation::f((InputLayer->Neurons[i] - mean) * weightedInvStdDev + biases);
#ifndef DNN_LEAN
								NeuronsD1[i] = Float(0);
#endif
							}
						}
					});
				}
				else 
				{
					for_i(PaddedC / VectorSize, [=](size_t c)
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

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto offsetC = n * PaddedCDHW + mapOffset;
							for (auto h = 0ull; h < H; h++)
							{
								const auto offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									const auto neuronsActive = BernoulliVecFloat(Keep);
									neuronsActive.store_a(&NeuronsActive[w]);

									(neuronsActive * Scale * Activation::fVec(mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, weightedInvStdDev, biases))).store_a(&Neurons[w]);
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

		void BackwardProp(const size_t batchSize)  final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto strideH = W * VectorSize;

			if (IsPlainFormat())
			{
				const auto partialHW = (HW / VectorSize) * VectorSize;
				
				for_i(C, [=](size_t c)
				{
					const auto mean = Mean[c];

					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);

					auto diffGammaFloat = Float(0);
					auto diffBetaFloat = Float(0);
					auto diffSrcFloat = Float(0);
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto start = c * HW + (n * CDHW);
						const auto part = start + partialHW;
						for (auto i = start; i < part; i += VectorSize)
						{
							diffSrc = VecFloat().load_a(&NeuronsActive[i]) * Activation::dfVec(VecFloat().load_a(&Neurons[i])) * VecFloat().load_a(&NeuronsD1[i]);

							diffGamma = mul_add(VecFloat().load_a(&InputLayer->Neurons[i]) - mean, diffSrc, diffGamma);
							diffBeta += diffSrc;
						}
						const auto end = start + HW;
						for (auto i = part; i < end; i++)
						{
							diffSrcFloat = NeuronsActive[i] * Activation::df(Neurons[i]) * NeuronsD1[i];

							diffGammaFloat += (InputLayer->Neurons[i] - mean) * diffSrcFloat;
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

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto start = c * HW + (n * CDHW);
						const auto part = start + partialHW;
						for (auto i = start; i < part; i += VectorSize)
						{
							diffSrc = VecFloat().load_a(&NeuronsActive[i]) * Activation::dfVec(VecFloat().load_a(&Neurons[i])) * VecFloat().load_a(&NeuronsD1[i]);

							// if not using global stats!
							diffSrc -= mul_add(VecFloat().load_a(&InputLayer->Neurons[i]) - mean, diffGammaFloat, diffBetaFloat);

							//diffSrc *= gamma;
							mul_add(diffSrc, gamma, VecFloat().load_a(&InputLayer->NeuronsD1[i])).store_a(&InputLayer->NeuronsD1[i]);
						}

						const auto end = start + HW;
						for (auto i = part; i < end; i++)
						{
							diffSrcFloat = NeuronsActive[i] * Activation::df(Neurons[i]) * NeuronsD1[i];

							// if not using global stats!
							diffSrcFloat -= (InputLayer->Neurons[i] - mean) * diffGammaFloat + diffBetaFloat;

							//diffSrc *= gamma;
							InputLayer->NeuronsD1[i] += diffSrcFloat * gamma;
						}
					}
				});
			}
			else
			{
				for_i(PaddedC / VectorSize, [=](size_t c)
				{
					const auto channelOffset = c * VectorSize;
					const auto mapOffset = channelOffset * HW;

					const auto mean = VecFloat().load_a(&Mean[channelOffset]);
					auto diffGamma = VecFloat(0);
					auto diffBeta = VecFloat(0);
					auto diffSrc = VecFloat(0);

					for (size_t n = 0; n < batchSize; ++n)
					{
						const auto offsetC = n * PaddedCDHW + mapOffset;
						for (auto h = 0ull; h < H; ++h)
						{
							const auto offsetH = offsetC + h * strideH;

							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								diffSrc = VecFloat().load_a(&NeuronsActive[w]) * Activation::dfVec(VecFloat().load_a(&Neurons[w])) * VecFloat().load_a(&NeuronsD1[w]);

								diffGamma = mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, diffSrc, diffGamma);
								diffBeta += diffSrc;
							}
						}
					}

					const auto invStdDev = VecFloat().load_a(&InvStdDev[channelOffset]);

					diffGamma *= invStdDev;

					if (Scaling)
					{
						(VecFloat().load_a(&WeightsD1[channelOffset]) += diffGamma).store_a(&WeightsD1[channelOffset]);
						(VecFloat().load_a(&BiasesD1[channelOffset]) += diffBeta).store_a(&BiasesD1[channelOffset]);
					}

					diffGamma *= invStdDev / Float(batchSize * HW);
					diffBeta /= Float(batchSize * HW);

					const auto gamma = Scaling ? VecFloat().load_a(&Weights[channelOffset]) * invStdDev : invStdDev;

					for (auto n = 0ull; n < batchSize; ++n)
					{
						const auto offsetC = n * PaddedCDHW + mapOffset;
						for (auto h = 0ull; h < H; ++h)
						{
							const auto offsetH = offsetC + h * strideH;

							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								diffSrc = VecFloat().load_a(&NeuronsActive[w]) * Activation::dfVec(VecFloat().load_a(&Neurons[w])) * VecFloat().load_a(&NeuronsD1[w]);

								// if not using global stats!
								diffSrc -= mul_add(VecFloat().load_a(&InputLayer->Neurons[w]) - mean, diffGamma, diffBeta);

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

		ByteVector GetImage(const Byte fillColor)  final override
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

		void ResetWeights(const Fillers weightFiller, const Float weightFillerScale, const Fillers biasFiller, const Float biasFillerScale) override
		{
			Weights = FloatVector(PaddedC, Float(1));
			Biases = FloatVector(PaddedC, Float(0));

			RunningMean = FloatVector(PaddedC, Float(0));
			RunningVariance = FloatVector(PaddedC, Float(1));
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

		size_t GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const override
		{
			return (2 * C * sizeof(Float)) + Layer::GetWeightsSize(persistOptimizer, optimizer);
		}

		size_t GetNeuronsSize(const size_t batchSize) const override
		{
			size_t totalSize = 0;

#ifndef DNN_LEAN
			totalSize += PaddedCDHW * batchSize * sizeof(Float) * 3;
#else
			totalSize += PaddedCDHW * batchSize * sizeof(Float) * 2;
#endif // DNN_LEAN

			return totalSize;
		}
	};

	typedef BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout> BatchNormHardSwishDropout;
	typedef BatchNormActivationDropout<Relu, LayerTypes::BatchNormReluDropout> BatchNormReluDropout;
}