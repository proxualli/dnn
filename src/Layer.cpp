#include "Model.h"

namespace dnn
{
	Layer::Layer(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const LayerTypes layerType, const size_t weightCount, const size_t biasCount, const size_t c, const size_t d, const size_t h, const size_t w, const size_t padD, const size_t padH, const size_t padW, const std::vector<Layer*>& inputs, const bool hasBias) :
		Device(device),
		Format(format),
		Name(name),
		LayerType(layerType),
		WeightCount(weightCount),
		BiasCount(biasCount),
		C(c),
		D(d),
		H(h),
		W(w),
		PadD(padD),
		PadH(padH),
		PadW(padW),
		Inputs(inputs),
		HasBias(hasBias&& biasCount > 0),
		HasWeights(weightCount > 0),
		WeightsFiller(Fillers::HeNormal),
		WeightsScale(Float(0.05)),
		WeightsLRM(Float(1)),
		WeightsWDM(Float(1)),
		BiasesFiller(Fillers::Constant),
		BiasesScale(Float(0)),
		BiasesLRM(Float(1)),
		BiasesWDM(Float(1)),
		HW(h * w),
		CDHW(c * d * h * w),
		PaddedC(DivUp(c)),
		PaddedCDHW(layerType != LayerTypes::Input ? (DivUp(c) * d * h * w) : c * d * h * w),
		HasPadding(padD > 0 || padH > 0 || padW > 0),
		InputLayer(inputs.size() > 0 ? inputs[0] : nullptr),
		RandomEngine(std::mt19937(physicalSeed())),
		Neurons(FloatVector()),
		NeuronsD1(FloatVector()),
		Weights(FloatVector(weightCount)),
		WeightsD1(FloatVector(weightCount)),
		Biases(FloatVector(biasCount)),
		BiasesD1(FloatVector(biasCount)),
		WeightsPar1(FloatVector()),
		WeightsPar2(FloatVector()),
		BiasesPar1(FloatVector()),
		BiasesPar2(FloatVector()),
		Moments(0),
		B1(Float(0.9)),
		B2(Float(0.999)),
		AdaDeltaEps(Float(1e-08)),
		AdaGradEps(Float(1e-08)),
		AdamEps(Float(1e-08)),
		AdamBeta2(Float(0.999)),
		AdamaxEps(Float(1e-08)),
		AdamaxBeta2(Float(0.999)),
		RMSPropEps(Float(1e-08)),
		RAdamEps(Float(1e-08)),
		RAdamBeta1(Float(0.9)),
		RAdamBeta2(Float(0.999)),
		Outputs(std::vector<Layer*>()),
		UseDefaultParameters(true),
		LockUpdate(false),
		RefreshingStats(false),
		LayerBeforeCost(false),
		SharesInput(false),
		NeuronsMin(Float(0)),
	    NeuronsMax(Float(0)),
	    NeuronsStdDev(Float(0)),
	    NeuronsMean(Float(0)),
	    WeightsMin(Float(0)),
	    WeightsMax(Float(0)),
	    WeightsStdDev(Float(0)),
	    WeightsMean(Float(0)),
	    BiasesMin(Float(0)),
	    BiasesMax(Float(0)),
	    BiasesStdDev(Float(0)),
	    BiasesMean(Float(0)),
		fpropTime(std::chrono::duration<Float>(Float(0))),
		bpropTime(std::chrono::duration<Float>(Float(0))),
		updateTime(std::chrono::duration<Float>(Float(0)))
	{
	}

	void Layer::SetParameters(const bool useDefaults, const Fillers weightsFiller, const Float weightsScale, const Float weightsLRM, const Float weightsWDM, const Fillers biasesFiller, const Float biasesScale, const Float biasesLRM, const Float biasesWDM)
	{
		UseDefaultParameters = useDefaults;
		WeightsFiller = weightsFiller;
		WeightsScale = weightsScale;
		WeightsLRM = weightsLRM;
		WeightsWDM = weightsWDM;
		BiasesFiller = biasesFiller;
		BiasesScale = biasesScale;
		BiasesLRM = biasesLRM;
		BiasesWDM = biasesWDM;
	}

	bool Layer::RefreshStatistics(const size_t batchSize)
	{
		if (!RefreshingStats.load())
		{
			RefreshingStats.store(true);

			if (!Neurons.empty())
			{
				const auto ncdhw = batchSize * CDHW;

				NeuronsMin = std::numeric_limits<Float>::max();
				NeuronsMax = std::numeric_limits<Float>::lowest();
				
				auto sum = Float(0);

				if (ncdhw % VectorSize == 0ull)
				{ 
					VecFloat neurons;
					for (auto i = 0ull; i < ncdhw; i += VectorSize)
					{
						neurons.load_a(&Neurons[i]);
						NeuronsMin = std::min(NeuronsMin, horizontal_min(neurons));
						NeuronsMax = std::max(NeuronsMax, horizontal_max(neurons));

						if ((NeuronsMin < Float(-1000)) || (NeuronsMax > Float(1000)))
							goto FAIL;
						
						sum += horizontal_add(neurons);
					}

					if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
					{
						NeuronsMean = sum / ncdhw;
						VecFloat vecSum = VecFloat(0);
						for (auto i = 0ull; i < ncdhw; i += VectorSize)
						{
							neurons.load_a(&Neurons[i]);
							vecSum += square(neurons - NeuronsMean);
						}
						sum = horizontal_add(vecSum);
						if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
							NeuronsStdDev = std::sqrt(sum / ncdhw);
						else
							NeuronsStdDev = Float(0);
					}
					else
						goto FAIL;
				}
				else
				{
					for (auto i = 0ull; i < ncdhw; i++)
					{
						NeuronsMin = std::min(NeuronsMin, Neurons[i]);
						NeuronsMax = std::max(NeuronsMax, Neurons[i]);

						if ((NeuronsMin < Float(-1000)) || (NeuronsMax > Float(1000)))
							goto FAIL;

						sum += Neurons[i];
					}

					if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
					{
						NeuronsMean = sum / ncdhw;
						sum = Float(0);
						for (auto i = 0ull; i < ncdhw; i++)
							sum += FloatSquare(Neurons[i] - NeuronsMean);

						if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
							NeuronsStdDev = std::sqrt(sum / ncdhw);
						else
							NeuronsStdDev = Float(0);
					}
					else
						goto FAIL;
				}
			}

			if (HasWeights)
			{
				WeightsMin = std::numeric_limits<Float>::max();
				WeightsMax = std::numeric_limits<Float>::lowest();
				auto sum = Float(0);
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					WeightsMin = std::min(WeightsMin, Weights[i]);
					WeightsMax = std::max(WeightsMax, Weights[i]);
					sum += Weights[i];
				}

				if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
				{
					WeightsMean = sum / Weights.size();
					sum = Float(0);
					for (auto i = 0ull; i < Weights.size(); i++)
						sum += FloatSquare(Weights[i] - WeightsMean);

					if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
						WeightsStdDev = std::sqrt(sum / Weights.size());
					else
						WeightsStdDev = Float(0);
				}
				else
					goto FAIL;

				if (HasBias)
				{
					BiasesMin = std::numeric_limits<Float>::max();
					BiasesMax = std::numeric_limits<Float>::lowest();
					sum = Float(0);
					for (auto i = 0ull; i < BiasCount; i++)
					{
						BiasesMin = std::min(BiasesMin, Biases[i]);
						BiasesMax = std::max(BiasesMax, Biases[i]);
						sum += Biases[i];
					}

					if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
					{
						BiasesMean = sum / BiasCount;
						sum = Float(0);
						for (auto i = 0ull; i < BiasCount; i++)
							sum += FloatSquare(Biases[i] - BiasesMean);

						if (!std::isnan<Float>(sum) && !std::isinf<Float>(sum))
							BiasesStdDev = std::sqrt(sum / BiasCount);
						else
							BiasesStdDev = Float(0);
					}
					else
						goto FAIL;
				}
			}

			RefreshingStats.store(false);

			return true;

		FAIL:
			NeuronsMin = Float(0);
			NeuronsMax = Float(0);
			NeuronsMean = Float(0);
			NeuronsStdDev = Float(0);

			BiasesMin = Float(0);
			BiasesMax = Float(0);
			BiasesMean = Float(0);
			BiasesStdDev = Float(0);

			RefreshingStats.store(false);

			return false;
		}
		else
			return true;
	}

	void Layer::CheckOptimizer(const Optimizers optimizer)
	{
		auto dirty = false;

		switch (optimizer)
		{
		case Optimizers::AdaDelta:
		{
			if (HasWeights)
			{
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					if (std::isnan<Float>(WeightsPar1[i]) || std::isinf<Float>(WeightsPar1[i]))
					{
						dirty = true;
						break;
					}
					if (std::isnan<Float>(WeightsPar2[i]) || std::isinf<Float>(WeightsPar2[i]))
					{
						dirty = true;
						break;
					}
				}
			}

			if (HasBias && !dirty)
			{
				for (auto i = 0ull; i < BiasCount; i++)
				{
					if (std::isnan<Float>(BiasesPar1[i]) || std::isinf<Float>(BiasesPar1[i]))
					{
						dirty = true;
						break;
					}
					if (std::isnan<Float>(BiasesPar2[i]) || std::isinf<Float>(BiasesPar2[i]))
					{
						dirty = true;
						break;
					}
				}
			}
		}
		break;

		case Optimizers::Adam:
		case Optimizers::RAdam:
		{
			if (HasWeights)
			{
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					if (std::isnan<Float>(WeightsPar1[i]) || std::isinf<Float>(WeightsPar1[i]))
					{
						dirty = true;
						break;
					}
					if (std::isnan<Float>(WeightsPar2[i]) || std::isinf<Float>(WeightsPar2[i]))
					{
						dirty = true;
						break;
					}
				}
			}

			if (HasBias && !dirty)
			{
				for (auto i = 0ull; i < BiasCount; i++)
				{
					if (std::isnan<Float>(BiasesPar1[i]) || std::isinf<Float>(BiasesPar1[i]))
					{
						dirty = true;
						break;
					}
					if (std::isnan<Float>(BiasesPar2[i]) || std::isinf<Float>(BiasesPar2[i]))
					{
						dirty = true;
						break;
					}
				}
			}

			if (std::isnan<Float>(B1) || std::isinf<Float>(B1))
				dirty = true;
			if (std::isnan<Float>(B2) || std::isinf<Float>(B2))
				dirty = true;
		}
		break;

		case Optimizers::Adamax:
		{
			if (HasWeights)
			{
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					if (std::isnan<Float>(WeightsPar1[i]) || std::isinf<Float>(WeightsPar1[i]))
					{
						dirty = true;
						break;
					}
					if (std::isnan<Float>(WeightsPar2[i]) || std::isinf<Float>(WeightsPar2[i]))
					{
						dirty = true;
						break;
					}
				}
			}

			if (HasBias && !dirty)
			{
				for (auto i = 0ull; i < BiasCount; i++)
				{
					if (std::isnan<Float>(BiasesPar1[i]) || std::isinf<Float>(BiasesPar1[i]))
					{
						dirty = true;
						break;
					}
					if (std::isnan<Float>(BiasesPar2[i]) || std::isinf<Float>(BiasesPar2[i]))
					{
						dirty = true;
						break;
					}
				}
			}

			if (std::isnan<Float>(B1) || std::isinf<Float>(B1))
				dirty = true;
		}
		break;

		case Optimizers::AdaGrad:
		case Optimizers::NAG:
		case Optimizers::RMSProp:
		case Optimizers::SGDMomentum:
		{
			if (HasWeights)
			{
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					if (std::isnan<Float>(WeightsPar1[i]) || std::isinf<Float>(WeightsPar1[i]))
					{
						dirty = true;
						break;
					}
				}
			}

			if (HasBias && !dirty)
			{
				for (auto i = 0ull; i < BiasCount; i++)
				{
					if (std::isnan<Float>(BiasesPar1[i]) || std::isinf<Float>(BiasesPar1[i]))
					{
						dirty = true;
						break;
					}
				}
			}
		}
		break;
		}

		if (dirty)
			ResetOptimizer(optimizer);
	}

	void Layer::ResetOptimizer(const Optimizers optimizer)
	{
		WeightsD1.resize(WeightsMemDesc->get_size(), Float(0));

		switch (optimizer)
		{
		case Optimizers::AdaDelta:
		case Optimizers::Adam:
		case Optimizers::Adamax:
		case Optimizers::RAdam:
			WeightsPar1 = FloatVector(WeightsMemDesc->get_size(), Float(0));
			WeightsPar2 = FloatVector(WeightsMemDesc->get_size(), Float(0));
			if (HasBias)
			{
				BiasesPar1 = FloatVector(BiasCount, Float(0));
				BiasesPar2 = FloatVector(BiasCount, Float(0));
			}
			B1 = Float(0.9);
			B2 = Float(0.999);
			Moments = 0;
			break;

		case Optimizers::AdaGrad:
		case Optimizers::NAG:
		case Optimizers::RMSProp:
		case Optimizers::SGDMomentum:
			WeightsPar1 = FloatVector(WeightsMemDesc->get_size(), Float(0));
			WeightsPar2.~vector();
			if (HasBias)
			{
				BiasesPar1 = FloatVector(BiasCount, Float(0));
				BiasesPar2.~vector();
			}
			Moments = 0;
			break;

		case Optimizers::SGD:
			WeightsPar1.~vector();
			WeightsPar2.~vector();
			if (HasBias)
			{
				BiasesPar1.~vector();
				BiasesPar2.~vector();
			}
			Moments = 0;
			break;
		}
	}

	void Layer::SetOptimizer(const Optimizers optimizer)
	{
		if (HasWeights)
		{
			WeightsD1.resize(WeightsMemDesc->get_size(), Float(0));

			if (HasBias)
				BiasesD1.resize(BiasCount, Float(0));

			switch (optimizer)
			{
			case Optimizers::AdaDelta:
			case Optimizers::Adam:
			case Optimizers::Adamax:
			case Optimizers::RAdam:
				WeightsPar1.resize(WeightsMemDesc->get_size(), Float(0));
				WeightsPar2.resize(WeightsMemDesc->get_size(), Float(0));
				if (HasBias)
				{
					BiasesPar1.resize(BiasCount, Float(0));
					BiasesPar2.resize(BiasCount, Float(0));
				}
				break;

			case Optimizers::AdaGrad:
			case Optimizers::NAG:
			case Optimizers::RMSProp:
			case Optimizers::SGDMomentum:
				WeightsPar1.resize(WeightsMemDesc->get_size(), Float(0));
				WeightsPar2.~vector();
				if (HasBias)
				{
					BiasesPar1.resize(BiasCount, Float(0));
					BiasesPar2.~vector();
				}
				break;

			case Optimizers::SGD:
				WeightsPar1.~vector();
				WeightsPar2.~vector();
				if (HasBias)
				{
					BiasesPar1.~vector();
					BiasesPar2.~vector();
				}
				break;
			}
		}
	}

	void Layer::ResetWeights(const Fillers weightsFiller, const Float weightsScale, const Fillers biasesFiller, const Float biasesScale)
	{
		if (HasWeights)
		{
			if (UseDefaultParameters)
			{
				WeightsFiller = weightsFiller;
				WeightsScale = weightsScale;
			}

			auto weights = FloatVector(WeightCount);

			switch (WeightsFiller)
			{
			case Fillers::Constant:
			{
				std::fill_n(weights.begin(), WeightCount, WeightsScale);
			}
			break;

			case Fillers::HeNormal:
			{
				const auto stddev = std::sqrt(Float(2) / Float(FanIn()));
				auto distribution = std::normal_distribution<Float>(Float(0), stddev);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::HeUniform:
			{
				const auto limit = std::sqrt(Float(6) / Float(FanIn()));
				auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::LeCunNormal:
			{
				auto stddev = std::sqrt(Float(1) / Float(FanIn()));
				auto distribution = std::normal_distribution<Float>(Float(0), stddev);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

		
			case Fillers::LeCunUniform:
			{
				auto limit = std::sqrt(Float(3) / Float(FanIn()));
				auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::Normal:
			{
				auto distribution = std::normal_distribution<Float>(Float(0), WeightsScale);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::TruncatedNormal:
			{
				auto distribution = std::normal_distribution<Float>(Float(0), WeightsScale);
				auto max = 2 * std::abs(WeightsScale);
				std::generate_n(weights.begin(), WeightCount, [&]()
				{
					Float value;
					do { value = distribution(RandomEngine); } while ((value < -max) || (value > max));
					return value;
				});
			}
			break;

			case Fillers::Uniform:
			{
				auto distribution = std::uniform_real_distribution<Float>(-WeightsScale, WeightsScale);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::XavierNormal:
			{
				const auto stddev = std::sqrt(Float(2) / (Float(FanIn()) + Float(FanOut())));
				auto distribution = std::normal_distribution<Float>(Float(0), stddev);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::XavierUniform:
			{
				const auto limit = std::sqrt(Float(6) / (Float(FanIn()) + Float(FanOut())));
				auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
				std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
			}
			break;
			}
			
			
			if (*PersistWeightsMemDesc != *WeightsMemDesc)
			{
				Weights.resize(WeightsMemDesc->get_size());
				WeightsD1.resize(WeightsMemDesc->get_size());

				auto memWeights = dnnl::memory(*PersistWeightsMemDesc, Device.first, weights.data());
				auto weightsMem = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());

				dnnl::reorder(memWeights, weightsMem).execute(Device.second, { {DNNL_ARG_FROM, memWeights}, {DNNL_ARG_TO, weightsMem} });
				Device.second.wait();
			}
			else
			{
				Weights.resize(WeightCount);
				WeightsD1.resize(WeightCount);

				std::copy(weights.begin(), weights.end(), Weights.begin());
			}
		}

		if (HasBias)
		{
			if (UseDefaultParameters)
			{
				BiasesFiller = biasesFiller;
				BiasesScale = biasesScale;
			}

			switch (BiasesFiller)
			{
			case Fillers::Constant:
			{
				std::fill_n(Biases.begin(), BiasCount, BiasesScale);
			}
			break;

			case Fillers::HeNormal:
			{
				auto stddev = std::sqrt(Float(2) / Float(FanIn()));
				auto distribution = std::normal_distribution<Float>(Float(0), stddev);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::HeUniform:
			{
				auto limit = std::sqrt(Float(6) / Float(FanIn()));
				auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::LeCunNormal:
			{
				auto stddev = std::sqrt(Float(1) / Float(FanIn()));
				auto distribution = std::normal_distribution<Float>(Float(0), stddev);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::LeCunUniform:
			{
				auto limit = std::sqrt(Float(3) / Float(FanIn()));
				auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::Normal:
			{
				auto distribution = std::normal_distribution<Float>(Float(0), BiasesScale);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::TruncatedNormal:
			{
				auto distribution = std::normal_distribution<Float>(Float(0), BiasesScale);
				auto max = 2 * std::abs(BiasesScale);
				std::generate_n(Biases.begin(), BiasCount, [&]()
				{
					Float value;
					do { value = distribution(RandomEngine); } while ((value < -max) || (value > max));
					return value;
				});
			}
			break;

			case Fillers::Uniform:
			{
				auto distribution = std::uniform_real_distribution<Float>(-BiasesScale, BiasesScale);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::XavierNormal:
			{
				auto stddev = std::sqrt(Float(2) / (Float(FanIn()) + Float(FanOut())));
				auto distribution = std::normal_distribution<Float>(Float(0), stddev);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			case Fillers::XavierUniform:
			{
				auto limit = std::sqrt(Float(6) / (Float(FanIn()) + Float(FanOut())));
				auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
				std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
			}
			break;

			default:
				std::fill_n(Biases.begin(), BiasCount, Float(0));
				break;
			}

		}
	}

	void Layer::ResetGradients()
	{
		std::fill(WeightsD1.begin(), WeightsD1.end(), Float(0));
		std::fill_n(BiasesD1.begin(), BiasCount, Float(0));
	}

	void Layer::UpdateWeights(const TrainingRate& rate, const Optimizers optimizer, const bool disableLocking)
	{
		if (HasWeights && (disableLocking || (!disableLocking && !LockUpdate.load())))
		{
			switch (optimizer)
			{
			case Optimizers::AdaDelta:
				AdaDelta(rate);
				break;
			case Optimizers::AdaGrad:
				AdaGrad(rate);
				break;
			case Optimizers::Adam:
				Adam(rate);
				break;
			case Optimizers::Adamax:
				Adamax(rate);
				break;
			case Optimizers::NAG:
				NAG(rate);
				break;
			case Optimizers::RMSProp:
				RMSProp(rate);
				break;
			case Optimizers::SGD:
				SGD(rate);
				break;
			case Optimizers::SGDMomentum:
				SGDMomentum(rate);
				break;
			case Optimizers::RAdam:
				RAdam(rate);
				break;
			}
		}
	}

	void Layer::AdaDelta(const TrainingRate& rate)
	{
		const auto lr = -rate.MaximumRate * WeightsLRM;
		const auto momentum = rate.Momentum;
		const auto oneMinMomentum = Float(1) - momentum;
		const auto eps = AdaDeltaEps;
		const auto batchRecip = Float(1) / rate.BatchSize;
		
		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
		{
			WeightsPar1[i] = (momentum * WeightsPar1[i]) + (oneMinMomentum * FloatSquare(WeightsD1[i] * batchRecip));
			const auto update = lr * (std::sqrt(WeightsPar2[i] + eps) / std::sqrt(WeightsPar1[i] + eps)) * WeightsD1[i] * batchRecip;
			WeightsPar2[i] = (momentum * WeightsPar2[i]) + (oneMinMomentum * FloatSquare(update));
			Weights[i] += update;
		}

		if (HasBias)
		{
			const auto lr = -rate.MaximumRate * BiasesLRM;

			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
			{
				BiasesPar1[i] = (momentum * BiasesPar1[i]) + (oneMinMomentum * FloatSquare(BiasesD1[i] * batchRecip));
				const auto update = lr * (std::sqrt(BiasesPar2[i] + eps) / std::sqrt(BiasesPar1[i] + eps)) * BiasesD1[i] * batchRecip;
				BiasesPar2[i] = (momentum * BiasesPar2[i]) + (oneMinMomentum * FloatSquare(update));
				Biases[i] += update;
			}
		}
	}

	void Layer::AdaGrad(const TrainingRate& rate)
	{
		const auto lr = rate.MaximumRate * WeightsLRM;
		const auto eps = AdaGradEps;
		const auto batchRecip = Float(1) / rate.BatchSize;

		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
		{
			WeightsPar1[i] += FloatSquare(WeightsD1[i] * batchRecip);
			Weights[i] -= lr * WeightsD1[i] / (std::sqrt(WeightsPar1[i]) + eps);
		}

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM;

			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
			{
				BiasesPar1[i] += FloatSquare(BiasesD1[i] * batchRecip);
				Biases[i] -= lr * BiasesD1[i] / (std::sqrt(BiasesPar1[i]) + eps);
			}
		}
	}

	void Layer::Adam(const TrainingRate& rate)
	{
		const auto beta1 = rate.Momentum;
		const auto beta2 = AdamBeta2;
		const auto lr = rate.MaximumRate * WeightsLRM;
		const auto eps = AdamEps;
		const auto b1 = B1;
		const auto b2 = B2;
		const auto oneMinusBeta1 = (Float(1) - beta1) / rate.BatchSize;
		const auto oneMinusBeta2 = Float(1) - beta2;
		const auto oneMinusB1 = Float(1) - b1;
		const auto oneMinusB2 = Float(1) - b2;
		const auto batchRecip = Float(1) / rate.BatchSize;

		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
		{
			WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i]);
			WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
			Weights[i] -= lr * (WeightsPar1[i] / oneMinusB1) / std::sqrt((WeightsPar2[i] / oneMinusB2) + eps);
		}

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM;

			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
			{
				BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i]);
				BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
				Biases[i] -= lr * (BiasesPar1[i] / oneMinusB1) / std::sqrt((BiasesPar2[i] / oneMinusB2) + eps);
			}
		}

		B1 *= beta1;
		B2 *= beta2;
	}

	void Layer::Adamax(const TrainingRate& rate)
	{
		const auto lr = rate.MaximumRate * WeightsLRM / (Float(1) - B1);
		const auto batchRecip = Float(1) / rate.BatchSize;
		const auto beta1 = rate.Momentum;
		const auto oneMinusBeta1 = (Float(1) - beta1) / rate.BatchSize;
		const auto beta2 = AdamaxBeta2;
		const auto eps = AdamaxEps;

		VecFloat weights, weightsD1, par1, par2;
		for (auto i = 0ull; i < Weights.size(); i += VectorSize)
		{
			par1.load_a(&WeightsPar1[i]);
			par1 = (beta1 * par1) + (oneMinusBeta1 * weightsD1.load_a(&WeightsD1[i]));
			par1.store_a(&WeightsPar1[i]);
			par2.load_a(&WeightsPar2[i]);
			par2 = max(beta2 * par2, abs(weightsD1 * batchRecip));
			par2.store_a(&WeightsPar2[i]);
			weights.load_a(&Weights[i]);
			weights -= lr * par1 / (par2 + eps);
			weights.store_a(&Weights[i]);
		}

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM / (Float(1) - B1);

			for (auto i = 0ull; i < BiasCount; i++)
			{
				BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i]);
				BiasesPar2[i] = std::max(beta2 * BiasesPar2[i], std::abs(BiasesD1[i] * batchRecip));
				Biases[i] -= lr * BiasesPar1[i] / (BiasesPar2[i] + eps);
			}
		}

		B1 *= beta1;
	}

	void Layer::NAG(const TrainingRate& rate)
	{
		const auto lr = rate.MaximumRate * WeightsLRM;
		const auto l2Penalty = rate.L2Penalty * WeightsWDM * lr;
		const auto momentum = rate.Momentum;
		const auto momentumPlusOne = momentum + Float(1);
		const auto batchRecip = Float(1) / rate.BatchSize * lr;
		
		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
		{
			const auto V = momentum * WeightsPar1[i] - (WeightsD1[i] * batchRecip + Weights[i] * l2Penalty);
			Weights[i] += -momentum * WeightsPar1[i] + momentumPlusOne * V;
			WeightsPar1[i] = V;
		}

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM;
			const auto batchRecip = Float(1) / rate.BatchSize * lr;

			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
			{
				const auto V = momentum * BiasesPar1[i] - BiasesD1[i] * batchRecip;
				Biases[i] += -momentum * BiasesPar1[i] + momentumPlusOne * V;
				BiasesPar1[i] = V;
			}
		}
	}

	void Layer::RMSProp(const TrainingRate& rate)
	{
		const auto lr = rate.MaximumRate * WeightsLRM / rate.BatchSize;
		const auto eps = RMSPropEps;
		const auto momentum = rate.Momentum;
		const auto oneMinusMomentum = Float(1) - momentum;
		const auto batchRecip = Float(1) / rate.BatchSize;

		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
		{
			WeightsPar1[i] = (momentum * WeightsPar1[i]) + (oneMinusMomentum * FloatSquare(WeightsD1[i] * batchRecip));
			Weights[i] -= lr * WeightsD1[i] / std::sqrt(WeightsPar1[i] + eps);
		}

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;

			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
			{
				BiasesPar1[i] = (momentum * BiasesPar1[i]) + (oneMinusMomentum * FloatSquare(BiasesD1[i] * batchRecip));
				Biases[i] -= lr * BiasesD1[i] / std::sqrt(BiasesPar1[i] + eps);
			}
		}
	}

	void Layer::SGD(const TrainingRate& rate)
	{
		const auto lr = rate.MaximumRate * WeightsLRM / rate.BatchSize;
		const auto l2Penalty = rate.MaximumRate * WeightsLRM * rate.L2Penalty * WeightsWDM;

		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
			Weights[i] -= (lr * WeightsD1[i]) - (l2Penalty * Weights[i]);

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;;
			
			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
				Biases[i] -= lr * BiasesD1[i];
		}
	}

	void Layer::SGDMomentum(const TrainingRate& rate)
	{
		const auto momentum = rate.Momentum;
		const auto lr = rate.MaximumRate * WeightsLRM / rate.BatchSize;
		const auto l2Penalty = rate.MaximumRate * WeightsLRM * rate.L2Penalty * WeightsWDM;

		#pragma omp simd
		for (auto i = 0ull; i < Weights.size(); i++)
		{
			WeightsPar1[i] = (momentum * WeightsPar1[i]) - (lr * WeightsD1[i]) - (l2Penalty * Weights[i]);
			Weights[i] += WeightsPar1[i];
		}

		if (HasBias)
		{
			const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;
			
			#pragma omp simd
			for (auto i = 0ull; i < BiasCount; i++)
			{
				BiasesPar1[i] = momentum * BiasesPar1[i] - lr * BiasesD1[i];
				Biases[i] += BiasesPar1[i];
			}
		}
	}

	void Layer::RAdam(const TrainingRate& rate)
	{
		const auto batchRecip = Float(1) / rate.BatchSize;
		const auto beta1 = rate.Momentum;
		const auto beta2 = RAdamBeta2;
		const auto lr = rate.MaximumRate * WeightsLRM;
		const auto eps = RAdamEps;
		const auto b1 = B1;
		const auto b2 = B2;
		const auto oneMinusBeta1 = (Float(1) - beta1) / rate.BatchSize;
		const auto oneMinusBeta2 = Float(1) - beta2;
		const auto oneMinusB1Recip = Float(1) / (Float(1) - b1);
		const auto oneMinusB2Recip = Float(1) / (Float(1) - b2);
		//const auto wd = rate.L2Penalty != Float(0) ? -rate.L2Penalty : Float(1);
		const auto pInf = Float(2) / oneMinusBeta2 - Float(1);
		const auto pt = pInf - (b2 * oneMinusB2Recip) * (2 * Moments);
		
		if (pt >= Float(5))
		{
			const auto rt = std::sqrt(((pt - Float(4)) * (pt - Float(2)) * pInf) / ((pInf - Float(4)) * (pInf - Float(2)) * pt));
			
			#pragma omp simd
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i]);  // mt
				WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));   // vt
				Weights[i] -= lr / (std::sqrt(WeightsPar2[i] * oneMinusB2Recip) + eps) * rt * (WeightsPar1[i] * oneMinusB1Recip);
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM;

				#pragma omp simd
				for (auto i = 0ull; i < BiasCount; i++)
				{
					BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i]);
					BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
					Biases[i] -= lr / (std::sqrt(BiasesPar2[i] * oneMinusB2Recip) + eps) * rt * (BiasesPar1[i] * oneMinusB1Recip);
				}
			}
		}
		else
		{
			#pragma omp simd
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i]);  // mt
				WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));   // vt
				Weights[i] -= lr * (WeightsPar1[i] * oneMinusB1Recip);
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM;

				#pragma omp simd
				for (auto i = 0ull; i < BiasCount; i++)
				{
					BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i]);
					BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
					Biases[i] -= lr * (BiasesPar1[i] * oneMinusB1Recip);
				}
			}
		}
		
		Moments++;
		B1 *= beta1;
		B2 *= beta2;
	}
}
