#pragma once
#include "Dataprovider.h"

namespace dnn
{
	class Model;
	
	struct TrainingRate
	{
		size_t BatchSize;
		size_t Cycles;
		size_t Epochs;
		size_t EpochMultiplier;
		size_t DecayAfterEpochs;
		size_t Interpolation;
		size_t ColorAngle;
		Float ColorCast;
		Float Distortion;
		Float Dropout;
		Float Cutout;
		Float AutoAugment;
		Float MaximumRate;
		Float MinimumRate;
		Float L2Penalty;
		Float Momentum;
		Float DecayFactor;
		Float Scaling;
		Float Rotation;
		bool HorizontalFlip;
		bool VerticalFlip;

		TrainingRate() :
			BatchSize(1),
			Cycles(1),
			Epochs(200),
			EpochMultiplier(1),
			DecayAfterEpochs(1),
			Interpolation(size_t(Interpolation::Cubic)),
			ColorAngle(0),
			ColorCast(Float(0)),
			Distortion(Float(0)),
			Dropout(Float(0)),
			Cutout(Float(0)),
			AutoAugment(Float(0)),
			MaximumRate(Float(0.05)),
			MinimumRate(Float(0.0001)),
			L2Penalty(Float(0.0005)),
			Momentum(Float(0.9)),
			DecayFactor(Float(1)),
			Scaling(Float(10.0)),
			Rotation(Float(10.0)),
			HorizontalFlip(false),
			VerticalFlip(false)
		{
		}

		TrainingRate(const Float maximumRate, const size_t batchSize, const size_t cycles, const size_t epochs, const size_t epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const size_t decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const size_t colorAngle, const Float distortion, const size_t interpolation, const Float scaling, const Float rotation) :
			MaximumRate(maximumRate),
			BatchSize(batchSize),
			Cycles(cycles),
			Epochs(epochs),
			EpochMultiplier(epochMultiplier),
			MinimumRate(minimumRate),
			L2Penalty(L2penalty),
			Momentum(momentum),
			DecayFactor(decayFactor),
			DecayAfterEpochs(decayAfterEpochs),
			HorizontalFlip(horizontalFlip),
			VerticalFlip(verticalFlip),
			Dropout(dropout),
			Cutout(cutout),
			AutoAugment(autoAugment),
			ColorCast(colorCast),
			ColorAngle(colorAngle),
			Distortion(distortion),
			Interpolation(interpolation),
			Scaling(scaling),
			Rotation(rotation)
		{
		}
	};

	enum class LayerTypes
	{
		Activation = 0,
		Add = 1,
		Average = 2,
		AvgPooling = 3,
		BatchNorm = 4,
		BatchNormHardLogistic = 5,
		BatchNormHardSwish = 6,
		BatchNormHardSwishDropout = 7,
		BatchNormRelu = 8,
		BatchNormReluDropout = 9,
		BatchNormSwish = 10,
		ChannelMultiply = 11,
		ChannelShuffle = 12,
		ChannelSplit = 13,
		ChannelZeroPad = 14,
		Concat = 15,
		Convolution = 16,
		ConvolutionTranspose = 17,
		Cost = 18,
		Dense = 19,
		DepthwiseConvolution = 20,
		Divide = 21,
		Dropout = 22,
		GlobalAvgPooling = 23,
		GlobalMaxPooling = 24,
		Input = 25,
		LocalResponseNormalization = 26,
		Max = 27,
		MaxPooling = 38,
		Min = 29,
		Multiply = 30,
		PartialDepthwiseConvolution = 31,
		Resampling = 32,
		Substract = 33
	};

	enum class Optimizers
	{
		AdaDelta = 0,
		AdaGrad = 1,
		Adam = 2,
		Adamax = 3,
		NAG = 4,
		RMSProp = 5,
		SGD = 6,
		SGDMomentum = 7,
		RAdam = 8
	};

	enum class Fillers
	{
		Constant = 0,
		HeNormal = 1,
		HeUniform = 2,
		LeCunNormal = 3,
		LeCunUniform = 4,
		Normal = 5,
		TruncatedNormal = 6,
		Uniform = 7,
		XavierNormal = 8,
		XavierUniform = 9
	};
	
	struct Device
	{
		const dnnl::engine engine;
		dnnl::stream stream;

		Device(const dnnl::engine& eng, dnnl::stream str) : engine(eng), stream(str) {}
	};
	
	class Layer
	{
	protected:
		dnn::Device Device;
		dnnl::memory::format_tag chosenFormat;
		std::mt19937 RandomEngine;
		
	public:
		const std::string Name;
		const LayerTypes LayerType;
		const size_t WeightCount;
		const size_t BiasCount;
		const size_t C;
		const size_t D;
		const size_t H;
		const size_t W;
		const size_t HW;
		const size_t CDHW;
		const size_t PaddedC;
		const size_t PaddedCDHW;
		const size_t PadD;
		const size_t PadH;
		const size_t PadW;
		const bool HasPadding;
		const std::vector<Layer*> Inputs;
		Layer* InputLayer;
		std::vector<Layer*> Outputs;
		bool LayerBeforeCost;
		bool SharesInput;
		const dnnl::memory::format_tag Format;
		const bool HasBias;
		const bool HasWeights;
		bool UseDefaultParameters;
		Fillers WeightsFiller;
		Float WeightsScale;
		Float WeightsLRM;
		Float WeightsWDM;
		Fillers BiasesFiller;
		Float BiasesScale;
		Float BiasesLRM;
		Float BiasesWDM;
		FloatVector Neurons;
		FloatVector NeuronsD1;
		FloatVector Weights;
		FloatVector WeightsD1;
		FloatVector Biases;
		FloatVector BiasesD1;
		FloatVector WeightsPar1;
		FloatVector WeightsPar2;
		FloatVector BiasesPar1;
		FloatVector BiasesPar2;
		size_t Moments;
		Float B1;
		Float B2;
		Float AdaDeltaEps;
		Float AdaGradEps;
		Float AdamEps;
		Float AdamBeta2;
		Float AdamaxEps;
		Float AdamaxBeta2;
		Float RMSPropEps;
		Float RAdamEps;
		Float RAdamBeta1;
		Float RAdamBeta2;
		Float NeuronsMin;
		Float NeuronsMax;
		Float NeuronsStdDev;
		Float NeuronsMean;
		Float WeightsMin;
		Float WeightsMax;
		Float WeightsStdDev;
		Float WeightsMean;
		Float BiasesMin;
		Float BiasesMax;
		Float BiasesStdDev;
		Float BiasesMean;
		std::atomic<bool> LockUpdate;
		std::atomic<bool> RefreshingStats;
		std::chrono::duration<Float> fpropTime;
		std::chrono::duration<Float> bpropTime;
		std::chrono::duration<Float> updateTime;
		std::unique_ptr<dnnl::memory::desc> DstMemDesc;
		std::unique_ptr<dnnl::memory::desc> DiffDstMemDesc;
		std::unique_ptr<dnnl::memory::desc> WeightsMemDesc;
		std::unique_ptr<dnnl::memory::desc> PersistWeightsMemDesc;

		Layer(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const LayerTypes layerType, const size_t weightCount, const size_t biasCount, const size_t c, const size_t d, const size_t h, const size_t w, const size_t padD, const size_t padH, const size_t padW, const std::vector<Layer*>& inputs, const bool hasBias = false) :
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
			HW(h* w),
			CDHW(c* d* h* w),
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
			chosenFormat = format;
		}

		virtual ~Layer() = default;

		void SetParameters(const bool useDefaults, const Fillers weightsFiller, const Float weightsScale, const Float weightsLRM, const Float weightsWDM, const Fillers biasesFiller, const Float biasesScale, const Float biasesLRM, const Float biasesWDM)
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

		bool IsPlainFormat() const { return Format == dnnl::memory::format_tag::ab || Format == dnnl::memory::format_tag::abc || Format == dnnl::memory::format_tag::abcd || Format == dnnl::memory::format_tag::abcde; };

		std::string GetDescriptionHeader() const
		{
			auto description = std::string("");

			description.append(std::string(" Type:") + dtab + std::string(magic_enum::enum_name<LayerTypes>(LayerType)));

			if (LayerType != LayerTypes::Input)
			{
				description.append(nwl + std::string(" Inputs:") + tab);
				for (auto i = 0ull; i < Inputs.size(); i++)
					description.append((i == 0 ? std::string("") : std::string(",")) + Inputs[i]->Name);
			}

			description.append(nwl + std::string(" Features:") + tab + std::to_string(C) + std::string("x") + std::to_string(H) + std::string("x") + std::to_string(W));
			description.append(nwl + std::string(" Neurons:") + tab + std::to_string(CDHW));
			description.append(nwl + std::string(" Format:") + tab + std::string(magic_enum::enum_name<dnnl::memory::format_tag>(chosenFormat)));

			return description;
		}

		std::string GetWeightsDescription(const bool visible = true) const
		{
			auto description = std::string("");

			if (visible)
			{
				description.append(nwl + std::string(" Weights:") + tab + std::to_string(WeightCount));
				description.append(nwl + std::string("  lr mult:") + tab + FloatToString(WeightsLRM));
				description.append(nwl + std::string("  wd mult:") + tab + FloatToString(WeightsWDM));

				if (HasBias)
				{
					description.append(nwl + std::string(" Biases:") + tab + std::to_string(BiasCount));
					description.append(nwl + std::string("  lr mult:") + tab + FloatToString(BiasesLRM));
					description.append(nwl + std::string("  wd mult:") + tab + FloatToString(BiasesWDM));
				}
			}

			return description;
		}

		virtual std::string GetDescription() const = 0;

		virtual size_t FanIn() const = 0;

		virtual size_t FanOut() const = 0;

		virtual bool Lockable() const
		{
			return WeightCount > 0;
		}

		virtual void InitializeDescriptors(const size_t) = 0;

#ifdef DNN_LEAN
		inline void ZeroGradient(const size_t batchSize)
		{
			ZeroFloatVectorAllocate(InputLayer->NeuronsD1, batchSize * InputLayer->PaddedCDHW);
		}

		inline void ZeroGradientMulti(const size_t batchSize)
		{
			for (auto i = 0ull; i < Inputs.size(); i++)
				ZeroFloatVectorAllocate(Inputs[i]->NeuronsD1, batchSize * Inputs[i]->PaddedCDHW);
		}

		inline void ReleaseGradient()
		{
			NeuronsD1.resize(0);
		}
#endif // DNN_LEAN

		virtual void SetBatchSize(const size_t batchSize)
		{
			while (RefreshingStats.load())
				std::this_thread::sleep_for(std::chrono::milliseconds(250));

			ZeroFloatVectorAllocate(Neurons, batchSize * PaddedCDHW);
#ifndef DNN_LEAN
			ZeroFloatVectorAllocate(NeuronsD1, batchSize * PaddedCDHW);
#else
			ReleaseGradient();
#endif // DNN_LEAN

			InitializeDescriptors(batchSize);
		}

		virtual void ForwardProp(const size_t batchSize, const bool training) = 0;

		virtual void BackwardProp(const size_t batchSize) = 0;

		bool RefreshStatistics(const size_t batchSize)
		{
			if (!RefreshingStats.load())
			{
				RefreshingStats.store(true);

				if (!Neurons.empty())
				{
					const auto ncdhw = batchSize * CDHW;

					NeuronsMin = std::numeric_limits<Float>::max();
					NeuronsMax = std::numeric_limits<Float>::lowest();

					float sum = Float(0);

					if (ncdhw % VectorSize == 0ull)
					{
						VecFloat neurons;
						for (auto i = 0ull; i < ncdhw; i += VectorSize)
						{
							neurons.load_a(&Neurons[i]);
							NeuronsMin = std::min(NeuronsMin, horizontal_min(neurons));
							NeuronsMax = std::max(NeuronsMax, horizontal_max(neurons));

							if ((NeuronsMin < -NEURONS_LIMIT) || (NeuronsMax > NEURONS_LIMIT))
								goto FAIL;

							sum += horizontal_add(neurons);
						}

						if (!std::isnan(sum) && !std::isinf(sum))
						{
							NeuronsMean = sum / ncdhw;
							VecFloat vecSum = VecFloat(0);
							for (auto i = 0ull; i < ncdhw; i += VectorSize)
							{
								neurons.load_a(&Neurons[i]);
								vecSum += square(neurons - NeuronsMean);
							}
							sum = horizontal_add(vecSum);
							if (!std::isnan(sum) && !std::isinf(sum))
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

							if ((NeuronsMin < -NEURONS_LIMIT) || (NeuronsMax > NEURONS_LIMIT))
								goto FAIL;

							sum += Neurons[i];
						}

						if (!std::isnan(sum) && !std::isinf(sum))
						{
							NeuronsMean = sum / ncdhw;
							sum = Float(0);
							for (auto i = 0ull; i < ncdhw; i++)
								sum += FloatSquare(Neurons[i] - NeuronsMean);

							if (!std::isnan(sum) && !std::isinf(sum))
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
					float sum = Float(0);

					for (auto i = 0ull; i < Weights.size(); i++)
					{
						WeightsMin = std::min(WeightsMin, Weights[i]);
						WeightsMax = std::max(WeightsMax, Weights[i]);

						if ((WeightsMin < -WEIGHTS_LIMIT) || (WeightsMax > WEIGHTS_LIMIT))
							goto FAIL;

						sum += Weights[i];
					}

					if (!std::isnan(sum) && !std::isinf(sum))
					{
						WeightsMean = sum / Weights.size();
						sum = Float(0);
						for (auto i = 0ull; i < Weights.size(); i++)
							sum += FloatSquare(Weights[i] - WeightsMean);

						if (!std::isnan(sum) && !std::isinf(sum))
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

							if ((BiasesMin < -WEIGHTS_LIMIT) || (BiasesMax > WEIGHTS_LIMIT))
								goto FAIL;

							sum += Biases[i];
						}

						if (!std::isnan(sum) && !std::isinf(sum))
						{
							BiasesMean = sum / BiasCount;
							sum = Float(0);
							for (auto i = 0ull; i < BiasCount; i++)
								sum += FloatSquare(Biases[i] - BiasesMean);

							if (!std::isnan(sum) && !std::isinf(sum))
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

		void CheckOptimizer(const Optimizers optimizer)
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
						if (std::isnan(WeightsPar1[i]) || std::isinf(WeightsPar1[i]))
						{
							dirty = true;
							break;
						}
						if (std::isnan(WeightsPar2[i]) || std::isinf(WeightsPar2[i]))
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
						if (std::isnan(BiasesPar1[i]) || std::isinf(BiasesPar1[i]))
						{
							dirty = true;
							break;
						}
						if (std::isnan(BiasesPar2[i]) || std::isinf(BiasesPar2[i]))
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
						if (std::isnan(WeightsPar1[i]) || std::isinf(WeightsPar1[i]))
						{
							dirty = true;
							break;
						}
						if (std::isnan(WeightsPar2[i]) || std::isinf(WeightsPar2[i]))
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
						if (std::isnan(BiasesPar1[i]) || std::isinf(BiasesPar1[i]))
						{
							dirty = true;
							break;
						}
						if (std::isnan(BiasesPar2[i]) || std::isinf(BiasesPar2[i]))
						{
							dirty = true;
							break;
						}
					}
				}

				if (std::isnan(B1) || std::isinf(B1))
					dirty = true;
				if (std::isnan(B2) || std::isinf(B2))
					dirty = true;
			}
			break;

			case Optimizers::Adamax:
			{
				if (HasWeights)
				{
					for (auto i = 0ull; i < Weights.size(); i++)
					{
						if (std::isnan(WeightsPar1[i]) || std::isinf(WeightsPar1[i]))
						{
							dirty = true;
							break;
						}
						if (std::isnan(WeightsPar2[i]) || std::isinf(WeightsPar2[i]))
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
						if (std::isnan(BiasesPar1[i]) || std::isinf(BiasesPar1[i]))
						{
							dirty = true;
							break;
						}
						if (std::isnan(BiasesPar2[i]) || std::isinf(BiasesPar2[i]))
						{
							dirty = true;
							break;
						}
					}
				}

				if (std::isnan(B1) || std::isinf(B1))
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
						if (std::isnan(WeightsPar1[i]) || std::isinf(WeightsPar1[i]))
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
						if (std::isnan(BiasesPar1[i]) || std::isinf(BiasesPar1[i]))
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

		void ResetOptimizer(const Optimizers optimizer)
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
				WeightsPar2.resize(0);
				if (HasBias)
				{
					BiasesPar1 = FloatVector(BiasCount, Float(0));
					BiasesPar2.resize(0);
				}
				Moments = 0;
				break;

			case Optimizers::SGD:
				WeightsPar1.resize(0);
				WeightsPar2.resize(0);
				if (HasBias)
				{
					BiasesPar1.resize(0);
					BiasesPar2.resize(0);
				}
				Moments = 0;
				break;
			}
		}

		void SetOptimizer(const Optimizers optimizer)
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
					WeightsPar2.resize(0);
					if (HasBias)
					{
						BiasesPar1.resize(BiasCount, Float(0));
						BiasesPar2.resize(0);
					}
					break;

				case Optimizers::SGD:
					WeightsPar1.resize(0);
					WeightsPar2.resize(0);
					if (HasBias)
					{
						BiasesPar1.resize(0);
						BiasesPar2.resize(0);
					}
					break;
				}
			}
		}
		
		virtual void ResetWeights(const Fillers weightsFiller, const Float weightsScale, const Fillers biasesFiller, const Float biasesScale)
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
						Float value = Float(0);
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

					auto memWeights = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weights.data());
					auto weightsMem = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());

					dnnl::reorder(memWeights, weightsMem).execute(Device.stream, { {DNNL_ARG_FROM, memWeights}, {DNNL_ARG_TO, weightsMem} });
					Device.stream.wait();
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
						Float value = Float(0);
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

		void ResetGradients()
		{
			std::fill(WeightsD1.begin(), WeightsD1.end(), Float(0));
			std::fill_n(BiasesD1.begin(), BiasCount, Float(0));
		}

		void UpdateWeights(const TrainingRate& rate, const Optimizers optimizer, const bool disableLocking)
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

		inline void AdaDelta(const TrainingRate& rate)
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

		inline void AdaGrad(const TrainingRate& rate)
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

		inline void Adam(const TrainingRate& rate)
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

		inline void Adamax(const TrainingRate& rate)
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

		inline void NAG(const TrainingRate& rate)
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

		inline void RMSProp(const TrainingRate& rate)
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

		inline void SGD(const TrainingRate& rate)
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

		inline void SGDMomentum(const TrainingRate& rate)
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

		inline void RAdam(const TrainingRate& rate)
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

		/*size_t OffsetPaddedMem(const size_t n, const size_t c, const size_t h, const size_t w) const
		{
			return n * PaddedCDHW + (c / VectorSize) * HW * VectorSize + h * W * VectorSize + w * VectorSize + (c % VectorSize);
		}*/
		
		virtual void Save(std::ostream& os, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD)
		{
			if (HasWeights)
			{
				os.write(reinterpret_cast<const char*>(&LockUpdate), sizeof(std::atomic<bool>));
				
				if (*WeightsMemDesc != *PersistWeightsMemDesc)
				{
					auto weights = FloatVector(WeightCount);
					auto memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
					auto weightsMem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weights.data());
					dnnl::reorder(memWeights, weightsMem).execute(Device.stream, { {DNNL_ARG_FROM, memWeights}, {DNNL_ARG_TO, weightsMem} });
					Device.stream.wait();
					
					os.write(reinterpret_cast<const char*>(weights.data()), std::streamsize(WeightCount * sizeof(Float)));
					if (HasBias)
						os.write(reinterpret_cast<const char*>(Biases.data()), std::streamsize(BiasCount * sizeof(Float)));
					
					if (persistOptimizer)
						switch (optimizer)
						{
						case Optimizers::AdaDelta:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::Adam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
						break;
						}
				}
				else
				{
					os.write(reinterpret_cast<const char*>(Weights.data()), std::streamsize(WeightCount * sizeof(Float)));
					if (HasBias)
						os.write(reinterpret_cast<const char*>(Biases.data()), std::streamsize(BiasCount * sizeof(Float)));

					if (persistOptimizer)
						switch (optimizer)
						{
						case Optimizers::Adam:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
						break;
						}
				}
			}
		}

		virtual void Load(std::istream& is, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD)
		{
			if (HasWeights)
			{
				is.read(reinterpret_cast<char*>(&LockUpdate), sizeof(std::atomic<bool>));
				
				if (*WeightsMemDesc != *PersistWeightsMemDesc)
				{
					auto weights = FloatVector(WeightCount);
					is.read(reinterpret_cast<char*>(weights.data()), std::streamsize(WeightCount * sizeof(Float)));
					auto memWeights = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weights.data());
					auto weightsMem = dnnl::memory(*WeightsMemDesc, Device.engine, Weights.data());
					dnnl::reorder(memWeights, weightsMem).execute(Device.stream, { {DNNL_ARG_FROM, memWeights}, {DNNL_ARG_TO, weightsMem} });
					Device.stream.wait();
					if (HasBias)
						is.read(reinterpret_cast<char*>(Biases.data()), std::streamsize(BiasCount * sizeof(Float)));
					
					if (persistOptimizer)
						switch (optimizer)
						{
						case Optimizers::AdaDelta:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::Adam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.engine, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.engine, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.stream, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.stream.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
						break;
						}
				}
				else
				{
					is.read(reinterpret_cast<char*>(Weights.data()), std::streamsize(WeightCount * sizeof(Float)));
					if (HasBias)
						is.read(reinterpret_cast<char*>(Biases.data()), std::streamsize(BiasCount * sizeof(Float)));

					if (persistOptimizer)
						switch (optimizer)
						{
						case Optimizers::Adam:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
						break;
						}
				}
			}
		}

		virtual std::streamsize GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const
		{
			std::streamsize weightsSize = 0;

			if (HasWeights)
			{
				weightsSize += sizeof(std::atomic<bool>);

				if (persistOptimizer)
				{
					switch (optimizer)
					{
					case Optimizers::AdaDelta:
					{
						weightsSize += 3 * WeightCount * sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::Adam:
					{
						weightsSize += 3 * WeightCount * sizeof(Float) + 2 * sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::Adamax:
					{
						weightsSize += 3 * WeightCount * sizeof(Float) + sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::AdaGrad:
					case Optimizers::NAG:
					case Optimizers::RMSProp:
					case Optimizers::SGDMomentum:
					{
						weightsSize += 2 * WeightCount * sizeof(Float);
						if (HasBias)
							weightsSize += 2 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::SGD:
					{
						weightsSize += WeightCount * sizeof(Float);
						if (HasBias)
							weightsSize += BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::RAdam:
					{
						weightsSize += 3 * WeightCount * sizeof(Float) + 2 * sizeof(Float) + sizeof(size_t);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					}
				}
				else
				{
					weightsSize += std::streamsize(WeightCount * sizeof(Float));
					if (HasBias)
						weightsSize += std::streamsize(BiasCount * sizeof(Float));
				}
			}

			return weightsSize;
		}
					
		virtual size_t GetNeuronsSize(const size_t batchSize) const
		{
			size_t neuronsSize = 0;

#ifndef DNN_LEAN
			neuronsSize += PaddedCDHW * batchSize * sizeof(Float) * 2;
#else
			neuronsSize += PaddedCDHW * batchSize * sizeof(Float);
#endif // DNN_LEAN

			return neuronsSize;
		}

		virtual ByteVector GetImage(const Byte fillColor) { return ByteVector(); }
	};
}
