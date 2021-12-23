#pragma once
#include "Dataprovider.h"

namespace dnn
{
	class Model;
	
	enum class Optimizers
	{
		AdaBound = 0,
		AdaBoundW = 1,
		AdaDelta = 2,
		AdaGrad = 3,
		Adam = 4,
		Adamax = 5,
		AdamW = 6,
		AmsBound = 7,
		AmsBoundW = 8,
		NAG = 9,
		RMSProp = 10,
		SGD = 11,
		SGDMomentum = 12,
		SGDW = 13
	};

	struct TrainingRate
	{
		Optimizers Optimizer;
		Float Momentum;
		Float Beta2;
		Float L2Penalty;
		Float Dropout;
		Float Eps;
		UInt BatchSize;
		UInt Height;
		UInt Width;
		UInt Cycles;
		UInt Epochs;
		UInt EpochMultiplier;
		Float MaximumRate;
		Float MinimumRate;
		Float FinalRate;
		Float Gamma;
		UInt DecayAfterEpochs;
		Float DecayFactor;
		bool HorizontalFlip;
		bool VerticalFlip;
		Float InputDropout;
		Float Cutout;
		bool CutMix;
		Float AutoAugment;
		Float ColorCast;
		UInt ColorAngle;
		Float Distortion;
		Interpolations Interpolation;
		Float Scaling;
		Float Rotation;
	
		TrainingRate() :
			Optimizer(Optimizers::NAG),
			Momentum(Float(0.9)),
			Beta2(Float(0.999)),
			L2Penalty(Float(0.0005)),
			Dropout(Float(0)),
			Eps(Float(1E-08)),
			BatchSize(1),
			Height(32),
			Width(32),
			Cycles(1),
			Epochs(200),
			EpochMultiplier(1),
			MaximumRate(Float(0.05)),
			MinimumRate(Float(0.0001)),
			FinalRate(Float(0.1)),
			Gamma(Float(0.003)),
			DecayAfterEpochs(1),
			DecayFactor(Float(1)),
			HorizontalFlip(false),
			VerticalFlip(false),
			InputDropout(Float(0)),
			Cutout(Float(0)),
			CutMix(false),
			AutoAugment(Float(0)),
			ColorCast(Float(0)),
			ColorAngle(0),
			Distortion(Float(0)),
			Interpolation(Interpolations::Linear),
			Scaling(Float(10.0)),
			Rotation(Float(12.0))			
		{
		}

		TrainingRate(const Optimizers optimizer, const Float momentum, const Float beta2, const Float l2Penalty, const Float dropout, const Float eps, const UInt batchSize, const UInt height, const UInt width, const UInt cycles, const UInt epochs, const UInt epochMultiplier, const Float maximumRate, const Float minimumRate, const Float finalRate, const Float gamma, const UInt decayAfterEpochs, const Float decayFactor, const bool horizontalFlip, const bool verticalFlip, const Float inputDropout, const Float cutout, const bool cutMix, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const Interpolations interpolation, const Float scaling, const Float rotation) :
			Optimizer(optimizer),
			Momentum(momentum),
			Beta2(beta2),
			L2Penalty(l2Penalty),
			Dropout(dropout),
			Eps(eps),
			BatchSize(batchSize),
			Cycles(cycles),
			Epochs(epochs),
			Height(height),
			Width(width),
			EpochMultiplier(epochMultiplier),
			MaximumRate(maximumRate),
			MinimumRate(minimumRate),
			FinalRate(finalRate),
			Gamma(gamma),
			DecayAfterEpochs(decayAfterEpochs),
			DecayFactor(decayFactor),
			HorizontalFlip(horizontalFlip),
			VerticalFlip(verticalFlip),
			InputDropout(inputDropout),
			Cutout(cutout),
			CutMix(cutMix),
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
		BatchNormMish = 8,
		BatchNormMishDropout = 9,
		BatchNormRelu = 10,
		BatchNormReluDropout = 11,
		BatchNormSwish = 12,
		BatchNormSwishDropout = 13,
		BatchNormTanhExp = 14,
		BatchNormTanhExpDropout = 15,
		ChannelMultiply = 16,
		ChannelShuffle = 17,
		ChannelSplit = 18,
		ChannelZeroPad = 19,
		Concat = 20,
		Convolution = 21,
		ConvolutionTranspose = 22,
		Cost = 23,
		Dense = 24,
		DepthwiseConvolution = 25,
		Divide = 26,
		Dropout = 27,
		GlobalAvgPooling = 28,
		GlobalMaxPooling = 29,
		Input = 30,
		LayerNorm = 31,
		LocalResponseNorm = 32,
		LogSoftmax = 33,
		Max = 34,
		MaxPooling = 35,
		Min = 36,
		Multiply = 37,
		PartialDepthwiseConvolution = 38,
		PRelu = 39,
		Resampling = 40,
		Softmax = 41,
		Substract = 42
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
	
	enum class FillerModes
	{
		Avg = 0,
		In = 1,
		Out = 2
	};

	struct Device
	{
		const dnnl::engine engine;
		dnnl::stream stream;
		
		Device(const dnnl::engine& eng, dnnl::stream str) : engine(eng), stream(str) 
		{ 
		}
	};
	
	struct Stats
	{
		Float Mean;
		Float StdDev;
		Float Min;
		Float Max;

		Stats() : Mean(0), StdDev(0), Min(0), Max(0)
		{
		}

		Stats(const Stats& stats) :
		   Mean(stats.Mean),
		   StdDev(stats.StdDev),
		   Min(stats.Min),
		   Max(stats.Max)
		{
		}

		Stats(const Float mean, const Float stddev, const Float min, const Float max) :
			Mean(mean),
			StdDev(stddev),
			Min(min),
			Max(max)
		{
		}
	};

	class Layer
	{
	protected:
		dnn::Device Device;
		dnnl::memory::format_tag ChosenFormat;
		std::mt19937 RandomEngine;

		auto IsInplaceBwd(const LayerTypes layerType, const std::vector<Layer*>& inputs) const
		{
			if (UseInplace && (layerType == LayerTypes::Activation || layerType == LayerTypes::LayerNorm || std::string(magic_enum::enum_name<LayerTypes>(layerType)).find("BatchNorm", 0) != std::string::npos) && (inputs.size() == 1) && (inputs[0]->LayerType == LayerTypes::Convolution || inputs[0]->LayerType == LayerTypes::DepthwiseConvolution || inputs[0]->LayerType == LayerTypes::ConvolutionTranspose))
				return true;
			else
				return false;
		}

		auto GetInplaceInputsBwd(const LayerTypes layerType, const std::vector<Layer*>& inputs) const
		{
			if (IsInplaceBwd(layerType, inputs))
				return std::vector<Layer*>(inputs);
			else
			{
				auto inputsInplace = std::vector<Layer*>();

				if (inputs.size() > 0)
				{
					for (auto input : inputs)
						inputsInplace.push_back(input->InplaceBwd ? input->InputLayerOriginal : input);
				}

				return inputsInplace;
			}
		}

	public:
		const std::string Name;
		const LayerTypes LayerType;
		const UInt WeightCount;
		const UInt BiasCount;
		const UInt C;
		UInt D;
		UInt H;
		UInt W;
		UInt HW;
		UInt CDHW;
		const UInt PaddedC;
		UInt PaddedCDHW;
		const UInt PadD;
		const UInt PadH;
		const UInt PadW;
		const bool HasPadding;
		std::vector<Layer*> Inputs;
		//std::vector<Layer*> Outputs;
		const std::vector<Layer*> InputsOriginal;
		const std::vector<Layer*> InputsInplaceBwd;
		Layer* InputLayer;
		Layer* InputLayerInplaceBwd;
		Layer* InputLayerOriginal;
		bool LayerBeforeCost;
		bool SharesInput;
		bool SharesInputOriginal;
		bool SharesInputInplace;
		dnnl::memory::format_tag Format;
		const bool Scaling;
		const bool HasBias;
		const bool HasWeights;
		const bool InplaceBwd;
		bool Enabled;
		bool UseDefaultParameters;
		Fillers WeightsFiller;
		FillerModes WeightsFillerMode;
		Float WeightsGain;
		Float WeightsScale;
		Float WeightsLRM;
		Float WeightsWDM;
		Fillers BiasesFiller;
		FillerModes BiasesFillerMode;
		Float BiasesGain;
		Float BiasesScale;
		Float BiasesLRM;
		Float BiasesWDM;
		FloatArray Neurons;
		FloatArray NeuronsD1;
		FloatVector Weights;
		FloatVector WeightsD1;
		FloatVector WeightsPar1;
		FloatVector WeightsPar2;
		FloatVector Biases;
		FloatVector BiasesD1;
		FloatVector BiasesPar1;
		FloatVector BiasesPar2;
		Float B1;
		Float B2;
		Float Gamma;
		Stats NeuronsStats;
		Stats WeightsStats;
		Stats BiasesStats;
		std::atomic<bool> LockUpdate;
		std::atomic<bool> RefreshingStats;
		std::chrono::duration<Float> fpropTime;
		std::chrono::duration<Float> bpropTime;
		std::chrono::duration<Float> updateTime;
		std::unique_ptr<dnnl::memory::desc> DstMemDesc;
		std::unique_ptr<dnnl::memory::desc> DiffDstMemDesc;
		std::unique_ptr<dnnl::memory::desc> WeightsMemDesc;
		std::unique_ptr<dnnl::memory::desc> PersistWeightsMemDesc;

		Layer(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const LayerTypes layerType, const UInt weightCount, const UInt biasCount, const UInt c, const UInt d, const UInt h, const UInt w, const UInt padD, const UInt padH, const UInt padW, const std::vector<Layer*>& inputs, const bool hasBias = false, const bool scaling = false, const bool enabled = true) :
			Device(device),
			Format(format),
			ChosenFormat(format),
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
			Inputs(std::vector<Layer*>(inputs)),						// Inputs is switched between non-inplace (forward) and inplace (backprop) during training 
			InputsOriginal(std::vector<Layer*>(inputs)),				// InputsOriginal = the non-inplace inputs 
			InputsInplaceBwd(GetInplaceInputsBwd(layerType, inputs)),	// InputsInplaceBwd = the inplace inputs backward prop
			InputLayer(inputs.size() > 0 ? inputs[0] : nullptr),
			InputLayerOriginal(inputs.size() > 0 ? inputs[0] : nullptr),
			InputLayerInplaceBwd(GetInplaceInputsBwd(layerType, inputs).size() > 0 ? GetInplaceInputsBwd(layerType, inputs)[0] : nullptr),
			InplaceBwd(IsInplaceBwd(layerType, inputs)),
			Enabled(enabled),
			Scaling(scaling),
			HasBias(hasBias && biasCount > 0),
			HasWeights(weightCount > 0),
			WeightsFiller(Fillers::HeNormal),
			WeightsFillerMode(FillerModes::In),
			WeightsGain(Float(1)),
			WeightsScale(Float(0.05)),
			WeightsLRM(Float(1)),
			WeightsWDM(Float(1)),
			BiasesFiller(Fillers::Constant),
			BiasesFillerMode(FillerModes::In),
			BiasesGain(Float(1)),
			BiasesScale(Float(0)),
			BiasesLRM(Float(1)),
			BiasesWDM(Float(1)),
			HW(h * w),
			CDHW(c * d * h * w),
			PaddedC(DivUp(c)),
			PaddedCDHW((layerType == LayerTypes::Input ? c : DivUp(c)) * d * h * w),
			HasPadding(padD > 0 || padH > 0 || padW > 0),
			RandomEngine(std::mt19937(Seed<unsigned>())),
			Neurons(FloatArray()),
			NeuronsD1(FloatArray()),
			Weights(FloatVector(weightCount)),
			WeightsD1(FloatVector(weightCount)),
			Biases(FloatVector(biasCount)),
			BiasesD1(FloatVector(biasCount)),
			WeightsPar1(FloatVector()),
			WeightsPar2(FloatVector()),
			BiasesPar1(FloatVector()),
			BiasesPar2(FloatVector()),
			B1(Float(0)),
			B2(Float(0)),
			Gamma(Float(0)),
			UseDefaultParameters(true),
			LockUpdate(false),
			RefreshingStats(false),
			LayerBeforeCost(false),
			SharesInput(false),
			SharesInputOriginal(false),
			SharesInputInplace(false),
			NeuronsStats(Stats()),
			WeightsStats(Stats()),
			BiasesStats(Stats()),
			fpropTime(std::chrono::duration<Float>(Float(0))),
			bpropTime(std::chrono::duration<Float>(Float(0))),
			updateTime(std::chrono::duration<Float>(Float(0)))
		{
		}

		virtual ~Layer() = default;
		
		virtual void UpdateResolution()
		{
			HW = H * W;
			CDHW = C * D * HW;
			PaddedCDHW = LayerType == LayerTypes::Input ? CDHW : PaddedC * HW;
		}

		void SetParameters(const bool useDefaults, const Fillers weightsFiller, const FillerModes weightsFillerMode, const Float weightsGain, const Float weightsScale, const Float weightsLRM, const Float weightsWDM, const Fillers biasesFiller, const FillerModes biasesFillerMode, const Float biasesGain, const Float biasesScale, const Float biasesLRM, const Float biasesWDM)
		{
			UseDefaultParameters = useDefaults;
			WeightsFiller = weightsFiller;
			WeightsFillerMode = weightsFillerMode;
			WeightsGain = weightsGain;
			WeightsScale = weightsScale;
			WeightsLRM = weightsLRM;
			WeightsWDM = weightsWDM;
			BiasesFiller = biasesFiller;
			BiasesFillerMode = biasesFillerMode;
			BiasesGain = biasesGain;
			BiasesScale = biasesScale;
			BiasesLRM = biasesLRM;
			BiasesWDM = biasesWDM;
		}

		bool IsPlainFormat() const 
		{ 
			return ChosenFormat == dnnl::memory::format_tag::ab || ChosenFormat == dnnl::memory::format_tag::abc || ChosenFormat == dnnl::memory::format_tag::abcd || ChosenFormat == dnnl::memory::format_tag::abcde; 
		}

		const bool IsBatchNorm() const 
		{ 
			return std::string(magic_enum::enum_name<LayerTypes>(LayerType)).find("BatchNorm", 0) != std::string::npos;
		}

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
			description.append(nwl + std::string(" Format:") + tab + std::string(dnnl_fmt_tag2str(static_cast<dnnl_format_tag_t>(ChosenFormat))));

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

		virtual UInt FanIn() const = 0;

		virtual UInt FanOut() const = 0;

		virtual bool Lockable() const
		{
			return WeightCount > 0;
		}

		virtual void InitializeDescriptors(const UInt) = 0;

#ifdef DNN_LEAN
		inline void ZeroGradient(const UInt batchSize)
		{
			InputLayer->NeuronsD1.resize(batchSize, InputLayer->C, InputLayer->H, InputLayer->W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
		}

		inline void ZeroGradientMulti(const UInt batchSize)
		{
			for (auto& inputLayer : Inputs)
				inputLayer->NeuronsD1.resize(batchSize, inputLayer->C, inputLayer->H, inputLayer->W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
		}

		inline void ReleaseGradient()
		{
			if (!InplaceBwd)
				NeuronsD1.release();
		}
#endif // DNN_LEAN

		virtual void SetBatchSize(const UInt batchSize)
		{
			while (RefreshingStats.load())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
				std::this_thread::yield();
			}
			
			Neurons.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
#ifndef DNN_LEAN
			if (!InplaceBwd)
				NeuronsD1.resize(batchSize, C, H, W, dnnl::memory::data_type::f32, BlockedFmt, Device.engine);
#else
			ReleaseGradient();
#endif // DNN_LEAN

			InitializeDescriptors(batchSize);
		}

		virtual void ForwardProp(const UInt batchSize, const bool training) = 0;

		virtual void BackwardProp(const UInt batchSize) = 0;

		bool RefreshStatistics(const UInt batchSize)
		{
			if (!RefreshingStats.load())
			{
				RefreshingStats.store(true);

				if (!Neurons.empty())
				{
					const auto ncdhw = batchSize * CDHW;

					NeuronsStats.Min = std::numeric_limits<Float>::max();
					NeuronsStats.Max = std::numeric_limits<Float>::lowest();
					
					auto sum = Float(0);
					
					if (ncdhw % VectorSize == 0ull)
					{
						VecFloat neurons;
						for (auto i = 0ull; i < ncdhw; i += VectorSize)
						{
							neurons.load_a(&Neurons[i]);
							NeuronsStats.Min = std::min(NeuronsStats.Min, horizontal_min(neurons));
							NeuronsStats.Max = std::max(NeuronsStats.Max, horizontal_max(neurons));

							if ((NeuronsStats.Min < -NEURONS_LIMIT) || (NeuronsStats.Max > NEURONS_LIMIT))
								goto FAIL;

							sum += horizontal_add(neurons);
						}

						if (!std::isnan(sum) && !std::isinf(sum))
						{
							NeuronsStats.Mean = sum / ncdhw;
							
							auto vecSum = VecFloat(0);
							
							for (auto i = 0ull; i < ncdhw; i += VectorSize)
							{
								neurons.load_a(&Neurons[i]);
								vecSum += square(neurons - NeuronsStats.Mean);
							}
							sum = horizontal_add(vecSum);
						
							if (!std::isnan(sum) && !std::isinf(sum))
								NeuronsStats.StdDev = std::sqrt(sum / ncdhw);
							else
								NeuronsStats.StdDev = Float(0);
						}
						else
							goto FAIL;
					}
					else
					{
						for (auto i = 0ull; i < ncdhw; i++)
						{
							NeuronsStats.Min = std::min(NeuronsStats.Min, Neurons[i]);
							NeuronsStats.Max = std::max(NeuronsStats.Max, Neurons[i]);

							if ((NeuronsStats.Min < -NEURONS_LIMIT) || (NeuronsStats.Max > NEURONS_LIMIT))
								goto FAIL;

							sum += Neurons[i];						
						}

						if (!std::isnan(sum) && !std::isinf(sum))
						{
							NeuronsStats.Mean = sum / ncdhw;
							sum = Float(0);
							
							PRAGMA_OMP_SIMD()
							for (auto i = 0ull; i < ncdhw; i++)
								sum += FloatSquare(Neurons[i] - NeuronsStats.Mean);
							
							if (!std::isnan(sum) && !std::isinf(sum))
								NeuronsStats.StdDev = std::sqrt(sum / ncdhw);
							else
								NeuronsStats.StdDev = Float(0);
						}
						else
							goto FAIL;
					}
				}

				if (HasWeights)
				{
					WeightsStats.Min = std::numeric_limits<Float>::max();
					WeightsStats.Max = std::numeric_limits<Float>::lowest();
					
					auto sum = Float(0);

					for (auto i = 0ull; i < Weights.size(); i++)
					{
						WeightsStats.Min = std::min(WeightsStats.Min, Weights[i]);
						WeightsStats.Max = std::max(WeightsStats.Max, Weights[i]);

						if ((WeightsStats.Min < -WEIGHTS_LIMIT) || (WeightsStats.Max > WEIGHTS_LIMIT))
							goto FAIL;

						sum += Weights[i];
					}

					if (!std::isnan(sum) && !std::isinf(sum))
					{
						WeightsStats.Mean = sum / Weights.size();
						sum = Float(0);
						for (auto i = 0ull; i < Weights.size(); i++)
							sum += FloatSquare(Weights[i] - WeightsStats.Mean);

						if (!std::isnan(sum) && !std::isinf(sum))
							WeightsStats.StdDev = std::sqrt(sum / Weights.size());
						else
							WeightsStats.StdDev = Float(0);
					}
					else
						goto FAIL;

					if (HasBias)
					{
						BiasesStats.Min = std::numeric_limits<Float>::max();
						BiasesStats.Max = std::numeric_limits<Float>::lowest();
						
						sum = Float(0);
						for (auto i = 0ull; i < BiasCount; i++)
						{
							BiasesStats.Min = std::min(BiasesStats.Min, Biases[i]);
							BiasesStats.Max = std::max(BiasesStats.Max, Biases[i]);

							if ((BiasesStats.Min < -WEIGHTS_LIMIT) || (BiasesStats.Max > WEIGHTS_LIMIT))
								goto FAIL;

							sum += Biases[i];
						}

						if (!std::isnan(sum) && !std::isinf(sum))
						{
							BiasesStats.Mean = sum / BiasCount;
							sum = Float(0);
							for (auto i = 0ull; i < BiasCount; i++)
								sum += FloatSquare(Biases[i] - BiasesStats.Mean);

							if (!std::isnan(sum) && !std::isinf(sum))
								BiasesStats.StdDev = std::sqrt(sum / BiasCount);
							else
								BiasesStats.StdDev = Float(0);
						}
						else
							goto FAIL;
					}
				}

				RefreshingStats.store(false);

				return true;

			FAIL:
				NeuronsStats.Min = Float(0);
				NeuronsStats.Max = Float(0);
				NeuronsStats.Mean = Float(0);
				NeuronsStats.StdDev = Float(0);

				WeightsStats.Min = Float(0);
				WeightsStats.Max = Float(0);
				WeightsStats.Mean = Float(0);
				WeightsStats.StdDev = Float(0);

				BiasesStats.Min = Float(0);
				BiasesStats.Max = Float(0);
				BiasesStats.Mean = Float(0);
				BiasesStats.StdDev = Float(0);

				RefreshingStats.store(false);

				return false;
			}
			else
				return true;
		}

		void CheckOptimizer(const Optimizers optimizer)
		{
			auto dirty = false;

			if (std::isnan(B1) || std::isinf(B1))
				dirty = true;
			if (std::isnan(B2) || std::isinf(B2))
				dirty = true;
			if (std::isnan(Gamma) || std::isinf(Gamma))
				dirty = true;

			switch (optimizer)
			{
			
			case Optimizers::AdaBound:
			case Optimizers::AdaBoundW:
			case Optimizers::AdaDelta:
			case Optimizers::Adam:
			case Optimizers::AdamW:
			case Optimizers::Adamax:
			case Optimizers::AmsBound:
			case Optimizers::AmsBoundW:
			{
				if (HasWeights)
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

				if (HasBias && !dirty)
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
			break;

			case Optimizers::AdaGrad:
			case Optimizers::NAG:
			case Optimizers::RMSProp:
			case Optimizers::SGDMomentum:
			case Optimizers::SGDW:
			{
				if (HasWeights)
					for (auto i = 0ull; i < Weights.size(); i++)
					{
						if (std::isnan(WeightsPar1[i]) || std::isinf(WeightsPar1[i]))
						{
							dirty = true;
							break;
						}
					}

				if (HasBias && !dirty)
					for (auto i = 0ull; i < BiasCount; i++)
					{
						if (std::isnan(BiasesPar1[i]) || std::isinf(BiasesPar1[i]))
						{
							dirty = true;
							break;
						}
					}
			}
			break;

			default:
				break;
			}

			if (dirty)
				ResetOptimizer(optimizer);
		}

		void ResetOptimizer(const Optimizers optimizer)
		{
			if (HasWeights)
			{
				B1 = Float(0);
				B2 = Float(0);
				Gamma = Float(0);

				const auto weightsSize = WeightsMemDesc->get_size() / sizeof(Float);
				const auto biasesSize = HasBias ? BiasCount : 0;

				WeightsD1.resize(weightsSize, Float(0));
				BiasesD1.resize(biasesSize, Float(0));
			
				switch (optimizer)
				{
				case Optimizers::AdaBound:
				case Optimizers::AdaBoundW:
				case Optimizers::AdaDelta:
				case Optimizers::Adam:
				case Optimizers::Adamax:
				case Optimizers::AdamW:
				case Optimizers::AmsBound:
				case Optimizers::AmsBoundW:
					WeightsPar1.resize(weightsSize);
					WeightsPar2.resize(weightsSize);
					BiasesPar1.resize(biasesSize);
					BiasesPar2.resize(biasesSize);
					std::fill(WeightsPar1.begin(), WeightsPar1.end(), Float(0));
					std::fill(WeightsPar2.begin(), WeightsPar2.end(), Float(0));
					std::fill(BiasesPar1.begin(), BiasesPar1.end(), Float(0));
					std::fill(BiasesPar2.begin(), BiasesPar2.end(), Float(0));
					break;

				case Optimizers::AdaGrad:
				case Optimizers::NAG:
				case Optimizers::RMSProp:
				case Optimizers::SGDMomentum:
				case Optimizers::SGDW:
					WeightsPar1.resize(weightsSize);
					WeightsPar2.resize(0);
					BiasesPar1.resize(biasesSize);
					BiasesPar2.resize(0);
					std::fill(WeightsPar1.begin(), WeightsPar1.end(), Float(0));
					std::fill(BiasesPar1.begin(), BiasesPar1.end(), Float(0));					
					break;

				case Optimizers::SGD:
					WeightsPar1.resize(0);
					WeightsPar2.resize(0);
					BiasesPar1.resize(0);
					BiasesPar2.resize(0);
					break;
				}
			}
		}

		void SetOptimizer(const Optimizers optimizer)
		{
			if (HasWeights)
			{
				const auto weightsSize = WeightsMemDesc->get_size() / sizeof(Float);
				const auto biasesSize = HasBias ? BiasCount : 0;

				WeightsD1.resize(weightsSize, Float(0));
				BiasesD1.resize(biasesSize, Float(0));

				switch (optimizer)
				{
				case Optimizers::AdaBound:
				case Optimizers::AdaBoundW:
				case Optimizers::AdaDelta:
				case Optimizers::Adam:
				case Optimizers::Adamax:
				case Optimizers::AdamW:
				case Optimizers::AmsBound:
				case Optimizers::AmsBoundW:
					WeightsPar1.resize(weightsSize, Float(0));
					WeightsPar2.resize(weightsSize, Float(0));
					BiasesPar1.resize(biasesSize, Float(0));
					BiasesPar2.resize(biasesSize, Float(0));
					break;

				case Optimizers::AdaGrad:
				case Optimizers::NAG:
				case Optimizers::RMSProp:
				case Optimizers::SGDMomentum:
				case Optimizers::SGDW:
					WeightsPar1.resize(weightsSize, Float(0));
					WeightsPar2.resize(0);
					BiasesPar1.resize(biasesSize, Float(0));
					BiasesPar2.resize(0);
					break;

				case Optimizers::SGD:
					WeightsPar1.resize(0);
					WeightsPar2.resize(0);
					BiasesPar1.resize(0);
					BiasesPar2.resize(0);
					break;
				}
			}
		}

		virtual void ResetWeights(const Fillers weightsFiller, const FillerModes weightsFillerMode, const Float weightsGain, const Float weightsScale, const Fillers biasesFiller, const FillerModes biasesFillerMode, const Float biasesGain, const Float biasesScale)
		{
			if (HasWeights)
			{
				if (UseDefaultParameters)
				{
					WeightsFiller = weightsFiller;
					WeightsFillerMode = weightsFillerMode;
					WeightsGain = weightsGain;
					WeightsScale = weightsScale;
				}

				auto weights = FloatVector(WeightCount);

				auto weightsScope = Float(FanIn());
				switch (weightsFillerMode)
				{
				case FillerModes::Avg:
					weightsScope = Float(FanIn() + FanOut()) / Float(2);
					break;
				case FillerModes::In:
					weightsScope = Float(FanIn());
					break;
				case FillerModes::Out:
					weightsScope = Float(FanOut());
					break;
				}

				switch (weightsFiller)
				{
				case Fillers::Constant:
				{
					std::fill_n(weights.begin(), WeightCount, WeightsScale);
				}
				break;

				case Fillers::HeNormal:
				{
					auto stddev = weightsGain * std::sqrt(Float(2) / weightsScope);
					auto distribution = std::normal_distribution<Float>(Float(0), stddev);
					std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::HeUniform:
				{
					auto limit = weightsGain * std::sqrt(Float(6) / weightsScope);
					auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
					std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::LeCunNormal:
				{
					auto stddev = weightsGain * std::sqrt(Float(1) / weightsScope);
					auto distribution = std::normal_distribution<Float>(Float(0), stddev);
					std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::LeCunUniform:
				{
					auto limit = weightsGain * std::sqrt(Float(3) / weightsScope);
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
					const auto limit = 2 * std::abs(WeightsScale);
					auto distribution = std::normal_distribution<Float>(Float(0), WeightsScale);
					auto x = limit + Float(1);
					std::generate_n(weights.begin(), WeightCount, [&]()
					{
						do { x = distribution(RandomEngine); } while (std::abs(x) > limit);
						return x;
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
					auto stddev = weightsGain * std::sqrt(Float(2) / Float(FanIn() + FanOut()));
					auto distribution = std::normal_distribution<Float>(Float(0), stddev);
					std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::XavierUniform:
				{
					auto limit = weightsGain * std::sqrt(Float(6) / Float(FanIn() + FanOut()));
					auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
					std::generate_n(weights.begin(), WeightCount, [&]() { return distribution(RandomEngine); });
				}
				break;
				}

				if (*PersistWeightsMemDesc != *WeightsMemDesc)
				{
					Weights.resize(WeightsMemDesc->get_size() / sizeof(Float));
					WeightsD1.resize(WeightsMemDesc->get_size() / sizeof(Float));

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
					BiasesFillerMode = biasesFillerMode;
					BiasesGain = biasesGain;
					BiasesScale = biasesScale;
				}

				auto biasesScope = Float(FanIn());
				switch (biasesFillerMode)
				{
				case FillerModes::Avg:
					biasesScope = Float(FanIn() + FanOut()) / Float(2);
					break;
				case FillerModes::In:
					biasesScope = Float(FanIn());
					break;
				case FillerModes::Out:
					biasesScope = Float(FanOut());
					break;
				}

				switch (biasesFiller)
				{
				case Fillers::Constant:
				{
					std::fill_n(Biases.begin(), BiasCount, BiasesScale);
				}
				break;

				case Fillers::HeNormal:
				{
					auto stddev = biasesGain * std::sqrt(Float(2) / biasesScope);
					auto distribution = std::normal_distribution<Float>(Float(0), stddev);
					std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::HeUniform:
				{
					auto limit = biasesGain * std::sqrt(Float(6) / biasesScope);
					auto distribution = std::uniform_real_distribution<Float>(-limit, limit);
					std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::LeCunNormal:
				{
					auto stddev = biasesGain * std::sqrt(Float(1) / biasesScope);
					auto distribution = std::normal_distribution<Float>(Float(0), stddev);
					std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::LeCunUniform:
				{
					auto limit = biasesGain * std::sqrt(Float(3) / biasesScope);
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
					const auto limit = 2 * std::abs(BiasesScale);
					auto x = limit + Float(1);
					std::generate_n(Biases.begin(), BiasCount, [&]()
					{
						do { x = distribution(RandomEngine); } while (std::abs(x) > limit);
						return x;
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
					auto stddev = biasesGain * std::sqrt(Float(2) / Float(FanIn() + FanOut()));
					auto distribution = std::normal_distribution<Float>(Float(0), stddev);
					std::generate_n(Biases.begin(), BiasCount, [&]() { return distribution(RandomEngine); });
				}
				break;

				case Fillers::XavierUniform:
				{
					auto limit = biasesGain * std::sqrt(Float(6) / Float(FanIn() + FanOut()));
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
			if (HasBias)
				std::fill_n(BiasesD1.begin(), BiasCount, Float(0));
		}

		void UpdateWeights(const TrainingRate& rate, const Optimizers optimizer, const bool disableLocking)
		{
			if (HasWeights && (disableLocking || (!disableLocking && !LockUpdate.load())))
			{
				switch (optimizer)
				{
				case Optimizers::AdaBound:
					AdaBound(rate);
					break;
				case Optimizers::AdaBoundW:
					AdaBoundW(rate);
					break;
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
				case Optimizers::AdamW:
					AdamW(rate);
					break;
				case Optimizers::AmsBound:
					AdaBound(rate, true);
					break;
				case Optimizers::AmsBoundW:
					AdaBoundW(rate, true);
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
				case Optimizers::SGDW:
					SGDW(rate);
					break;
				}
			}
		}

		inline void AdaBound(const TrainingRate& rate, const bool amsbound = false)
		{
			const auto beta1 = rate.Momentum;
			const auto beta2 = rate.Beta2;
			const auto eps = rate.Eps;
			const auto oneMinusBeta1 = Float(1) - beta1;
			const auto oneMinusBeta2 = Float(1) - beta2;
			const auto batchRecip = Float(1) / rate.BatchSize;
			B1 = B1 == Float(0) ? beta1 : B1;
			B2 = B2 == Float(0) ? beta2 : B2;
			const auto oneMinusB1 = Float(1) - B1;
			const auto oneMinusB2 = Float(1) - B2;
			Gamma = Gamma == Float(0) ? rate.Gamma : Gamma;
			const auto finalRate = rate.FinalRate * rate.MaximumRate * WeightsLRM;
			const auto lowerBound = finalRate * (Float(1) - (Float(1) / (Gamma + rate.Gamma)));
			const auto upperBound = finalRate * (Float(1) + (Float(1) / Gamma));
			const auto weightDecay = rate.L2Penalty * WeightsWDM;
			const auto step_size = rate.MaximumRate * WeightsLRM * std::sqrt(oneMinusB2) / oneMinusB1;

			if (!amsbound)
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i] * batchRecip);
					WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
					Weights[i] -= Clamp<Float>(step_size / (std::sqrt(WeightsPar2[i]) + eps), lowerBound, upperBound) * WeightsPar1[i];
				}
			else
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i] * batchRecip);
					WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
					Weights[i] -= Clamp<Float>(step_size / (std::sqrt(std::max(WeightsPar1[i], WeightsPar2[i])) + eps), lowerBound, upperBound) * WeightsPar1[i];
				}

			if (HasBias)
			{
				const auto finalRate = rate.FinalRate * rate.MaximumRate * BiasesLRM;
				const auto lowerBound = finalRate * (Float(1) - (Float(1) / (Gamma + rate.Gamma)));
				const auto upperBound = finalRate * (Float(1) + (Float(1) / Gamma));
				const auto weightDecay = rate.L2Penalty * BiasesWDM;
				const auto step_size = rate.MaximumRate * BiasesLRM * std::sqrt(oneMinusB2) / oneMinusB1;

				if (!amsbound)
					PRAGMA_OMP_SIMD()
					for (auto i = 0ull; i < BiasCount; i++)
					{
						BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i] * batchRecip);
						BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
						Biases[i] -= Clamp<Float>(step_size / (std::sqrt(BiasesPar2[i]) + eps), lowerBound, upperBound) * BiasesPar1[i];
					}
				else
					PRAGMA_OMP_SIMD()
					for (auto i = 0ull; i < BiasCount; i++)
					{
						BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i] * batchRecip);
						BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
						Biases[i] -= Clamp<Float>(step_size / (std::sqrt(std::max(BiasesPar1[i], BiasesPar2[i])) + eps), lowerBound, upperBound) * BiasesPar1[i];
					}
			}

			B1 *= beta1;
			B2 *= beta2;
			Gamma += rate.Gamma;
		}

		inline void AdaBoundW(const TrainingRate& rate, const bool amsbound = false)
		{
			const auto beta1 = rate.Momentum;
			const auto beta2 = rate.Beta2;
			const auto eps = rate.Eps;
			const auto oneMinusBeta1 = Float(1) - beta1;
			const auto oneMinusBeta2 = Float(1) - beta2;
			const auto batchRecip = Float(1) / rate.BatchSize;
			B1 = B1 == Float(0) ? beta1 : B1;
			B2 = B2 == Float(0) ? beta2 : B2;
			const auto oneMinusB1 = Float(1) - B1;
			const auto oneMinusB2 = Float(1) - B2;
			Gamma = Gamma == Float(0) ? rate.Gamma : Gamma;
			const auto finalRate = rate.FinalRate * rate.MaximumRate * WeightsLRM;
			const auto lowerBound = finalRate * (Float(1) - (Float(1) / (Gamma + rate.Gamma)));
			const auto upperBound = finalRate * (Float(1) + (Float(1) / Gamma));
			const auto weightDecay = rate.L2Penalty * WeightsWDM;
			const auto step_size = rate.MaximumRate * WeightsLRM * std::sqrt(oneMinusB2) / oneMinusB1;

			if (!amsbound)
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					WeightsD1[i] += weightDecay * Weights[i];
					WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i] * batchRecip);
					WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
					Weights[i] -= Clamp<Float>(step_size / (std::sqrt(WeightsPar2[i]) + eps), lowerBound, upperBound) * WeightsPar1[i];
				}
			else
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < Weights.size(); i++)
				{
					WeightsD1[i] += weightDecay * Weights[i];
					WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i] * batchRecip);
					WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
					Weights[i] -= Clamp<Float>(step_size / (std::sqrt(std::max(WeightsPar1[i], WeightsPar2[i])) + eps), lowerBound, upperBound) * WeightsPar1[i];
				}

			if (HasBias)
			{
				const auto finalRate = rate.FinalRate * rate.MaximumRate * BiasesLRM;
				const auto lowerBound = finalRate * (Float(1) - (Float(1) / (Gamma + rate.Gamma)));
				const auto upperBound = finalRate * (Float(1) + (Float(1) / Gamma));
				const auto weightDecay = rate.L2Penalty * BiasesWDM;
				const auto step_size = rate.MaximumRate * BiasesLRM * std::sqrt(oneMinusB2) / oneMinusB1;

				if (!amsbound)
					PRAGMA_OMP_SIMD()
					for (auto i = 0ull; i < BiasCount; i++)
					{
						BiasesD1[i] += weightDecay * Biases[i];
						BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i] * batchRecip);
						BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
						Biases[i] -= Clamp<Float>(step_size / (std::sqrt(BiasesPar2[i]) + eps), lowerBound, upperBound) * BiasesPar1[i];
					}
				else
					PRAGMA_OMP_SIMD()
					for (auto i = 0ull; i < BiasCount; i++)
					{
						BiasesD1[i] += weightDecay * Biases[i];
						BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i] * batchRecip);
						BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
						Biases[i] -= Clamp<Float>(step_size / (std::sqrt(std::max(BiasesPar1[i], BiasesPar2[i])) + eps), lowerBound, upperBound) * BiasesPar1[i];
					}
			}
			
			B1 *= beta1;
			B2 *= beta2;
			Gamma += rate.Gamma;
		}

		inline void AdaDelta(const TrainingRate& rate)
		{
			const auto lr = -rate.MaximumRate * WeightsLRM;
			const auto momentum = rate.Momentum;
			const auto oneMinMomentum = Float(1) - momentum;
			const auto eps = rate.Eps;
			const auto batchRecip = Float(1) / rate.BatchSize;

			PRAGMA_OMP_SIMD()
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
				PRAGMA_OMP_SIMD()
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
			const auto eps = rate.Eps;
			const auto batchRecip = Float(1) / rate.BatchSize;

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] += FloatSquare(WeightsD1[i] * batchRecip);
				Weights[i] -= lr * WeightsD1[i] / (std::sqrt(WeightsPar1[i]) + eps);
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM;
				PRAGMA_OMP_SIMD()
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
			const auto beta2 = rate.Beta2;
			const auto lr = rate.MaximumRate * WeightsLRM;
			const auto eps = rate.Eps;
			const auto oneMinusBeta1 = (Float(1) - beta1) / rate.BatchSize;
			const auto oneMinusBeta2 = Float(1) - beta2;
			const auto batchRecip = Float(1) / rate.BatchSize;
			B1 = B1 == Float(0) ? beta1 : B1;
			B2 = B2 == Float(0) ? beta2 : B2;
			const auto oneMinusB1 = Float(1) - B1;
			const auto oneMinusB2 = Float(1) - B2;

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i]);
				WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
				Weights[i] -= lr * (WeightsPar1[i] / oneMinusB1) / std::sqrt((WeightsPar2[i] / oneMinusB2) + eps);
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM;
				PRAGMA_OMP_SIMD()
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
			const auto beta1 = rate.Momentum;
			B1 = B1 == Float(0) ? beta1 : B1;
			const auto lr = rate.MaximumRate * WeightsLRM / (Float(1) - B1);
			const auto batchRecip = Float(1) / rate.BatchSize;
			const auto oneMinusBeta1 = (Float(1) - beta1) / rate.BatchSize;
			const auto beta2 = rate.Beta2;
			const auto eps = rate.Eps;
									
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

		inline void AdamW(const TrainingRate& rate)
		{
			const auto beta1 = rate.Momentum;
			const auto beta2 = rate.Beta2;
			const auto lr = rate.MaximumRate * WeightsLRM;
			const auto weightDecay = rate.L2Penalty * WeightsWDM;
			const auto eps = rate.Eps;
			const auto oneMinusBeta1 = Float(1) - beta1;
			const auto oneMinusBeta2 = Float(1) - beta2;
			const auto batchRecip = Float(1) / rate.BatchSize;
			B1 = B1 == Float(0) ? beta1 : B1;
			B2 = B2 == Float(0) ? beta2 : B2;
			const auto oneMinusB1 = Float(1) - B1;
			const auto oneMinusB2 = Float(1) - B2;

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (beta1 * WeightsPar1[i]) + (oneMinusBeta1 * WeightsD1[i] * batchRecip);
				WeightsPar2[i] = (beta2 * WeightsPar2[i]) + (oneMinusBeta2 * FloatSquare(WeightsD1[i] * batchRecip));
				Weights[i] -= lr * ((WeightsPar1[i] / oneMinusB1) / std::sqrt((WeightsPar2[i] / oneMinusB2) + eps) + (weightDecay * Weights[i]));
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM;
				const auto weightDecay = rate.L2Penalty * BiasesWDM;
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < BiasCount; i++)
				{
					BiasesPar1[i] = (beta1 * BiasesPar1[i]) + (oneMinusBeta1 * BiasesD1[i] * batchRecip);
					BiasesPar2[i] = (beta2 * BiasesPar2[i]) + (oneMinusBeta2 * FloatSquare(BiasesD1[i] * batchRecip));
					Biases[i] -= lr * ((BiasesPar1[i] / oneMinusB1) / std::sqrt((BiasesPar2[i] / oneMinusB2) + eps) + (weightDecay * Biases[i]));
				}
			}

			B1 *= beta1;
			B2 *= beta2;
		}

		inline void NAG(const TrainingRate& rate)
		{
			const auto lr = rate.MaximumRate * WeightsLRM;
			const auto l2Penalty = rate.L2Penalty * WeightsWDM * lr;
			const auto momentum = rate.Momentum;
			const auto momentumPlusOne = momentum + Float(1);
			const auto batchRecip = Float(1) / rate.BatchSize * lr;
			
			PRAGMA_OMP_SIMD()
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
				PRAGMA_OMP_SIMD()
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
			const auto eps = rate.Eps;
			const auto momentum = rate.Momentum;
			const auto oneMinusMomentum = Float(1) - momentum;
			const auto batchRecip = Float(1) / rate.BatchSize;

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (momentum * WeightsPar1[i]) + (oneMinusMomentum * FloatSquare(WeightsD1[i] * batchRecip));
				Weights[i] -= lr * WeightsD1[i] / std::sqrt(WeightsPar1[i] + eps);
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;
				PRAGMA_OMP_SIMD()
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

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
				Weights[i] -= (lr * WeightsD1[i]) - (l2Penalty * Weights[i]);

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;;
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < BiasCount; i++)
					Biases[i] -= lr * BiasesD1[i];
			}
		}

		inline void SGDMomentum(const TrainingRate& rate)
		{
			const auto momentum = rate.Momentum;
			const auto lr = rate.MaximumRate * WeightsLRM / rate.BatchSize;
			const auto l2Penalty = rate.MaximumRate * WeightsLRM * rate.L2Penalty * WeightsWDM;

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (momentum * WeightsPar1[i]) - (lr * WeightsD1[i]) - (l2Penalty * Weights[i]);
				Weights[i] += WeightsPar1[i];
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < BiasCount; i++)
				{
					BiasesPar1[i] = (momentum * BiasesPar1[i]) - (lr * BiasesD1[i]);
					Biases[i] += BiasesPar1[i];
				}
			}
		}

		inline void SGDW(const TrainingRate& rate)
		{
			const auto momentum = rate.Momentum;
			const auto lr = rate.MaximumRate * WeightsLRM / rate.BatchSize;
			const auto l2Penalty = rate.L2Penalty * WeightsWDM;

			PRAGMA_OMP_SIMD()
			for (auto i = 0ull; i < Weights.size(); i++)
			{
				WeightsPar1[i] = (momentum * WeightsPar1[i]) - (lr * WeightsD1[i]);
				Weights[i] += WeightsPar1[i] - (l2Penalty * Weights[i]);
			}

			if (HasBias)
			{
				const auto lr = rate.MaximumRate * BiasesLRM / rate.BatchSize;
				PRAGMA_OMP_SIMD()
				for (auto i = 0ull; i < BiasCount; i++)
				{
					BiasesPar1[i] = (momentum * BiasesPar1[i]) - (lr * BiasesD1[i]);
					Biases[i] += BiasesPar1[i];
				}
			}
		}

		/*UInt OffsetPaddedMem(const UInt n, const UInt c, const UInt h, const UInt w) const
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
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::AdaDelta:
						case Optimizers::Adam:
						case Optimizers::Adamax:
						case Optimizers::AdamW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
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

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::Adamax:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								os.write(reinterpret_cast<const char*>(&Gamma), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						case Optimizers::SGDW:
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

						case Optimizers::SGD:
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
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::AdaDelta:
						case Optimizers::Adam:
						case Optimizers::Adamax:
						case Optimizers::AdamW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::Adamax:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								os.write(reinterpret_cast<const char*>(&Gamma), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						case Optimizers::SGDW:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::SGD:
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
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::AdaDelta:
						case Optimizers::Adam:
						case Optimizers::Adamax:
						case Optimizers::AdamW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
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

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::Adamax:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								is.read(reinterpret_cast<char*>(&Gamma), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						case Optimizers::SGDW:
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

						case Optimizers::SGD:
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
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::AdaDelta:
						case Optimizers::Adam:
						case Optimizers::Adamax:
						case Optimizers::AdamW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::Adamax:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::Adam:
							case Optimizers::AdamW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}

							switch (optimizer)
							{
							case Optimizers::AdaBound:
							case Optimizers::AdaBoundW:
							case Optimizers::AmsBound:
							case Optimizers::AmsBoundW:
								is.read(reinterpret_cast<char*>(&Gamma), std::streamsize(sizeof(Float)));
								break;
							default:
								break;
							}
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						case Optimizers::SGDW:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::SGD:
						break;
						}
				}
			}
		}

		virtual std::streamsize GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const
		{
			auto weightsSize = std::streamsize(0);

			if (HasWeights)
			{
				weightsSize += sizeof(std::atomic<bool>);

				if (persistOptimizer)
				{
					switch (optimizer)
					{
					case Optimizers::AdaBound:
					case Optimizers::AdaBoundW:
					case Optimizers::AdaDelta:
					case Optimizers::Adam:
					case Optimizers::Adamax:
					case Optimizers::AdamW:
					case Optimizers::AmsBound:
					case Optimizers::AmsBoundW:
					{
						weightsSize += 3 * WeightCount * sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);

						switch (optimizer)
						{
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::Adam:
						case Optimizers::Adamax:
						case Optimizers::AdamW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
							weightsSize += sizeof(Float);
							break;
						default:
							break;
						}

						switch (optimizer)
						{
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::Adam:
						case Optimizers::AdamW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
							weightsSize += sizeof(Float);
							break;
						default:
							break;
						}

						switch (optimizer)
						{
						case Optimizers::AdaBound:
						case Optimizers::AdaBoundW:
						case Optimizers::AmsBound:
						case Optimizers::AmsBoundW:
							weightsSize += sizeof(Float);
							break;
						default:
							break;
						}
					}
					break;

					case Optimizers::AdaGrad:
					case Optimizers::NAG:
					case Optimizers::RMSProp:
					case Optimizers::SGDMomentum:
					case Optimizers::SGDW:
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
					
		virtual UInt GetNeuronsSize(const UInt batchSize) const
		{
#ifndef DNN_LEAN
			return batchSize * PaddedCDHW *  sizeof(Float) * (InplaceBwd ? 1 : 2);
#else
			return batchSize * PaddedCDHW * sizeof(Float);
#endif // DNN_LEAN
		}

		virtual ByteArray GetImage(const Byte) { return ByteArray(); }
	};
}