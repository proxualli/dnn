#pragma once
#include "Activation.h"
#include "Add.h"
#include "Average.h"
#include "AvgPooling.h"
#include "BatchNorm.h"
#include "BatchNormActivation.h"
#include "BatchNormActivationDropout.h"
#include "BatchNormRelu.h"
#include "ChannelMultiply.h"
#include "ChannelShuffle.h"
#include "ChannelSplit.h"
#include "ChannelZeroPad.h"
#include "Concat.h"
#include "Convolution.h"
#include "ConvolutionTranspose.h"
#include "Cost.h"
#include "Dense.h"
#include "DepthwiseConvolution.h"
#include "Divide.h"
#include "Dropout.h"
#include "GlobalAvgPooling.h"
#include "GlobalMaxPooling.h"
#include "Input.h"
#include "LayerNorm.h"
#include "LocalResponseNorm.h"
#include "LogSoftmax.h"
#include "Max.h"
#include "MaxPooling.h"
#include "Min.h"
#include "Multiply.h"
#include "PartialDepthwiseConvolution.h"
#include "PRelu.h"
#include "Softmax.h"
#include "Substract.h"
#include "Resampling.h"


namespace dnn
{
	enum class TaskStates
	{
		Paused = 0,
		Running = 1,
		Stopped = 2
	};

	enum class States
	{
		Idle = 0,
		NewEpoch = 1,
		Testing = 2,
		Training = 3,
		SaveWeights = 4,
		Completed = 5
	};

	struct CheckMsg
	{
		UInt Row;
		UInt Column;
		bool Error;
		std::string Message;
		
		CheckMsg(const UInt row = 0, const UInt column = 0, const std::string& message = "", const bool error = true) :
			Row(row),
			Column(column),
			Message(message),
			Error(error)
		{
		}
	};
	
	struct TrainingStrategy
	{
		Float Epochs;
		UInt BatchSize;
		UInt Height;
		UInt Width;
		Float Momentum;
		Float Beta2;
		Float Gamma;
		Float L2Penalty;
		Float Dropout;
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

		TrainingStrategy(const Float epochs, const UInt batchSize, const UInt height, const UInt width, const Float momentum, const Float beta2, const Float gamma, const Float l2penalty, const Float dropout, const bool horizontalFlip, const bool verticalFlip, const Float inputDropout, const Float cutout, const bool cutMix, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const Interpolations interpolation, const Float scaling, const Float rotation) :
			Epochs(epochs),
			BatchSize(batchSize),
			Height(height),
			Width(width),
			Momentum(momentum),
			Beta2(beta2),
			Gamma(gamma),
			L2Penalty(l2penalty),
			Dropout(dropout),
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
			if (Epochs <= 0 || Epochs >= 1)
				throw std::invalid_argument("Epochs out of range in TrainingStrategy");
			if (BatchSize == 0)
				throw std::invalid_argument("BatchSize cannot be zero in TrainingStrategy");
			if (Height == 0)
				throw std::invalid_argument("Height cannot be zero in TrainingStrategy");
			if (Width == 0)
				throw std::invalid_argument("Width cannot be zero in TrainingStrategy");
			if (Momentum < 0 || Momentum > 1)
				throw std::invalid_argument("Momentum out of range in TrainingStrategy");
			if (Beta2 < 0 || Beta2 > 1)
				throw std::invalid_argument("Beta2 out of range in TrainingStrategy");
			if (Gamma < 0 || Gamma > 1)
				throw std::invalid_argument("Gamma out of range in TrainingStrategy");
			if (L2Penalty < 0 || L2Penalty > 1)
				throw std::invalid_argument("L2Penalty out of range in TrainingStrategy");
			if (Dropout < 0 || Dropout >= 1)
				throw std::invalid_argument("Dropout out of range in TrainingStrategy");
			if (InputDropout < 0 || InputDropout >= 1)
				throw std::invalid_argument("InputDropout out of range in TrainingStrategy");
			if (Cutout < 0 || Cutout > 1)
				throw std::invalid_argument("Cutout out of range in TrainingStrategy");
			if (AutoAugment < 0 || AutoAugment > 1)
				throw std::invalid_argument("AutoAugment out of range in TrainingStrategy");
			if (ColorCast < 0 || ColorCast > 1)
				throw std::invalid_argument("ColorCast out of range in TrainingStrategy");
			if (ColorAngle > 180)
				throw std::invalid_argument("ColorAngle cannot be zero in TrainingStrategy");
			if (Distortion < 0 || Distortion > 1)
				throw std::invalid_argument("Distortion out of range in TrainingStrategy");
			if (Scaling < 0 || Scaling > 100)
				throw std::invalid_argument("Scaling out of range in TrainingStrategy");
			if (Rotation < 0 || Rotation > 180)
				throw std::invalid_argument("Rotation out of range in TrainingStrategy");
		}
	};

	
	class Model
	{
	private:
		std::future<void> task;
		std::vector<UInt> RandomTrainingSamples;
		std::vector<bool> TrainingSamplesHFlip;
		std::vector<bool> TrainingSamplesVFlip;
		std::vector<bool> TestingSamplesHFlip;
		std::vector<bool> TestingSamplesVFlip;
		
	public:
		const std::string Name;
		const std::string Definition;
		const dnnl::engine Engine;
		dnn::Device Device;
		dnnl::memory::format_tag Format;
		Dataprovider* DataProv;
		Datasets Dataset;
		std::atomic<States> State;
		std::atomic<TaskStates> TaskState;
		Costs CostFuction;
		Optimizers Optimizer;
		UInt CostIndex;
		UInt LabelIndex;
		UInt GroupIndex;
		UInt TotalCycles;
		UInt TotalEpochs;
		UInt CurrentCycle;
		UInt CurrentEpoch;
		UInt SampleIndex;
		//UInt LogInterval;
		UInt BatchSize;
		UInt GoToEpoch;
		UInt AdjustedTrainingSamplesCount;
		UInt AdjustedTestingSamplesCount;
		UInt TrainSkipCount;
		UInt TestSkipCount;
		UInt TrainOverflowCount;
		UInt TestOverflowCount;
		UInt C;
		UInt D;
		UInt H;
		UInt W;
		UInt PadC;
		UInt PadD;
		UInt PadH;
		UInt PadW;
		bool MirrorPad;
		bool RandomCrop;
		bool MeanStdNormalization;
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
		Float AlphaFiller;
		Float BetaFiller;
		Float BatchNormMomentum;
		Float BatchNormEps;
		Float Dropout;
		UInt TrainErrors;
		UInt TestErrors;
		Float TrainLoss;
		Float TestLoss;
		Float AvgTrainLoss;
		Float AvgTestLoss;
		Float TrainErrorPercentage;
		Float TestErrorPercentage;
		Float Accuracy;
		Float SampleSpeed;
		Float Rate;
		bool BatchNormScaling;
		bool HasBias;
		bool PersistOptimizer;
		bool DisableLocking;
		TrainingRate CurrentTrainingRate;
		std::vector<TrainingRate> TrainingRates;
		std::vector<TrainingStrategy> TrainingStrategies;
		bool UseTrainingStrategy;
		std::vector<std::unique_ptr<Layer>> Layers;
		std::vector<Cost*> CostLayers;
		std::chrono::duration<Float> fpropTime;
		std::chrono::duration<Float> bpropTime;
		std::chrono::duration<Float> updateTime;
		std::atomic<UInt> FirstUnlockedLayer;
		std::atomic<bool> BatchSizeChanging;
		std::atomic<bool> ResettingWeights;

		void(*NewEpoch)(UInt, UInt, UInt, UInt, Float, Float, Float, bool, bool, Float, Float, bool, Float, Float, UInt, Float, UInt, Float, Float, Float, UInt, UInt, UInt, Float, Float, Float, Float, Float, Float, UInt, Float, Float, Float, UInt);

		Model(const std::string& name, const std::string& definition, Dataprovider* dataprovider) :
			Name(name),
			Definition(definition),
			DataProv(dataprovider),
			Engine(dnnl::engine(dnnl::engine::kind::cpu, 0)),
			Device(dnn::Device(Engine, dnnl::stream(Engine))),
			Format(dnnl::memory::format_tag::any),
			PersistOptimizer(false),
			DisableLocking(true),
			Optimizer(Optimizers::SGD),
			TaskState(TaskStates::Stopped),
			State(States::Idle),
			Dataset(Datasets::cifar10),				// Dataset
			C(3),									// Dim
			D(1),
			H(8),
			W(8),
			MeanStdNormalization(true),				// MeanStd
			MirrorPad(false),						// MirrorPad or ZeroPad
			PadD(0),
			PadC(0),
			PadH(0),
			PadW(0),
			RandomCrop(false),						// RandomCrop
			BatchNormScaling(true),					// Scaling
			BatchNormMomentum(Float(0.995)),		// Momentum
			BatchNormEps(Float(1e-04)),				// Eps
			Dropout(Float(0)),						// Dropout
			WeightsFiller(Fillers::HeNormal),		// WeightsFiller
			WeightsFillerMode(FillerModes::In),		// WeightsFillerMode
			WeightsGain(Float(1)),					// WeightsGain
			WeightsScale(Float(0.05)),				// WeightsScale
			WeightsLRM(Float(1)),					// WeightsLRM
			WeightsWDM(Float(1)),					// WeightsWDM
			BiasesFiller(Fillers::Constant),		// BiasesFiller
			BiasesFillerMode(FillerModes::In),		// BiasesFillerMode
			BiasesGain(Float(1)),					// BiasesGain
			BiasesScale(Float(0)),					// BiasesScale
			BiasesLRM(Float(1)),					// BiasesLRM
			BiasesWDM(Float(1)),					// BiasesWDM
			HasBias(true),							// Biases
			AlphaFiller(Float(0)),					// Alpha
			BetaFiller(Float(0)),					// Beta
			TotalCycles(0),
			CurrentCycle(1),
			TotalEpochs(0),
			GoToEpoch(1),
			CurrentEpoch(1),
			SampleIndex(0),
			BatchSize(1),
			Rate(Float(0)),
			TrainLoss(Float(0)),
			AvgTrainLoss(Float(0)),
			TrainErrors(0),
			TrainErrorPercentage(Float(0)),
			TestLoss(Float(0)),
			AvgTestLoss(Float(0)),
			TestErrors(0),
			TestErrorPercentage(Float(0)),
			Accuracy(Float(0)),
			LabelIndex(0),
			GroupIndex(0),
			CostIndex(0),
			CostFuction(Costs::CategoricalCrossEntropy),
			CostLayers(std::vector<Cost*>()),
			Layers(std::vector<std::unique_ptr<Layer>>()),
			TrainingRates(std::vector<TrainingRate>()),
			fpropTime(std::chrono::duration<Float>(Float(0))),
			bpropTime(std::chrono::duration<Float>(Float(0))),
			updateTime(std::chrono::duration<Float>(Float(0))),
			SampleSpeed(Float(0)),
			NewEpoch(nullptr),
			AdjustedTrainingSamplesCount(0),
			AdjustedTestingSamplesCount(0),
			TrainOverflowCount(0),
			TestOverflowCount(0),
			TrainSkipCount(0),
			TestSkipCount(0),
			BatchSizeChanging(false),
			ResettingWeights(false),
			FirstUnlockedLayer(1),
			UseTrainingStrategy(false),
			TrainingStrategies()
			//LogInterval(10000)
		{
#ifdef DNN_LOG
			dnnl_set_verbose(2);
#else
			dnnl_set_verbose(0);
#endif
			
#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::all);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::prefer_ymm);
#elif defined(DNN_AVX2)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::avx2);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::prefer_ymm);
#elif defined(DNN_AVX)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::avx);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::prefer_ymm);
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::sse41);
			dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::no_hints);
#endif
		}

		virtual ~Model() = default;
		
		// todo: check memory requirements !!!
		void ChangeResolution(const UInt batchSize, const UInt h, const UInt w)
		{
			if (batchSize < 1)
				throw std::invalid_argument("Invalid batchSize value in ChangeResolution function");
			if (h < 1)
				throw std::invalid_argument("Invalid height value in ChangeResolution function");
			if (w < 1)
				throw std::invalid_argument("Invalid width value in ChangeResolution function");

			Layers[0]->H = h;
			Layers[0]->W = w;

			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				BatchSizeChanging.store(true);

				for (auto& layer : Layers)
					layer->UpdateResolution();
				
				for (auto& layer : Layers)
					layer->SetBatchSize(batchSize);
				
				AdjustedTrainingSamplesCount = (DataProv->TrainingSamplesCount % batchSize == 0) ? DataProv->TrainingSamplesCount : ((DataProv->TrainingSamplesCount / batchSize) + 1) * batchSize;
				AdjustedTestingSamplesCount = (DataProv->TestingSamplesCount % batchSize == 0) ? DataProv->TestingSamplesCount : ((DataProv->TestingSamplesCount / batchSize) + 1) * batchSize;
				TrainSkipCount = batchSize - (AdjustedTrainingSamplesCount - DataProv->TrainingSamplesCount);
				TestSkipCount = batchSize - (AdjustedTestingSamplesCount - DataProv->TestingSamplesCount);
				TrainOverflowCount = AdjustedTrainingSamplesCount - batchSize;
				TestOverflowCount = AdjustedTestingSamplesCount - batchSize;;

				BatchSize = batchSize;

				BatchSizeChanging.store(false);
			}

			if (BatchSize == batchSize)
			{
				H = h;
				W = w;
			}
			//else
			//{
			//	// revert if unsuccessful
			//	Layers[0]->H = H;
			//	Layers[0]->W = W;

			//	for (auto& layer : Layers)
			//		layer->UpdateResolution();

			//	SetBatchSize(BatchSize);
			//}
		}

		void ChangeDropout(const Float dropout, const UInt batchSize)
		{
			if (dropout < 0 || dropout >= 1)
				throw std::invalid_argument("Invalid dropout value in ChangeDropout function");

			for (auto& layer : Layers)
			{
				switch (layer->LayerType)
				{
				case LayerTypes::BatchNormHardSwishDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer.get());
					if (bn)
					{
						bn->UpdateDropout(dropout);
						bn->SetBatchSize(batchSize);
					}
				}
				break;
				case LayerTypes::BatchNormMishDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer.get());
					if (bn)
					{
						bn->UpdateDropout(dropout);
						bn->SetBatchSize(batchSize);
					}
				}
				break;
				case LayerTypes::BatchNormReluDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer.get());
					if (bn)
					{
						bn->UpdateDropout(dropout);
						bn->SetBatchSize(batchSize);
					}
				}
				break;
				case LayerTypes::BatchNormSwishDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer.get());
					if (bn)
					{
						bn->UpdateDropout(dropout);
						bn->SetBatchSize(batchSize);
					}
				}
				break;
				case LayerTypes::BatchNormTanhExpDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer.get());
					if (bn)
					{
						bn->UpdateDropout(dropout);
						bn->SetBatchSize(batchSize);
					}
				}
				break;
				case LayerTypes::Dropout:
				{
					auto drop = dynamic_cast<dnn::Dropout*>(layer.get());
					if (drop)
					{
						drop->UpdateDropout(dropout);
						drop->SetBatchSize(batchSize);
					}
				}
				break;
				default:
					break;
				}
			}

			Dropout = dropout;
		}

		bool SetFormat(bool plain = false)
		{
			if (TaskState.load() == TaskStates::Stopped)
			{
				Format = plain ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::any;
				for (auto& layer : Layers)
					layer->Format = Format;
				
				return true;
			}
			else
			    return false;
		}

		void ResetWeights()
		{
			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				ResettingWeights.store(true);

				for (auto& layer : Layers)
				{
					while (layer->RefreshingStats.load())
						std::this_thread::sleep_for(std::chrono::milliseconds(100));

					layer->ResetWeights(WeightsFiller, WeightsFillerMode, WeightsGain, WeightsScale, BiasesFiller, BiasesFillerMode, BiasesGain, BiasesScale);
					layer->ResetOptimizer(Optimizer);
				}

				ResettingWeights.store(false);
			}
		}

		bool IsUniqueLayerName(std::string name) const
		{
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			
			for (auto& layer : Layers)
			{
				auto layerName = layer->Name;
				std::transform(layerName.begin(), layerName.end(), layerName.begin(), ::tolower);
				if (layerName == name)
					return false;
			}

			return true;
		}

		void SetLocking(const bool locked)
		{
			for (auto &layer : Layers)
				if (layer->Lockable() && !DisableLocking)
					layer->LockUpdate.store(locked);
			
			if (!DisableLocking)
			{
				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
				{
					Layers[i]->bpropTime = std::chrono::duration<Float>(0);
					Layers[i]->updateTime = std::chrono::duration<Float>(0);

					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}
				}
			}
		}

		void SetLayerLocking(const UInt layerIndex, const bool locked)
		{
			if (layerIndex < Layers.size() && Layers[layerIndex]->Lockable() && !DisableLocking)
			{
				Layers[layerIndex]->LockUpdate.store(locked);

				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
				{
					Layers[i]->bpropTime = std::chrono::duration<Float>(0);
					Layers[i]->updateTime = std::chrono::duration<Float>(0);

					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}
				}
			}
		}		
	
		auto GetWeightsSize(const bool persistOptimizer, const Optimizers optimizer) const
		{
			std::streamsize weightsSize = 0;

			for (auto &layer : Layers)
				weightsSize += layer->GetWeightsSize(persistOptimizer, optimizer);

			return weightsSize;
		}

		auto GetNeuronsSize(const UInt batchSize) const
		{
			UInt neuronsSize = 0;

			for (auto &layer : Layers)
				neuronsSize += layer->GetNeuronsSize(batchSize);

			return neuronsSize;
		}

		bool BatchNormalizationUsed() const
		{
			for (const auto &layer : Layers)
				if (layer->IsBatchNorm())
					return true;

			return false;
		}

		void AddTrainingRate(const TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples)
		{
			if (clear)
				TrainingRates.clear();

			TotalCycles = 1;
			GoToEpoch = gotoEpoch;
			
			const auto LR = rate.MaximumRate;
			const auto Epochs = rate.Epochs;

			auto decayAfterEpochs = rate.DecayAfterEpochs;
			if (rate.Epochs < decayAfterEpochs)
				decayAfterEpochs = rate.Epochs;

			auto totIteration = rate.Epochs / decayAfterEpochs;
			auto newRate = rate.MaximumRate;

			for (auto i = 0ull; i < totIteration; i++)
			{
				if (rate.Optimizer == Optimizers::AdaBoundW || rate.Optimizer == Optimizers::AdamW || rate.Optimizer == Optimizers::AmsBoundW || rate.Optimizer == Optimizers::SGDW)
				{
					const auto weightDecayMultiplier = newRate / LR;
					const auto weightDecayNormalized = rate.L2Penalty / std::pow(Float(rate.BatchSize) / (Float(trainSamples) / rate.BatchSize) * Epochs, Float(0.5));

					if ((i + 1) >= gotoEpoch)
						TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				}
				else
				{
					if ((i + 1) >= gotoEpoch)
						TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix,  rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				}

				if (newRate * rate.DecayFactor > rate.MinimumRate)
					newRate *= rate.DecayFactor;
				else
					newRate = rate.MinimumRate;
			}

			if (rate.Optimizer == Optimizers::AdaBoundW || rate.Optimizer == Optimizers::AdamW || rate.Optimizer == Optimizers::AmsBoundW || rate.Optimizer == Optimizers::SGDW)
			{
				const auto weightDecayMultiplier = newRate / LR;
				const auto weightDecayNormalized = rate.L2Penalty / std::pow(Float(rate.BatchSize) / (Float(trainSamples) / rate.BatchSize) * Epochs, Float(0.5));

				if ((totIteration * decayAfterEpochs) < rate.Epochs)
					TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
			}
			else
			{
				if ((totIteration * decayAfterEpochs) < rate.Epochs)
					TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
			}
		}

		void AddTrainingRateSGDR(const TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples)
		{
			if (clear)
				TrainingRates.clear();

			TotalCycles = rate.Cycles;
			GoToEpoch = gotoEpoch;

			const auto LR = rate.MaximumRate;
			auto maxRate = rate.MaximumRate;
			auto minRate = rate.MinimumRate;
			auto epoch = 0ull;

			for (auto c = 0ull; c < TotalCycles; c++)
			{
				const auto totalEpochs = rate.Epochs * (c > 0 ? (rate.EpochMultiplier != 1 ? c * rate.EpochMultiplier : 1) : 1);
				for (auto i = 0ull; i < totalEpochs; i++)
				{
					const auto newRate = (minRate + Float(0.5) * (maxRate - minRate) * (Float(1) + std::cos(Float(i) / Float(totalEpochs) * Float(3.1415926535897932384626433832))));
					
					epoch++;
					
					if (rate.Optimizer == Optimizers::AdaBoundW || rate.Optimizer == Optimizers::AdamW || rate.Optimizer == Optimizers::AmsBoundW || rate.Optimizer == Optimizers::SGDW)
					{
						const auto weightDecayMultiplier = newRate / LR;
						const auto weightDecayNormalized = rate.L2Penalty / std::pow(Float(rate.BatchSize) / (Float(trainSamples) / rate.BatchSize) * totalEpochs, Float(0.5));

						if (epoch >= gotoEpoch)
						{
							if (UseTrainingStrategy)
							{
								TrainingRate tmpRate;

								auto mult = Float(0);
								for (const auto& strategy : TrainingStrategies)
								{
									mult += strategy.Epochs;
									if (epoch <= static_cast<UInt>(mult * totalEpochs))
										tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);
									else
										break;
								}

								TrainingRates.push_back(tmpRate);
							}
							else
								TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, rate.Gamma, 1, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
						}
					}
					else
					{
						if (epoch >= gotoEpoch)
						{
							if (UseTrainingStrategy)
							{
								TrainingRate tmpRate;

								auto mult = Float(0);
								for (const auto& strategy : TrainingStrategies)
								{
									mult += strategy.Epochs;
									if (epoch <= static_cast<UInt>(mult * totalEpochs))
										tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);
									else
										break;
								}

								TrainingRates.push_back(tmpRate);
							}
							else
								TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, rate.Gamma, 1, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
						}
					}
				}
				
				if (rate.DecayFactor != Float(1))
				{
					maxRate *= rate.DecayFactor;
					minRate *= rate.DecayFactor;
				}
			}
		}

		bool CheckTaskState() const
		{
			while (TaskState.load() == TaskStates::Paused)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
				std::this_thread::yield();
			}

			return TaskState.load() == TaskStates::Running;
		}

		void TrainingAsync()
		{
			task = std::async(std::launch::async, [=] { Training(); });
		}

		void TestingAsync()
		{
			task = std::async(std::launch::async, [=] { Testing(); });
		}

		void StopTask()
		{
			if (TaskState.load() != TaskStates::Stopped)
			{
				TaskState.store(TaskStates::Stopped);

				if (task.valid())
					try
				    {
					    task.get();
				    }
				    catch (const std::runtime_error& e)
				    {
					    std::cout << "Async task threw exception: " << e.what() << std::endl;
				    }

				State.store(States::Completed);
			}
		}

		void PauseTask()
		{
			if (TaskState.load() == TaskStates::Running)
				TaskState.store(TaskStates::Paused);
		}

		void ResumeTask()
		{
			if (TaskState.load() == TaskStates::Paused)
				TaskState.store(TaskStates::Running);
		}

		void SetOptimizer(const Optimizers optimizer)
		{
			if (optimizer != Optimizer)
			{
				for (auto& layer : Layers)
				{
					layer->InitializeDescriptors(BatchSize);
					layer->SetOptimizer(optimizer);
				}

				Optimizer = optimizer;
			}
		}

		void ResetOptimizer()
		{
			for (auto &layer : Layers)
				layer->ResetOptimizer(Optimizer);
		}

#ifdef DNN_STOCHASTIC
		void CostFunction(const States state)
		{
			for (auto cost : CostLayers)
			{
				auto loss = Float(0);

				for (auto i = 0ull; i < cost->C; i++)
					loss += cost->Neurons[i] * cost->Weight;

				if (state == States::Training)
					cost->TrainLoss += loss;
				else
					cost->TestLoss += loss;
			}
		}

		void Recognized(const States state, const std::vector<LabelInfo>& sampleLabel)
		{
			for (auto cost : CostLayers)
			{
				const auto inputLayer = cost->InputLayer;
				const auto labelIndex = cost->LabelIndex;

				auto hotIndex = 0ull;
				auto maxValue = std::numeric_limits<Float>::lowest();
				for (auto i = 0ull; i < cost->C; i++)
				{
					if (inputLayer->Neurons[i] > maxValue)
					{
						maxValue = inputLayer->Neurons[i];
						hotIndex = i;
					}
				}

				if (hotIndex != sampleLabel[labelIndex].LabelA)
				{
					if (state == States::Training)
						cost->TrainErrors++;
					else
						cost->TestErrors++;
				}

				if (state == States::Testing)
					cost->ConfusionMatrix[hotIndex][sampleLabel[labelIndex].LabelA]++;
			}
		}
#endif

		void CostFunctionBatch(const States state, const UInt batchSize, const bool overflow, const UInt skipCount)
		{
			for (auto cost : CostLayers)
			{
				for (auto b = 0ull; b < batchSize; b++)
				{
					if (overflow && b >= skipCount)
						return;

					const auto batchOffset = b * cost->C;
					auto loss = Float(0);

					for (auto i = 0ull; i < cost->C; i++)
						loss += cost->Neurons[batchOffset + i] * cost->Weight;

					if (state == States::Training)
						cost->TrainLoss += loss;
					else
						cost->TestLoss += loss;
				}
			}
		}

		void RecognizedBatch(const States state, const UInt batchSize, const bool overflow, const UInt skipCount, const std::vector<std::vector<LabelInfo>>& sampleLabels)
		{
			for (auto cost : CostLayers)
			{
				const auto &inputLayer = cost->InputLayer;
				const auto labelIndex = cost->LabelIndex;

				for (auto b = 0ull; b < batchSize; b++)
				{
					if (overflow && b >= skipCount)
						return;

					const auto sampleOffset = b * inputLayer->C;

					auto hotIndex = 0ull;
					auto maxValue = std::numeric_limits<Float>::lowest();
					for (auto i = 0ull; i < inputLayer->C; i++)
					{
						if (inputLayer->Neurons[i + sampleOffset] > maxValue)
						{
							maxValue = inputLayer->Neurons[i + sampleOffset];
							hotIndex = i;
						}
					}

					if (hotIndex != sampleLabels[b][labelIndex].LabelA)
					{
						if (state == States::Training)
							cost->TrainErrors++;
						else
							cost->TestErrors++;
					}

					if (state == States::Testing)
						cost->ConfusionMatrix[hotIndex][sampleLabels[b][labelIndex].LabelA]++;
				}
			}
		}

		void SetBatchSize(const UInt batchSize)
		{
			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				BatchSizeChanging.store(true);

				for (auto &layer : Layers)
					layer->SetBatchSize(batchSize);

				AdjustedTrainingSamplesCount = (DataProv->TrainingSamplesCount % batchSize == 0) ? DataProv->TrainingSamplesCount : ((DataProv->TrainingSamplesCount / batchSize) + 1) * batchSize;
				AdjustedTestingSamplesCount = (DataProv->TestingSamplesCount % batchSize == 0) ? DataProv->TestingSamplesCount : ((DataProv->TestingSamplesCount / batchSize) + 1) * batchSize;
				TrainSkipCount = batchSize - (AdjustedTrainingSamplesCount - DataProv->TrainingSamplesCount);
				TestSkipCount = batchSize - (AdjustedTestingSamplesCount - DataProv->TestingSamplesCount);
				TrainOverflowCount = AdjustedTrainingSamplesCount - batchSize;
				TestOverflowCount = AdjustedTestingSamplesCount - batchSize;;

				BatchSize = batchSize;

				BatchSizeChanging.store(false);
			}
		}

		void SwitchInplaceBwd(const bool enable)
		{
			if (enable)
				for (auto& layer : Layers)
				{
					layer->Inputs = std::vector<Layer*>(layer->InputsInplaceBwd);
					layer->InputLayer = layer->InputLayerInplaceBwd;
					layer->SharesInput = layer->SharesInputInplace;
				}
			else
				for (auto& layer : Layers)
				{
					layer->Inputs = std::vector<Layer*>(layer->InputsOriginal);
					layer->InputLayer = layer->InputLayerOriginal;
					layer->SharesInput = layer->SharesInputOriginal;
				}
		}

		std::vector<Layer*> GetLayerInputs(const std::vector<std::string>& inputs) const
		{
			auto list = std::vector<Layer*>();

			for (auto& name : inputs)
			{
				auto exists = false;

				for (auto& layer : Layers)
				{
					if (layer->Name == name)
					{
						list.push_back(layer.get());
						exists = true;
					}
				}

				if (!exists)
					throw std::invalid_argument(std::string("Invalid input layer: " + name).c_str());
			}

			return list;
		}

		std::vector<Layer*> GetLayerOutputs(const Layer* parentLayer, const bool inplace = false) const
		{
			auto outputs = std::vector<Layer*>();

			for (auto& layer : Layers)
				if (layer->Name != parentLayer->Name)
					for (auto input : inplace ? layer->InputsInplaceBwd : layer->InputsOriginal)
						if (input->Name == parentLayer->Name)
							outputs.push_back(layer.get());

			return outputs;
		}

		std::vector<Layer*> SetRelations()
		{
			// This determines how the backprop step correctly flows
			// When SharesInput is true we have to add our diff vector instead of just copying it because there's more than one layer involved

			// determine SharesInputOriginal and SharesInput
			for (auto& layer : Layers)
				layer->SharesInput = false;
			
			auto unreferencedLayers = std::vector<Layer*>();

			for (auto& layer : Layers)
			{
				auto outputsCount = GetLayerOutputs(layer.get()).size();

				if (outputsCount > 1)
				{
					for (auto& l : Layers)
					{
						if (l->Name == layer->Name)
							continue;

						for (auto input : l->InputsOriginal)
						{
							if (input->Name == layer->Name)
							{
								l->SharesInput = l->InplaceBwd ? false : true;
								l->SharesInputOriginal = l->InplaceBwd ? false : true;
								outputsCount--;
								break;
							}
						}
						
						if (outputsCount == 1)
							break;
					}
				}
				else
				{
					if (outputsCount == 0 && layer->LayerType != LayerTypes::Cost)
						unreferencedLayers.push_back(layer.get());
				}
			}

			// determine SharesInputInplace
			for (auto& layer : Layers)
			{
				auto outputsCount = GetLayerOutputs(layer.get(), true).size();

				if (outputsCount > 1)
				{
					for (auto& l : Layers)
					{
						if (l->Name == layer->Name)
							continue;

						for (auto input : l->InputsInplaceBwd)
						{
							if (input->Name == layer->Name)
							{
								l->SharesInputInplace = l->InplaceBwd ? false : true;
								outputsCount--;
								break;
							}
						}

						if (outputsCount == 1)
							break;
					}
				}
			}

			return unreferencedLayers;
		}
	
		void Training()
		{
			if (TaskState.load() == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);

				auto timer = std::chrono::high_resolution_clock();
				auto timePoint = timer.now();
				auto timePointGlobal = timer.now();
				auto bpropTimeCount = std::chrono::duration<Float>(Float(0));
				auto updateTimeCount = std::chrono::duration<Float>(Float(0));
                auto elapsedTime = std::chrono::duration<Float>(Float(0));

				TotalEpochs = 0;
				for (const auto& rate : TrainingRates)
					TotalEpochs += rate.Epochs;
				TotalEpochs += GoToEpoch - 1;

				auto useCycli = false;
				for (const auto& rate : TrainingRates)
					if (rate.Cycles != 1)
						useCycli = true;

				CurrentEpoch = GoToEpoch - 1;
				CurrentTrainingRate = TrainingRates[0];
				Rate = CurrentTrainingRate.MaximumRate;
				CurrentCycle = CurrentTrainingRate.Cycles;
			
				if (CurrentTrainingRate.BatchSize > BatchSize)
					if (GetTotalFreeMemory() < GetNeuronsSize(CurrentTrainingRate.BatchSize - BatchSize))
					{                           
						std::cout << std::string("Memory required: ") << std::to_string(GetNeuronsSize(CurrentTrainingRate.BatchSize - BatchSize) / 1024 / 1024) << " MB with BatchSize " << std::to_string(CurrentTrainingRate.BatchSize) << std::endl << std::endl;
						State.store(States::Completed);
						return;
					}
				std::cout << std::string("Memory required: ") << std::to_string(GetNeuronsSize(CurrentTrainingRate.BatchSize - BatchSize) / 1024 / 1024) << " MB with BatchSize " << std::to_string(CurrentTrainingRate.BatchSize) << std::endl << std::endl;
				ChangeResolution(CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width);

				auto learningRateEpochs = CurrentTrainingRate.Epochs;
				auto learningRateIndex = 0ull;

				RandomTrainingSamples = std::vector<UInt>(DataProv->TrainingSamplesCount);
				for (auto i = 0ull; i < DataProv->TrainingSamplesCount; i++)
					RandomTrainingSamples[i] = i;

				TrainingSamplesHFlip = std::vector<bool>();
				TrainingSamplesVFlip = std::vector<bool>();
				TestingSamplesHFlip = std::vector<bool>();
				TestingSamplesVFlip = std::vector<bool>();

				for (auto index = 0ull; index < DataProv->TrainingSamplesCount; index++)
				{
					TrainingSamplesHFlip.push_back(Bernoulli<bool>());
					TrainingSamplesVFlip.push_back(Bernoulli<bool>());
				}
				for (auto index = 0ull; index < DataProv->TestingSamplesCount; index++)
				{
					TestingSamplesHFlip.push_back(Bernoulli<bool>());
					TestingSamplesVFlip.push_back(Bernoulli<bool>());
				}
				
				SetOptimizer(CurrentTrainingRate.Optimizer);
				if (!PersistOptimizer)
					for (auto& layer : Layers)
						layer->ResetOptimizer(Optimizer);
				else
					for (auto& layer : Layers)
						layer->CheckOptimizer(Optimizer);

				FirstUnlockedLayer.store(Layers.size() - 2);
				for (auto i = 0ull; i < Layers.size(); i++)
					if (Layers[i]->Lockable() && !Layers[i]->LockUpdate.load())
					{
						FirstUnlockedLayer.store(i);
						break;
					}

				
				while (CurrentEpoch < TotalEpochs)
				{
					if (CurrentEpoch - (GoToEpoch - 1) == learningRateEpochs)
					{
						learningRateIndex++;
						CurrentTrainingRate = TrainingRates[learningRateIndex];
						Rate = CurrentTrainingRate.MaximumRate;
						
						if (H != CurrentTrainingRate.Height || W != CurrentTrainingRate.Width || BatchSize != CurrentTrainingRate.BatchSize)
							ChangeResolution(CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width);
								
						if (Dropout != CurrentTrainingRate.Dropout)
							ChangeDropout(CurrentTrainingRate.Dropout, BatchSize);

						learningRateEpochs += CurrentTrainingRate.Epochs;

						if (CurrentTrainingRate.Optimizer != Optimizer)
						{
							SetOptimizer(CurrentTrainingRate.Optimizer);
							if (!PersistOptimizer)
								for (auto& layer : Layers)
									layer->ResetOptimizer(Optimizer);
							else
								for (auto& layer : Layers)
									layer->CheckOptimizer(Optimizer);
						}
					}

					CurrentEpoch++;
					CurrentCycle = CurrentTrainingRate.Cycles;

					if (CurrentTrainingRate.HorizontalFlip)
						for (auto index = 0ull; index < DataProv->TrainingSamplesCount; index++)
							TrainingSamplesHFlip[index].flip();

					if (CurrentTrainingRate.VerticalFlip)
						for (auto index = 0ull; index < DataProv->TrainingSamplesCount; index++)
							TrainingSamplesVFlip[index].flip();

					if (CheckTaskState())
					{
						State.store(States::Training);

						const auto shuffleCount = UniformInt<UInt>(16, 32);
						for (auto shuffle = 0ull; shuffle < shuffleCount; shuffle++)
							std::shuffle(std::begin(RandomTrainingSamples), std::end(RandomTrainingSamples), std::mt19937(Seed<unsigned>()));

						for (auto cost : CostLayers)
							cost->Reset();

#ifdef DNN_STOCHASTIC				
						if (BatchSize == 1)
						{
							for (SampleIndex = 0; SampleIndex < DataProv->TrainingSamplesCount; SampleIndex++)
							{
								// Forward
								timePointGlobal = timer.now();
								auto SampleLabel = TrainSample(SampleIndex);
								Layers[0]->fpropTime = timer.now() - timePointGlobal;

								for (auto cost : CostLayers)
									cost->SetSampleLabel(SampleLabel);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									timePoint = timer.now();
									Layers[i]->ForwardProp(1, true);
									Layers[i]->fpropTime = timer.now() - timePoint;
								}

								CostFunction(State.load());
								Recognized(State.load(), SampleLabel);
								fpropTime = timer.now() - timePointGlobal;

								// Backward
								bpropTimeCount = std::chrono::duration<Float>(Float(0));
								updateTimeCount = std::chrono::duration<Float>(Float(0));
								if (UseInplace)
									SwitchInplaceBwd(true);
								for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
								{
									if (Layers[i]->HasWeights)
									{
										timePoint = timer.now();
										Layers[i]->ResetGradients();
										Layers[i]->BackwardProp(BatchSize);
										Layers[i]->bpropTime = timer.now() - timePoint;
										timePoint = timer.now();
										Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
										Layers[i]->updateTime = timer.now() - timePoint;
										updateTimeCount += Layers[i]->updateTime;
									}
									else
									{
										timePoint = timer.now();
										Layers[i]->BackwardProp(1);
										Layers[i]->bpropTime = timer.now() - timePoint;
									}
									bpropTimeCount += Layers[i]->bpropTime;
								}
								if (UseInplace)
									SwitchInplaceBwd(false);
								bpropTime = bpropTimeCount;
								updateTime = updateTimeCount;

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
						}
						else
						{
#endif
							auto overflow = false;
							for (SampleIndex = 0; SampleIndex < AdjustedTrainingSamplesCount; SampleIndex += BatchSize)
							{
								// Forward
								timePointGlobal = timer.now();
								auto SampleLabels = TrainBatch(SampleIndex, BatchSize);
								Layers[0]->fpropTime = timer.now() - timePointGlobal;

								for (auto cost : CostLayers)
									cost->SetSampleLabels(SampleLabels);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									timePoint = timer.now();
									Layers[i]->ForwardProp(BatchSize, true);
									Layers[i]->fpropTime = timer.now() - timePoint;
								}
								
								overflow = SampleIndex >= TrainOverflowCount;
								CostFunctionBatch(State.load(), BatchSize, overflow, TrainSkipCount);
								RecognizedBatch(State.load(), BatchSize, overflow, TrainSkipCount, SampleLabels);
								fpropTime = timer.now() - timePointGlobal;

								// Backward
								bpropTimeCount = std::chrono::duration<Float>(Float(0));
								updateTimeCount = std::chrono::duration<Float>(Float(0));
								if (UseInplace)
									SwitchInplaceBwd(true);
								for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
								{
									if (Layers[i]->HasWeights)
									{
										timePoint = timer.now();
										Layers[i]->ResetGradients();
										Layers[i]->BackwardProp(BatchSize);
										Layers[i]->bpropTime = timer.now() - timePoint;
										timePoint = timer.now();
										Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
										Layers[i]->updateTime = timer.now() - timePoint;
										updateTimeCount += Layers[i]->updateTime;
									}
									else
									{
										timePoint = timer.now();
										Layers[i]->BackwardProp(BatchSize);
										Layers[i]->bpropTime = timer.now() - timePoint;
									}
									bpropTimeCount += Layers[i]->bpropTime;
								}
								if (UseInplace)
									SwitchInplaceBwd(false);
								bpropTime = bpropTimeCount;
								updateTime = updateTimeCount;

								elapsedTime = timer.now() - timePointGlobal;
								SampleSpeed = BatchSize / (Float(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count()) / 1000000);

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
#ifdef DNN_STOCHASTIC
						}
#endif
					}
					else
						break;

					if (CheckTaskState())
					{
						State.store(States::Testing);
#ifdef DNN_STOCHASTIC	

						
						if (BatchSize == 1)
						{
							for (SampleIndex = 0; SampleIndex < DataProv->TestingSamplesCount; SampleIndex++)
							{
								auto SampleLabel = TestSample(SampleIndex);

								for (auto cost : CostLayers)
									cost->SetSampleLabel(SampleLabel);

								for (auto i = 1u; i < Layers.size(); i++)
									Layers[i]->ForwardProp(1, false);

								CostFunction(State.load());
								Recognized(State.load(), SampleLabel);

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
						}
						else
						{
#endif
							auto overflow = false;
							for (SampleIndex = 0; SampleIndex < AdjustedTestingSamplesCount; SampleIndex += BatchSize)
							{
								timePointGlobal = timer.now();

								timePoint = timer.now();
								auto SampleLabels = TestBatch(SampleIndex, BatchSize);
								Layers[0]->fpropTime = timer.now() - timePoint;

								for (auto cost : CostLayers)
									cost->SetSampleLabels(SampleLabels);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									timePoint = timer.now();
									Layers[i]->ForwardProp(BatchSize, false);
									Layers[i]->fpropTime = timer.now() - timePoint;
								}

								fpropTime = timer.now() - timePointGlobal;

								overflow = SampleIndex >= TestOverflowCount;
								CostFunctionBatch(State.load(), BatchSize, overflow, TestSkipCount);
								RecognizedBatch(State.load(), BatchSize, overflow, TestSkipCount, SampleLabels);

								elapsedTime = timer.now() - timePointGlobal;
								SampleSpeed = BatchSize / (Float(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count()) / 1000000);

								if (TaskState.load() != TaskStates::Running && !CheckTaskState())
									break;
							}
#ifdef DNN_STOCHASTIC
						}
#endif
						if (CheckTaskState())
						{
							for (auto cost : CostLayers)
							{
								cost->AvgTrainLoss = cost->TrainLoss / DataProv->TrainingSamplesCount;
								cost->AvgTestLoss = cost->TestLoss / DataProv->TestingSamplesCount;
								cost->TrainErrorPercentage = cost->TrainErrors / Float(DataProv->TrainingSamplesCount / 100);
								cost->TestErrorPercentage = cost->TestErrors / Float(DataProv->TestingSamplesCount / 100);
							}

							TrainLoss = CostLayers[CostIndex]->TrainLoss;
							TrainErrors = CostLayers[CostIndex]->TrainErrors;
							AvgTrainLoss = CostLayers[CostIndex]->AvgTrainLoss;
							TrainErrorPercentage = CostLayers[CostIndex]->TrainErrorPercentage;
							TestLoss = CostLayers[CostIndex]->TestLoss;
							TestErrors = CostLayers[CostIndex]->TestErrors;
							AvgTestLoss = CostLayers[CostIndex]->AvgTestLoss;
							TestErrorPercentage = CostLayers[CostIndex]->TestErrorPercentage;
							Accuracy = Float(100) - TestErrorPercentage;

							// save the weights
							State.store(States::SaveWeights);
							auto directory = DataProv->StorageDirectory / std::string("definitions") / Name;
							auto fileName = (directory / (std::to_string(CurrentEpoch) + std::string("-") + std::to_string(CurrentCycle) + std::string("-(") + StringToLower(std::string(magic_enum::enum_name<Datasets>(Dataset))) + std::string(")(") + StringToLower(std::string(magic_enum::enum_name<Optimizers>(Optimizer))) + std::string(")-") + std::to_string(TrainErrors) + std::string("-") + std::to_string(TestErrors) + std::string(".bin"))).string();
							
							std::filesystem::create_directories(directory);
							SaveWeights(fileName, PersistOptimizer);
							
							State.store(States::NewEpoch);

							NewEpoch(CurrentCycle, CurrentEpoch, TotalEpochs, static_cast<UInt>(CurrentTrainingRate.Optimizer), CurrentTrainingRate.Beta2, CurrentTrainingRate.Gamma, CurrentTrainingRate.Eps, CurrentTrainingRate.HorizontalFlip, CurrentTrainingRate.VerticalFlip, CurrentTrainingRate.InputDropout, CurrentTrainingRate.Cutout, CurrentTrainingRate.CutMix, CurrentTrainingRate.AutoAugment, CurrentTrainingRate.ColorCast, CurrentTrainingRate.ColorAngle, CurrentTrainingRate.Distortion, static_cast<UInt>(CurrentTrainingRate.Interpolation), CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, CurrentTrainingRate.MaximumRate, CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width, CurrentTrainingRate.Momentum, CurrentTrainingRate.L2Penalty, CurrentTrainingRate.Dropout, AvgTrainLoss, TrainErrorPercentage, Float(100) - TrainErrorPercentage, TrainErrors, AvgTestLoss, TestErrorPercentage, Float(100) - TestErrorPercentage, TestErrors);
						}
						else
							break;
					}
					else
						break;
				}

				State.store(States::Completed);
			}
		}

		void Testing()
		{
			if (TaskState == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);

				State.store(States::Idle);

				auto timer = std::chrono::high_resolution_clock();
				auto timePoint = timer.now();
				auto timePointGlobal = timer.now();
				auto elapsedTime = std::chrono::duration<Float>(Float(0));

				CurrentTrainingRate = TrainingRates[0];
				// check first if we have enough memory available
				if (CurrentTrainingRate.BatchSize > BatchSize)
					if (GetTotalFreeMemory() < GetNeuronsSize(CurrentTrainingRate.BatchSize - BatchSize))
					{
						State.store(States::Completed);
						return;
					}
				SetBatchSize(CurrentTrainingRate.BatchSize);
				Rate = CurrentTrainingRate.MaximumRate;

				TrainingSamplesHFlip = std::vector<bool>();
				TrainingSamplesVFlip = std::vector<bool>();
				TestingSamplesHFlip = std::vector<bool>();
				TestingSamplesVFlip = std::vector<bool>();
				for (auto index = 0ull; index < DataProv->TrainingSamplesCount; index++)
				{
					TrainingSamplesHFlip.push_back(Bernoulli<bool>());
					TrainingSamplesVFlip.push_back(Bernoulli<bool>());
				}
				for (auto index = 0ull; index < DataProv->TestingSamplesCount; index++)
				{
					TestingSamplesHFlip.push_back(Bernoulli<bool>());
					TestingSamplesVFlip.push_back(Bernoulli<bool>());
				}

				State.store(States::Testing);

				for (auto cost : CostLayers)
					cost->Reset();

#ifdef DNN_STOCHASTIC
				if (BatchSize == 1)
				{
					for (SampleIndex = 0; SampleIndex < DataProv->TestingSamplesCount; SampleIndex++)
					{
						auto SampleLabel = TestAugmentedSample(SampleIndex);

						for (auto cost : CostLayers)
							cost->SetSampleLabel(SampleLabel);

						for (auto i = 1ull; i < Layers.size(); i++)
							Layers[i]->ForwardProp(1, false);

						CostFunction(State.load());
						Recognized(State.load(), SampleLabel);

						if (TaskState.load() != TaskStates::Running && !CheckTaskState())
							break;
					}
				}
				else
				{
#else
				auto overflow = false;
				for (SampleIndex = 0; SampleIndex < AdjustedTestingSamplesCount; SampleIndex += BatchSize)
				{
					timePointGlobal = timer.now();

					auto SampleLabels = TestAugmentedBatch(SampleIndex, BatchSize);
					Layers[0]->fpropTime = timer.now() - timePointGlobal;

					for (auto cost : CostLayers)
						cost->SetSampleLabels(SampleLabels);

					for (auto i = 1ull; i < Layers.size(); i++)
					{
						timePoint = timer.now();
						Layers[i]->ForwardProp(BatchSize, false);
						Layers[i]->fpropTime = timer.now() - timePoint;
					}

					overflow = SampleIndex >= TestOverflowCount;
					CostFunctionBatch(State.load(), BatchSize, overflow, TestSkipCount);
					RecognizedBatch(State.load(), BatchSize, overflow, TestSkipCount, SampleLabels);

					fpropTime = timer.now() - timePointGlobal;
					
					SampleSpeed = BatchSize / (Float(std::chrono::duration_cast<std::chrono::microseconds>(fpropTime).count()) / 1000000);
					
					if (TaskState.load() != TaskStates::Running && !CheckTaskState())
						break;
				}
#endif
#ifdef DNN_STOCHASTIC
				}
#endif
			for (auto cost : CostLayers)
			{
				cost->AvgTestLoss = cost->TestLoss / DataProv->TestingSamplesCount;
				cost->TestErrorPercentage = cost->TestErrors / Float(DataProv->TestingSamplesCount / 100);
			}

			TestLoss = CostLayers[CostIndex]->TestLoss;
			AvgTestLoss = CostLayers[CostIndex]->AvgTestLoss;
			TestErrors = CostLayers[CostIndex]->TestErrors;
			TestErrorPercentage = CostLayers[CostIndex]->TestErrorPercentage;
			Accuracy = Float(100) - TestErrorPercentage;

			/*
			auto fileName = std::string("C:\\test.txt");
			auto ofs = std::ofstream(fileName);
			if (!ofs.bad())
			{
				for (auto i = 0ull; i < 10000ull; i++)
					ofs << labels[i] << std::endl;
				ofs.flush();
				ofs.close();
			}
			*/

			State.store(States::Completed);;
			}
		}

		bool GetInputSnapShot(std::vector<Float>* snapshot, std::vector<UInt>* label)
		{
			if (!Layers[0]->Neurons.empty() && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				if (State.load() == States::Training && SampleIndex < DataProv->TrainingSamplesCount)
				{
					*label = DataProv->TrainingLabels[RandomTrainingSamples[SampleIndex]];

					for (auto i = 0ull; i < Layers[0]->CDHW; i++)
						(*snapshot)[i] = Layers[0]->Neurons[i];

					return true;
				}
				else if (State.load() == States::Testing && SampleIndex < DataProv->TestingSamplesCount)
				{
					*label = DataProv->TestingLabels[SampleIndex];

					for (auto i = 0ull; i < Layers[0]->CDHW; i++)
						(*snapshot)[i] = Layers[0]->Neurons[i];

					return true;
				}
			}

			return false;
		}

		std::vector<LabelInfo> GetLabelInfo(std::vector<UInt> labels)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<LabelInfo>(hierarchies);

			for (auto hierarchie = 0ull; hierarchie < hierarchies; hierarchie++)
			{
				SampleLabels[hierarchie].LabelA = labels[hierarchie];
				SampleLabels[hierarchie].LabelB = labels[hierarchie];
				SampleLabels[hierarchie].Lambda = Float(1);				
			}

			return SampleLabels;
		}

		std::vector<LabelInfo> GetCutMixLabelInfo(std::vector<UInt> labels, std::vector<UInt> mixLabels, double lambda)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<LabelInfo>(hierarchies);

			for (auto hierarchie = 0ull; hierarchie < hierarchies; hierarchie++)
			{
				SampleLabels[hierarchie].LabelA = labels[hierarchie];
				SampleLabels[hierarchie].LabelB = mixLabels[hierarchie];
				SampleLabels[hierarchie].Lambda = Float(lambda);
			}

			return SampleLabels;
		}

#ifdef DNN_STOCHASTIC
		std::vector<LabelInfo> TrainSample(const UInt index)
		{
			const auto rndIndex = RandomTrainingSamples[index];
			auto imgByte = DataProv->TrainingSamples[rndIndex];

			const auto rndIndexMix = (index + 1 >= DataProv->TrainingSamplesCount) ? RandomTrainingSamples[1] : RandomTrainingSamples[index + 1];
			auto imgByteMix = Image<Byte>(DataProv->TrainingSamples[rndIndexMix]);

			auto label = DataProv->TrainingLabels[rndIndex];
			auto labelMix = DataProv->TrainingLabels[rndIndexMix];
			
			std::vector<LabelInfo> SampleLabel;
		
			auto cutout = false;
			if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
			{
				if (CurrentTrainingRate.CutMix)
				{
					double lambda = BetaDistribution<double>(1, 1);
					imgByte = Image<Byte>::RandomCutMix(imgByte, imgByteMix, &lambda);
					SampleLabel = GetCutMixLabelInfo(label, labelMix, lambda);
				}
				else
				{
					SampleLabel = GetLabelInfo(label);
					cutout = true;
				}
			}
			else
				SampleLabel = GetLabelInfo(label);

			if (CurrentTrainingRate.HorizontalFlip && TrainingSamplesHFlip[rndIndex])
				imgByte = Image<Byte>::HorizontalMirror(imgByte);

			if (CurrentTrainingRate.VerticalFlip && TrainingSamplesVFlip[rndIndex])
				imgByte = Image<Byte>::VerticalMirror(imgByte);

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

			if (imgByteMix.D != D || imgByte.H != H || imgByte.W != W)
				imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (cutout)
				imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

			if (CurrentTrainingRate.Dropout > Float(0))
				imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.Dropout, DataProv->Mean);

			if (RandomCrop)
				imgByte = Image<Byte>::RandomCrop(imgByte, D, H, W, DataProv->Mean);

			for (auto c = 0ull; c < imgByte.C; c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

				for (auto d = 0ull; d < imgByte.D; d++)
					for (auto h = 0ull; h < imgByte.H; h++)
						for (auto w = 0ull; w < imgByte.W; w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<LabelInfo> TestSample(const UInt index)
		{
			auto label = DataProv->TestingLabels[index];
			auto SampleLabel = GetLabelInfo(label);

			auto imgByte = DataProv->TestingSamples[index];

			if (imgByte.D != D || imgByte.H != H || imgByte.W != W)
				imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

			imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (RandomCrop)
				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

			for (auto c = 0ull; c < imgByte.C; c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

				for (auto d = 0ull; d < imgByte.D; d++)
					for (auto h = 0ull; h < imgByte.H; h++)
						for (auto w = 0ull; w < imgByte.W; w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<LabelInfo> TestAugmentedSample(const UInt index)
		{
			auto label = DataProv->TestingLabels[index];
			auto SampleLabel = GetLabelInfo(label);

			auto imgByte = DataProv->TestingSamples[index];

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

			if (CurrentTrainingRate.HorizontalFlip && TestingSamplesHFlip[index])
				imgByte = Image<Byte>::HorizontalMirror(imgByte);

			if (CurrentTrainingRate.VerticalFlip && TestingSamplesVFlip[index])
				imgByte = Image<Byte>::VerticalMirror(imgByte);

			if (imgByte.D != D || imgByte.H != H || imgByte.W != W)
				imgByte = Image<Byte>::Resize(imgByte, D, H, W, static_cast<Interpolations>(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, static_cast<Interpolations>(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (Bernoulli<bool>(CurrentTrainingRate.Cutout) && !CurrentTrainingRate.CutMix)
				imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

			if (RandomCrop)
				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

			if (CurrentTrainingRate.Dropout > Float(0))
				imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.Dropout, DataProv->Mean);

			for (auto c = 0ull; c < imgByte.C; c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

				for (auto d = 0ull; d < imgByte.D; d++)
					for (auto h = 0ull; h < imgByte.H; h++)
						for (auto w = 0ull; w < imgByte.W; w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}
#endif

		std::vector<std::vector<LabelInfo>> TrainBatch(const UInt index, const UInt batchSize)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(hierarchies));
			const auto resize = DataProv->TrainingSamples[0].D != D || DataProv->TrainingSamples[0].H != H || DataProv->TrainingSamples[0].W != W;
			
			for_i(batchSize, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto randomIndex = (index + batchIndex >= DataProv->TrainingSamplesCount) ? RandomTrainingSamples[batchIndex] : RandomTrainingSamples[index + batchIndex];
				auto imgByte = Image<Byte>(DataProv->TrainingSamples[randomIndex]);

				const auto randomIndexMix = (index + batchSize - (batchIndex + 1) >= DataProv->TrainingSamplesCount) ? RandomTrainingSamples[batchSize - (batchIndex + 1)] : RandomTrainingSamples[index + batchSize - (batchIndex + 1)];
				auto imgByteMix = Image<Byte>(DataProv->TrainingSamples[randomIndexMix]);

				auto labels = DataProv->TrainingLabels[randomIndex];
				auto mixLabels = DataProv->TrainingLabels[randomIndexMix];
				
				auto cutout = false;
				if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
				{
					if (CurrentTrainingRate.CutMix)
					{
						double lambda = BetaDistribution<double>(1, 1);
						imgByte = Image<Byte>::RandomCutMix(imgByte, imgByteMix, &lambda);
						SampleLabels[batchIndex] = GetCutMixLabelInfo(labels, mixLabels, lambda);
					}
					else
					{
						SampleLabels[batchIndex] = GetLabelInfo(labels);
						cutout = true;
					}
				}
				else
					SampleLabels[batchIndex] = GetLabelInfo(labels);
				
				if (CurrentTrainingRate.HorizontalFlip && TrainingSamplesHFlip[randomIndex])
					imgByte = Image<Byte>::HorizontalMirror(imgByte);

				if (CurrentTrainingRate.VerticalFlip && TrainingSamplesVFlip[randomIndex])
					imgByte = Image<Byte>::VerticalMirror(imgByte);

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (cutout)
					imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

				if (RandomCrop)
					imgByte = Image<Byte>::RandomCrop(imgByte, D, H, W, DataProv->Mean);

				if (CurrentTrainingRate.Dropout > Float(0))
					imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.Dropout, DataProv->Mean);
				
				for (auto c = 0ull; c < imgByte.C; c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddevRecip = Float(1) / (MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c));

					for (auto d = 0ull; d < imgByte.D; d++)
						for (auto h = 0ull; h < imgByte.H; h++)
							for (auto w = 0ull; w < imgByte.W; w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W) + w] = (imgByte(c, d, h, w) - mean) * stddevRecip;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TestBatch(const UInt index, const UInt batchSize)
		{
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(DataProv->Hierarchies));
			const auto resize = DataProv->TestingSamples[0].D != D || DataProv->TestingSamples[0].H != H || DataProv->TestingSamples[0].W != W;

			for_i(batchSize, MEDIUM_COMPUTE, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestingSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = DataProv->TestingLabels[sampleIndex];
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = DataProv->TestingSamples[sampleIndex];

				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				for (auto c = 0ull; c < imgByte.C; c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddevRecip = Float(1) / (MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c));

					for (auto d = 0ull; d < imgByte.D; d++)
						for (auto h = 0ull; h < imgByte.H; h++)
							for (auto w = 0ull; w < imgByte.W; w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W) + w] = (imgByte(c, d, h, w) - mean) * stddevRecip;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TestAugmentedBatch(const UInt index, const UInt batchSize)
		{
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(DataProv->Hierarchies));
			const auto resize = DataProv->TestingSamples[0].D != D || DataProv->TestingSamples[0].H != H || DataProv->TestingSamples[0].W != W;

			for_i(batchSize, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestingSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = DataProv->TestingLabels[sampleIndex];
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = DataProv->TestingSamples[sampleIndex];

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					imgByte = Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

				if (CurrentTrainingRate.HorizontalFlip && TestingSamplesHFlip[sampleIndex])
					imgByte = Image<Byte>::HorizontalMirror(imgByte);

				if (CurrentTrainingRate.VerticalFlip && TestingSamplesVFlip[sampleIndex])
					imgByte = Image<Byte>::VerticalMirror(imgByte);

				if (resize)
					imgByte = Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (Bernoulli<bool>(CurrentTrainingRate.Cutout) && !CurrentTrainingRate.CutMix)
					imgByte = Image<Byte>::RandomCutout(imgByte, DataProv->Mean);
				
				if (RandomCrop)
					imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				if (CurrentTrainingRate.Dropout > Float(0))
					imgByte = Image<Byte>::Dropout(imgByte, CurrentTrainingRate.Dropout, DataProv->Mean);

				for (auto c = 0ull; c < imgByte.C; c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(imgByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(imgByte, c);

					for (auto d = 0ull; d < imgByte.D; d++)
						for (auto h = 0ull; h < imgByte.H; h++)
							for (auto w = 0ull; w < imgByte.W; w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}
			
		void ForwardProp(const UInt batchSize)
		{
			for (auto &layer : Layers)
				layer->ForwardProp(batchSize, State.load() == States::Training);
		}

		void BackwardProp(const UInt batchSize)
		{
			if (UseInplace)
				SwitchInplaceBwd(true);

			for (auto i = Layers.size() - 1; i > 0ull; --i)
			{
				if (Layers[i]->HasWeights)
				{
					Layers[i]->ResetGradients();
					Layers[i]->BackwardProp(batchSize);
					if (!DisableLocking)
						Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
				}
				else
					Layers[i]->BackwardProp(batchSize);
			}

			if (UseInplace)
				SwitchInplaceBwd(false);
		}
		
		int SaveWeights(std::string fileName, const bool persistOptimizer = false) const
		{
			auto os = std::ofstream(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

			if (!os.bad() && os.is_open())
			{
				for (auto& layer : Layers)
					layer->Save(os, persistOptimizer, Optimizer);

				os.close();

				return 0;
			}

			return -1;
		}

		int LoadWeights(std::string fileName, const bool persistOptimizer = false)
		{
			const auto optimizer = GetOptimizerFromString(fileName);

			if (GetFileSize(fileName) == GetWeightsSize(persistOptimizer, optimizer))
			{
				SetOptimizer(optimizer);

				auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

				if (!is.bad() && is.is_open())
				{
					for (auto& layer : Layers)
						layer->Load(is, persistOptimizer, Optimizer);

					is.close();

					return 0;
				}
			}

			return -1;
		}

		int SaveLayerWeights(std::string fileName, const UInt layerIndex, const bool persistOptimizer = false) const
		{
			auto os = std::ofstream(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

			if (!os.bad() && os.is_open())
			{
				Layers[layerIndex]->Save(os, persistOptimizer, Optimizer);

				os.close();

				return 0;
			}

			return -1;
		}

		int LoadLayerWeights(std::string fileName, const UInt layerIndex, const bool persistOptimizer = false)
		{
			if (GetFileSize(fileName) == Layers[layerIndex]->GetWeightsSize(persistOptimizer, Optimizer))
			{
				auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

				if (!is.bad() && is.is_open())
				{
					Layers[layerIndex]->Load(is, persistOptimizer, Optimizer);

					is.close();

					return 0;
				}
			}

			return -1;
		}

		Optimizers GetOptimizerFromString(std::string fileName) const
		{
			const auto& optimizers = magic_enum::enum_entries<Optimizers>();
			for (const auto& optimizer : optimizers)
			{
				const auto& optimizerString = std::string("(") + StringToLower(std::string(optimizer.second)) + std::string(")");
				if (fileName.find(optimizerString) != std::string::npos)
					return optimizer.first;
			}

			return Optimizers::SGD;
		}
	};
}