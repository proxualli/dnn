#pragma once
#include "Dataprovider.h"
#include "Layer.h"
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
#include "LocalResponseNormalization.h"
#include "Max.h"
#include "MaxPooling.h"
#include "Min.h"
#include "Multiply.h"
#include "PartialDepthwiseConvolution.h"
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
		size_t Row;
		size_t Column;
		bool Error;
		std::string Message;
		
		CheckMsg(const size_t row = 0, const size_t column = 0, const std::string& message = "", const bool error = true) :
			Row(row),
			Column(column),
			Message(message),
			Error(error)
		{
		}
	};

	class Model
	{
	private:
		std::future<void> task;
		std::vector<size_t> RandomTrainingSamples;
		std::vector<bool> TrainingSamplesHFlip;
		std::vector<bool> TrainingSamplesVFlip;
		std::vector<bool> TestingSamplesHFlip;
		std::vector<bool> TestingSamplesVFlip;

	public:
		const std::string Name;
		const dnnl::engine Engine;
		dnn::Device Device;
		dnnl::memory::format_tag Format;
		Dataprovider* DataProv;
		Datasets Dataset;
		std::atomic<States> State;
		std::atomic<TaskStates> TaskState;
		Costs CostFuction;
		Optimizers Optimizer;
		size_t CostIndex;
		size_t LabelIndex;
		size_t GroupIndex;
		size_t TotalCycles;
		size_t TotalEpochs;
		size_t CurrentCycle;
		size_t CurrentEpoch;
		size_t SampleIndex;
		//size_t LogInterval;
		size_t BatchSize;
		size_t GoToEpoch;
		size_t AdjustedTrainingSamplesCount;
		size_t AdjustedTestingSamplesCount;
		size_t TrainSkipCount;
		size_t TestSkipCount;
		size_t TrainOverflowCount;
		size_t TestOverflowCount;
		size_t SampleC;
		size_t SampleD;
		size_t SampleH;
		size_t SampleW;
		size_t PadD;
		size_t PadH;
		size_t PadW;
		bool MirrorPad;
		bool RandomCrop;
		bool MeanStdNormalization;
		Fillers WeightsFiller;
		Float WeightsScale;
		Float WeightsLRM;
		Float WeightsWDM;
		Fillers BiasesFiller;
		Float BiasesScale;
		Float BiasesLRM;
		Float BiasesWDM;
		Float AlphaFiller;
		Float BetaFiller;
		Float BatchNormMomentum;
		Float BatchNormEps;
		Float Dropout;
		size_t TrainErrors;
		size_t TestErrors;
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
		std::vector<std::shared_ptr<Layer>> Layers;
		std::vector<Cost*> CostLayers;
		std::vector<TrainingRate> TrainingRates;
		std::chrono::duration<Float> fpropTime;
		std::chrono::duration<Float> bpropTime;
		std::chrono::duration<Float> updateTime;
		std::atomic<size_t> FirstUnlockedLayer;
		std::atomic<bool> BatchSizeChanging;
		std::atomic<bool> ResettingWeights;

		void(*NewEpoch)(size_t, size_t, size_t, bool, bool, Float, Float, Float, Float, size_t, Float, size_t, Float, Float, Float, size_t, Float, Float, Float, Float, Float, size_t, Float, Float, Float, size_t);

		Model(const std::string name, Dataprovider* dataprovider) :
			Name(name),
			DataProv(dataprovider),
			Engine(dnnl::engine(dnnl::engine::kind::cpu, 0)),
			Device(dnn::Device(Engine, dnnl::stream(Engine))),
			Format(dnnl::memory::format_tag::any),
			PersistOptimizer(false),
			DisableLocking(true),
			Optimizer(Optimizers::AdaDelta),
			TaskState(TaskStates::Stopped),
			State(States::Idle),
			Dataset(Datasets::cifar10),			// Dataset
			SampleC(0),							// Dim
			SampleD(0),
			SampleH(0),
			SampleW(0),
			MeanStdNormalization(true),			// MeanStd
			MirrorPad(false),					// MirrorPad or ZeroPad
			PadD(0),
			PadH(0),
			PadW(0),
			RandomCrop(false),					// RandomCrop
			BatchNormScaling(true),				// Scaling
			BatchNormMomentum(Float(0.995)),	// Momentum
			BatchNormEps(Float(1e-04)),			// Eps
			Dropout(Float(0)),					// Dropout
			WeightsFiller(Fillers::HeNormal),	// WeightsFiller
			WeightsScale(Float(0.05)),			// WeightsScale
			WeightsLRM(Float(1)),				// WeightsLRM
			WeightsWDM(Float(1)),				// WeightsWDM
			BiasesFiller(Fillers::Constant),	// BiasesFiller
			BiasesScale(Float(0)),				// BiasesScale
			BiasesLRM(Float(1)),				// BiasesLRM
			BiasesWDM(Float(1)),				// BiasesWDM
			HasBias(true),						// Biases
			AlphaFiller(Float(0)),				// Alpha
			BetaFiller(Float(0)),				// Beta
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
			Layers(std::vector< std::shared_ptr<Layer>>()),
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
			FirstUnlockedLayer(1)
			//LogInterval(10000)
		{
#ifdef DNN_LOG
			dnnl_set_verbose(2);
#else
			dnnl_set_verbose(0);
#endif

#ifdef DNN_AVX512
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::all);
#else
			dnnl::set_max_cpu_isa(dnnl::cpu_isa::avx2);
#endif
		}

		virtual ~Model() = default;
				
		void ResetWeights()
		{
			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				ResettingWeights.store(true);

				for (auto l = 0ull; l < Layers.size(); l++)
				{
					while (Layers[l]->RefreshingStats.load())
						std::this_thread::sleep_for(std::chrono::milliseconds(100));

					Layers[l]->ResetWeights(WeightsFiller, WeightsScale, BiasesFiller, BiasesScale);
					Layers[l]->ResetOptimizer(Optimizer);
				}

				ResettingWeights.store(false);
			}
		}

		bool IsUniqueLayerName(const std::string& name) const
		{
			std::string nameLower(name);
			std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
			
			for (auto &layer : Layers)
			{
				auto &layerName = layer->Name;
				std::transform(layerName.begin(), layerName.end(), layerName.begin(), ::tolower);
				if (layerName == nameLower)
					return false;
			}

			return true;
		}

		void SetHyperParameters(const Float adaDeltaEps, const Float adaGradEps, const Float adamEps, const Float adamBeta2, const Float adamaxEps, const Float adamaxBeta2, const Float rmsPropEps, const Float radamEps, const Float radamBeta1, const Float radamBeta2)
		{
			for (auto &layer : Layers)
			{
				layer->AdaDeltaEps = adaDeltaEps;
				layer->AdaGradEps = adaGradEps;
				layer->AdamEps = adamEps;
				layer->AdamBeta2 = adamBeta2;
				layer->AdamaxEps = adamaxEps;
				layer->AdamaxBeta2 = adamaxBeta2;
				layer->RMSPropEps = rmsPropEps;
				layer->RAdamEps = radamEps;
				layer->RAdamBeta1 = radamBeta1;
				layer->RAdamBeta2 = radamBeta2;
			}
		}

		std::vector<std::shared_ptr<Layer>> GetLayerInputs(const std::vector<std::string>& inputs) const
		{
			auto list = std::vector<std::shared_ptr<Layer>>();

			bool exists;
			for (auto &name : inputs)
			{
				exists = false;
				for (auto &layer : Layers)
				{
					if (layer->Name == name)
					{
						list.push_back(layer);
						exists = true;
					}
				}

				if (!exists)
					throw std::invalid_argument(std::string("Invalid input layer: " + name).c_str());
			}

			return list;
		}

		std::vector<std::shared_ptr<Layer>> GetLayerOutputs(const Layer& parentLayer) const
		{
			auto list = std::vector<std::shared_ptr<Layer>>();

			for (auto &layer : Layers)
				if (layer->Name != parentLayer.Name)
					for (auto &inputs : layer->Inputs)
						if (inputs->Name == parentLayer.Name)
						{
							list.push_back(inputs);
							break;
						}
			
			return list;
		}

		std::vector<std::shared_ptr<Layer>> SetRelations()
		{
			// This determines how the backprop step correctly flows
			// When SharesInput is true we have to add our diff vector instead of just copying it because there's more than one layer involved
			for (auto &layer : Layers)
			{
				layer->SharesInput = false;
				//layer->Outputs = GetLayerOutputs(*layer.get());
			}

			auto unreferencedLayers = std::vector<std::shared_ptr<Layer>>();

			for (auto &layer : Layers)
			{
				auto count = GetLayerOutputs(*layer.get()).size();

				if (count > 1)
				{
					for (auto &l : Layers)
					{
						if (l->Name == layer->Name)
							continue;

						for (auto &inputs : l->Inputs)
						{
							if (inputs->Name == layer->Name)
							{
								l->SharesInput = true;
								count--;
								break;
							}
						}

						if (count == 1)
							break;
					}
				}
				else
				{
					if (count == 0 && layer->LayerType != LayerTypes::Cost)
						unreferencedLayers.push_back(layer);
				}
			}

			return unreferencedLayers;
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

		void SetLayerLocking(const size_t layerIndex, const bool locked)
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
	
		auto GetWeightsSize(const bool persistOptimizer) const
		{
			std::streamsize weightsSize = 0;

			for (auto &layer : Layers)
				weightsSize += layer->GetWeightsSize(persistOptimizer, Optimizer);

			return weightsSize;
		}

		auto GetNeuronsSize(const size_t batchSize) const
		{
			size_t neuronsSize = 0;

			for (auto &layer : Layers)
				neuronsSize += layer->GetNeuronsSize(batchSize);

			return neuronsSize;
		}

		bool BatchNormalizationUsed() const
		{
			for (auto &layer : Layers)
				if (layer->LayerType == LayerTypes::BatchNorm || layer->LayerType == LayerTypes::BatchNormHardLogistic || layer->LayerType == LayerTypes::BatchNormHardSwish || layer->LayerType == LayerTypes::BatchNormHardSwishDropout || layer->LayerType == LayerTypes::BatchNormRelu || layer->LayerType == LayerTypes::BatchNormReluDropout || layer->LayerType == LayerTypes::BatchNormSwish)
					return true;

			return false;
		}

		void AddTrainingRate(const TrainingRate rate, const bool clear, const size_t gotoEpoch)
		{
			if (clear)
				TrainingRates.clear();

			TotalCycles = 1;
			GoToEpoch = gotoEpoch;

			auto decayAfterEopchs = rate.DecayAfterEpochs;
			if (rate.Epochs < decayAfterEopchs)
				decayAfterEopchs = rate.Epochs;

			auto totIteration = rate.Epochs / decayAfterEopchs;
			auto newRate = rate.MaximumRate;

			for (auto i = 0ull; i < totIteration; i++)
			{
				if ((i + 1) >= gotoEpoch)
					TrainingRates.push_back(TrainingRate(newRate, rate.BatchSize, rate.Cycles, decayAfterEopchs, rate.EpochMultiplier, rate.MinimumRate, rate.L2Penalty, rate.Momentum, Float(1), 1, rate.HorizontalFlip, rate.VerticalFlip, rate.Dropout, rate.Cutout, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				if (newRate * rate.DecayFactor > rate.MinimumRate)
					newRate *= rate.DecayFactor;
				else
					newRate = rate.MinimumRate;
			}

			if ((totIteration * decayAfterEopchs) < rate.Epochs)
				TrainingRates.push_back(TrainingRate(newRate, rate.BatchSize, rate.Cycles, rate.Epochs - (totIteration * decayAfterEopchs), rate.EpochMultiplier, rate.MinimumRate, rate.L2Penalty, rate.Momentum, Float(1), decayAfterEopchs, rate.HorizontalFlip, rate.VerticalFlip, rate.Dropout, rate.Cutout, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
		}

		void AddTrainingRateSGDR(const TrainingRate rate, const bool clear, const size_t gotoEpoch)
		{
			if (clear)
				TrainingRates.clear();

			TotalCycles = rate.Cycles;
			GoToEpoch = gotoEpoch;

			auto maxRate = rate.MaximumRate;
			auto minRate = rate.MinimumRate;
			auto epoch = 0ull;
			for (auto c = 0ull; c < rate.Cycles; c++)
			{
				const auto total = rate.Epochs * (c > 0 ? (rate.EpochMultiplier != 1 ? c * rate.EpochMultiplier : 1) : 1);
				for (auto i = 0ull; i < total; i++)
				{
					const auto newRate = (minRate + Float(0.5) * (maxRate - minRate) * (Float(1) + std::cos(Float(i) / Float(total) * Float(3.1415926535897932384626433832))));

					epoch++;
					if (epoch >= gotoEpoch)
						TrainingRates.push_back(TrainingRate(newRate, rate.BatchSize, c + 1, 1, rate.EpochMultiplier, minRate, rate.L2Penalty, rate.Momentum, Float(1), 1, rate.HorizontalFlip, rate.VerticalFlip, rate.Dropout, rate.Cutout, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
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
			if (TaskState.load() != TaskStates::Running)
			{
				if (TaskState.load() == TaskStates::Paused)
				{
					while (TaskState.load() == TaskStates::Paused)
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
				}
				else
					return false;
			}

			return true;
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
			for (auto &layer : Layers)
				layer->SetOptimizer(optimizer);

			Optimizer = optimizer;
		}

#ifdef DNN_STOCHASTIC
		void CostFunction(const States state)
		{
			for (auto &cost : CostLayers)
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

		void Recognized(const States state, const std::vector<size_t>& sampleLabel)
		{
			for (auto &cost : CostLayers)
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

				if (hotIndex != sampleLabel[labelIndex])
				{
					if (state == States::Training)
						cost->TrainErrors++;
					else
						cost->TestErrors++;
				}

				if (state == States::Testing)
					cost->ConfusionMatrix[hotIndex][sampleLabel[labelIndex]]++;
			}
		}
#endif

		void CostFunctionBatch(const States state, const size_t batchSize, const bool overflow, const size_t skipCount)
		{
			for (auto &cost : CostLayers)
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

		void RecognizedBatch(const States state, const size_t batchSize, const bool overflow, const size_t skipCount, const std::vector<std::vector<size_t>>& sampleLabels)
		{
			for (auto &cost : CostLayers)
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

					if (hotIndex != sampleLabels[b][labelIndex])
					{
						if (state == States::Training)
							cost->TrainErrors++;
						else
							cost->TestErrors++;
					}

					if (state == States::Testing)
						cost->ConfusionMatrix[hotIndex][sampleLabels[b][labelIndex]]++;
				}
			}
		}

		void SetBatchSize(const size_t batchSize)
		{
			if (!BatchSizeChanging.load() && !ResettingWeights.load())
			{
				BatchSizeChanging.store(true);

				for (auto i = 0ull; i < Layers.size(); i++)
					Layers[i]->SetBatchSize(batchSize);

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

		void Training()
		{
			if (TaskState.load() == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);

				//auto oldWeightSaveFileName = std::string();
                
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
						std::cout << std::string("Total model size: ") << std::to_string(GetNeuronsSize(CurrentTrainingRate.BatchSize - BatchSize)/1024/1024) << " MB" << std::endl << std::endl;
						State.store(States::Completed);
						return;
					}
				std::cout << std::string("Total model size: ") << std::to_string(GetNeuronsSize(CurrentTrainingRate.BatchSize - BatchSize)/1024/1024) << " MB" << std::endl << std::endl;
				SetBatchSize(CurrentTrainingRate.BatchSize);
			
				auto learningRateEpochs = CurrentTrainingRate.Epochs;
				auto learningRateIndex = 0ull;

				RandomTrainingSamples = std::vector<size_t>(DataProv->TrainingSamplesCount);
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

				SetOptimizer(Optimizer);
				if (!PersistOptimizer)
					for (auto l = 1ull; l < Layers.size(); l++)
						Layers[l]->ResetOptimizer(Optimizer);
				else
					for (auto l = 1ull; l < Layers.size(); l++)
						Layers[l]->CheckOptimizer(Optimizer);

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

						if (BatchSize != CurrentTrainingRate.BatchSize)
							SetBatchSize(CurrentTrainingRate.BatchSize);

						learningRateEpochs += CurrentTrainingRate.Epochs;
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

						const auto shuffleCount = UniformInt<size_t>(16, 32);
						for (auto shuffle = 0ull; shuffle < shuffleCount; shuffle++)
							std::shuffle(std::begin(RandomTrainingSamples), std::end(RandomTrainingSamples), std::mt19937(physicalSeed()));

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

								for (auto i = 0ull; i < Layers.size(); i++)
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
							SaveWeights((DataProv->StorageDirectory / "Definitions" / (Name + "-weights") / (Name + " (epoch " + std::to_string(CurrentEpoch) + " - " + std::to_string(TestErrors) + " errors).weights")).string().c_str(), PersistOptimizer);

							//auto fileName = (DataProv->StorageDirectory / "Definitions" /  (Name + "-weights") / (Name + " (epoch " + std::to_string(CurrentEpoch) + " - " + std::to_string(TestErrors) + " errors).weights")).string().c_str();
							//if (TestErrors <= BestScore)
							//{
							//	BestScore = TestErrors;
							//	oldWeightSaveFileName = fileName;
							//}
							/*if (!oldWeightSaveFileName.empty() && file_exist(oldWeightSaveFileName))
							DeleteFile(oldWeightSaveFileName.c_str());
							oldWeightSaveFileName = fileName;*/

							//auto cycle = CurrentCycle;
							/*if (CurrentEpoch - (GoToEpoch - 1) == learningRateEpochs)
								cycle = TrainingRates[learningRateIndex+1].Cycles;*/

							State.store(States::NewEpoch);
							NewEpoch(CurrentCycle, CurrentEpoch, TotalEpochs, CurrentTrainingRate.HorizontalFlip, CurrentTrainingRate.VerticalFlip, CurrentTrainingRate.Dropout, CurrentTrainingRate.Cutout, CurrentTrainingRate.AutoAugment, CurrentTrainingRate.ColorCast, CurrentTrainingRate.ColorAngle, CurrentTrainingRate.Distortion, CurrentTrainingRate.Interpolation, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, CurrentTrainingRate.MaximumRate, CurrentTrainingRate.BatchSize, CurrentTrainingRate.Momentum, CurrentTrainingRate.L2Penalty, AvgTrainLoss, TrainErrorPercentage, Float(100) - TrainErrorPercentage, TrainErrors, AvgTestLoss, TestErrorPercentage, Float(100) - TestErrorPercentage, TestErrors);
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
				auto timePointGlobal = timer.now();;
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
			for (auto &cost : CostLayers)
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

		bool GetInputSnapShot(std::vector<Float>* snapshot, std::vector<size_t>* label)
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

#ifdef DNN_STOCHASTIC
		std::vector<size_t> TrainSample(const size_t index)
		{
			const auto randomIndex = RandomTrainingSamples[index];

			auto SampleLabel = DataProv->TrainingLabels[randomIndex];

			auto dstImageByte = DataProv->TrainingSamples[randomIndex];

			if (CurrentTrainingRate.HorizontalFlip && TrainingSamplesHFlip[randomIndex])
				dstImageByte = Image<Byte>::HorizontalMirror(dstImageByte);

			if (CurrentTrainingRate.VerticalFlip && TrainingSamplesVFlip[randomIndex])
				dstImageByte = Image<Byte>::VerticalMirror(dstImageByte);

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				dstImageByte = Image<Byte>::ColorCast(dstImageByte, CurrentTrainingRate.ColorAngle);

			if (dstImageByte.Depth != SampleD || dstImageByte.Height != SampleH || dstImageByte.Width != SampleW)
				dstImageByte = Image<Byte>::Resize(dstImageByte, SampleD, SampleH, SampleW, Interpolation(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				dstImageByte = Image<Byte>::AutoAugment(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				dstImageByte = Image<Byte>::Padding(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				dstImageByte = Image<Byte>::Distorted(dstImageByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolation(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
				dstImageByte = Image<Byte>::RandomCutout(dstImageByte, DataProv->Mean);

			if (CurrentTrainingRate.Dropout > Float(0))
				dstImageByte = Image<Byte>::Dropout(dstImageByte, CurrentTrainingRate.Dropout, DataProv->Mean);

			if (RandomCrop)
				dstImageByte = Image<Byte>::RandomCrop(dstImageByte, SampleD, SampleH, SampleW, DataProv->Mean);

			for (auto c = 0ull; c < dstImageByte.Channels; c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(dstImageByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(dstImageByte, c);

				for (auto d = 0ull; d < dstImageByte.Depth; d++)
					for (auto h = 0ull; h < dstImageByte.Height; h++)
						for (auto w = 0ull; w < dstImageByte.Width; w++)
							Layers[0]->Neurons[(c * dstImageByte.ChannelSize()) + (d * dstImageByte.Area()) + (h * dstImageByte.Width) + w] = (dstImageByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<size_t> TestSample(const size_t index)
		{
			auto SampleLabel = DataProv->TestingLabels[index];

			auto dstImageByte = DataProv->TestingSamples[index];

			if (dstImageByte.Depth != SampleD || dstImageByte.Height != SampleH || dstImageByte.Width != SampleW)
				dstImageByte = Image<Byte>::Resize(dstImageByte, SampleD, SampleH, SampleW, Interpolation(CurrentTrainingRate.Interpolation));

			dstImageByte = Image<Byte>::Padding(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (RandomCrop)
				dstImageByte = Image<Byte>::Crop(dstImageByte, Position::Center, SampleD, SampleH, SampleW, DataProv->Mean);

			for (auto c = 0ull; c < dstImageByte.Channels; c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(dstImageByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(dstImageByte, c);

				for (auto d = 0ull; d < dstImageByte.Depth; d++)
					for (auto h = 0ull; h < dstImageByte.Height; h++)
						for (auto w = 0ull; w < dstImageByte.Width; w++)
							Layers[0]->Neurons[(c * dstImageByte.ChannelSize()) + (d * dstImageByte.Area()) + (h * dstImageByte.Width) + w] = (dstImageByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<size_t> TestAugmentedSample(const size_t index)
		{
			auto SampleLabel = DataProv->TestingLabels[index];

			auto dstImageByte = DataProv->TestingSamples[index];

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				dstImageByte = Image<Byte>::ColorCast(dstImageByte, CurrentTrainingRate.ColorAngle);

			if (CurrentTrainingRate.HorizontalFlip && TestingSamplesHFlip[index])
				dstImageByte = Image<Byte>::HorizontalMirror(dstImageByte);

			if (CurrentTrainingRate.VerticalFlip && TestingSamplesVFlip[index])
				dstImageByte = Image<Byte>::VerticalMirror(dstImageByte);

			if (dstImageByte.Depth != SampleD || dstImageByte.Height != SampleH || dstImageByte.Width != SampleW)
				dstImageByte = Image<Byte>::Resize(dstImageByte, SampleD, SampleH, SampleW, Interpolation(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				dstImageByte = Image<Byte>::AutoAugment(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				dstImageByte = Image<Byte>::Padding(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				dstImageByte = Image<Byte>::Distorted(dstImageByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolation(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
				dstImageByte = Image<Byte>::RandomCutout(dstImageByte, DataProv->Mean);

			if (RandomCrop)
				dstImageByte = Image<Byte>::Crop(dstImageByte, Position::Center, SampleD, SampleH, SampleW, DataProv->Mean);

			if (CurrentTrainingRate.Dropout > Float(0))
				dstImageByte = Image<Byte>::Dropout(dstImageByte, CurrentTrainingRate.Dropout, DataProv->Mean);

			for (auto c = 0ull; c < dstImageByte.Channels; c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(dstImageByte, c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(dstImageByte, c);

				for (auto d = 0ull; d < dstImageByte.Depth; d++)
					for (auto h = 0ull; h < dstImageByte.Height; h++)
						for (auto w = 0ull; w < dstImageByte.Width; w++)
							Layers[0]->Neurons[(c * dstImageByte.ChannelSize()) + (d * dstImageByte.Area()) + (h * dstImageByte.Width) + w] = (dstImageByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}
#endif

		std::vector<std::vector<size_t>> TrainBatch(const size_t index, const size_t batchSize)
		{
			auto SampleLabels = std::vector<std::vector<size_t>>(batchSize, std::vector<size_t>(DataProv->Hierarchies));

			const auto resize = DataProv->TrainingSamples[0].Depth != SampleD || DataProv->TrainingSamples[0].Height != SampleH || DataProv->TrainingSamples[0].Width != SampleW;

			for_i(batchSize, [=, &SampleLabels](const size_t batchIndex)
			{
				const auto randomIndex = (index + batchIndex >= DataProv->TrainingSamplesCount) ? RandomTrainingSamples[batchIndex] : RandomTrainingSamples[index + batchIndex];

				SampleLabels[batchIndex] = DataProv->TrainingLabels[randomIndex];

				auto dstImageByte = Image<Byte>(DataProv->TrainingSamples[randomIndex]);

				if (CurrentTrainingRate.HorizontalFlip && TrainingSamplesHFlip[randomIndex])
					dstImageByte = Image<Byte>::HorizontalMirror(dstImageByte);

				if (CurrentTrainingRate.VerticalFlip && TrainingSamplesVFlip[randomIndex])
					dstImageByte = Image<Byte>::VerticalMirror(dstImageByte);

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					dstImageByte = Image<Byte>::ColorCast(dstImageByte, CurrentTrainingRate.ColorAngle);

				if (resize)
					dstImageByte = Image<Byte>::Resize(dstImageByte, SampleD, SampleH, SampleW, Interpolation(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					dstImageByte = Image<Byte>::AutoAugment(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					dstImageByte = Image<Byte>::Padding(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					dstImageByte = Image<Byte>::Distorted(dstImageByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolation(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
					dstImageByte = Image<Byte>::RandomCutout(dstImageByte, DataProv->Mean);

				if (RandomCrop)
					dstImageByte = Image<Byte>::RandomCrop(dstImageByte, SampleD, SampleH, SampleW, DataProv->Mean);

				if (CurrentTrainingRate.Dropout > Float(0))
					dstImageByte = Image<Byte>::Dropout(dstImageByte, CurrentTrainingRate.Dropout, DataProv->Mean);

				for (auto c = 0ull; c < dstImageByte.Channels; c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(dstImageByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(dstImageByte, c);

					for (auto d = 0ull; d < dstImageByte.Depth; d++)
						for (auto h = 0ull; h < dstImageByte.Height; h++)
							for (auto w = 0ull; w < dstImageByte.Width; w++)
								Layers[0]->Neurons[batchIndex * dstImageByte.Size() + (c * dstImageByte.ChannelSize()) + (d * dstImageByte.Area()) + (h * dstImageByte.Width) + w] = (dstImageByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<size_t>> TestBatch(const size_t index, const size_t batchSize)
		{
			auto SampleLabels = std::vector<std::vector<size_t>>(batchSize, std::vector<size_t>(DataProv->Hierarchies));

			const auto resize = DataProv->TestingSamples[0].Depth != SampleD || DataProv->TestingSamples[0].Height != SampleH || DataProv->TestingSamples[0].Width != SampleW;

			for_i(batchSize, MEDIUM_COMPUTE, [=, &SampleLabels](const size_t batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestingSamplesCount) ? batchIndex : index + batchIndex;

				SampleLabels[batchIndex] = DataProv->TestingLabels[sampleIndex];

				auto dstImageByte = DataProv->TestingSamples[sampleIndex];

				if (resize)
					dstImageByte = Image<Byte>::Resize(dstImageByte, SampleD, SampleH, SampleW, Interpolation(CurrentTrainingRate.Interpolation));

				dstImageByte = Image<Byte>::Padding(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				dstImageByte = Image<Byte>::Crop(dstImageByte, Position::Center, SampleD, SampleH, SampleW, DataProv->Mean);

				for (auto c = 0ull; c < dstImageByte.Channels; c++)
				{
					const Float mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(dstImageByte, c);
					const Float stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(dstImageByte, c);

					for (auto d = 0ull; d < dstImageByte.Depth; d++)
						for (auto h = 0ull; h < dstImageByte.Height; h++)
							for (auto w = 0ull; w < dstImageByte.Width; w++)
								Layers[0]->Neurons[batchIndex * dstImageByte.Size() + (c * dstImageByte.ChannelSize()) + (d * dstImageByte.Area()) + (h * dstImageByte.Width) + w] = (dstImageByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<size_t>> TestAugmentedBatch(const size_t index, const size_t batchSize)
		{
			auto SampleLabels = std::vector<std::vector<size_t>>(batchSize, std::vector<size_t>(DataProv->Hierarchies));

			const auto resize = DataProv->TestingSamples[0].Depth != SampleD || DataProv->TestingSamples[0].Height != SampleH || DataProv->TestingSamples[0].Width != SampleW;

			for_i(batchSize, [=, &SampleLabels](const size_t batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestingSamplesCount) ? batchIndex : index + batchIndex;

				SampleLabels[batchIndex] = DataProv->TestingLabels[sampleIndex];

				auto dstImageByte = DataProv->TestingSamples[sampleIndex];

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					dstImageByte = Image<Byte>::ColorCast(dstImageByte, CurrentTrainingRate.ColorAngle);

				if (CurrentTrainingRate.HorizontalFlip && TestingSamplesHFlip[sampleIndex])
					dstImageByte = Image<Byte>::HorizontalMirror(dstImageByte);

				if (CurrentTrainingRate.VerticalFlip && TestingSamplesVFlip[sampleIndex])
					dstImageByte = Image<Byte>::VerticalMirror(dstImageByte);

				if (resize)
					dstImageByte = Image<Byte>::Resize(dstImageByte, SampleD, SampleH, SampleW, Interpolation(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					dstImageByte = Image<Byte>::AutoAugment(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					dstImageByte = Image<Byte>::Padding(dstImageByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					dstImageByte = Image<Byte>::Distorted(dstImageByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolation(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
					dstImageByte = Image<Byte>::RandomCutout(dstImageByte, DataProv->Mean);

				if (RandomCrop)
					dstImageByte = Image<Byte>::Crop(dstImageByte, Position::Center, SampleD, SampleH, SampleW, DataProv->Mean);

				if (CurrentTrainingRate.Dropout > Float(0))
					dstImageByte = Image<Byte>::Dropout(dstImageByte, CurrentTrainingRate.Dropout, DataProv->Mean);

				for (auto c = 0ull; c < dstImageByte.Channels; c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : Image<Byte>::GetChannelMean(dstImageByte, c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : Image<Byte>::GetChannelStdDev(dstImageByte, c);

					for (auto d = 0ull; d < dstImageByte.Depth; d++)
						for (auto h = 0ull; h < dstImageByte.Height; h++)
							for (auto w = 0ull; w < dstImageByte.Width; w++)
								Layers[0]->Neurons[batchIndex * dstImageByte.Size() + (c * dstImageByte.ChannelSize()) + (d * dstImageByte.Area()) + (h * dstImageByte.Width) + w] = (dstImageByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}
	
		
		void ForwardProp(const size_t batchSize)
		{
			for (size_t i = 0; i < Layers.size(); i++)
				Layers[i]->ForwardProp(batchSize, State.load() == States::Training);
		}

		void BackwardProp(const size_t batchSize)
		{
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
		}
		
		int SaveWeights(std::string fileName, const bool persistOptimizer = false) const
		{
			auto os = std::ofstream(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

			if (!os.bad() && os.is_open())
			{
				for (auto i = 0ull; i < Layers.size(); i++)
					Layers[i]->Save(os, persistOptimizer, Optimizer);

				os.close();

				return 0;
			}

			return -1;
		}

		int LoadWeights(std::string fileName, const bool persistOptimizer = false)
		{
			if (GetFileSize(fileName) == GetWeightsSize(persistOptimizer))
			{
				auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

				if (!is.bad() && is.is_open())
				{
					for (auto i = 0ull; i < Layers.size(); i++)
						Layers[i]->Load(is, persistOptimizer, Optimizer);

					is.close();

					return 0;
				}
			}

			return -1;
		}

		int SaveLayerWeights(std::string fileName, const size_t layerIndex, const bool persistOptimizer = false) const
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

		int LoadLayerWeights(std::string fileName, const size_t layerIndex, const bool persistOptimizer = false)
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
	};
}