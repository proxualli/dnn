#include "Definition.h"

using namespace dnn;

std::unique_ptr<dnn::Model> model;
std::unique_ptr<dnn::Dataprovider> dataprovider;

#ifdef DNN_DLL
#if defined _WIN32 || defined __CYGWIN__ || defined __MINGW32__
#if defined DNN_LOG 
FILE* stream;
#endif
BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpReserved)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
#if defined DNN_LOG
		AllocConsole();
		_wfreopen_s(&stream, L"CONOUT$", L"w", stdout);
#endif
	break;

	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
		break;

	case DLL_PROCESS_DETACH:
#if defined DNN_LOG
		fclose(stream);
		FreeConsole();
#endif
		break;
	}

	return TRUE;
}

#ifdef DNN_EXPORTS
#ifdef __GNUC__
#define DNN_API __attribute__ ((dllexport))
#else
#define DNN_API __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
#endif
#else
#ifdef __GNUC__
#define DNN_API __attribute__ ((dllimport))
#else
#define DNN_API __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
#endif
#endif
#else
#if __GNUC__ >= 4
#define DNN_API __attribute__ ((visibility ("default")))
#else
#define DNN_API
#endif
#endif
#else
#define DNN_API
#endif


extern "C" DNN_API void DNNSetNewEpochDelegate(void(*newEpoch)(UInt, UInt, UInt, UInt, Float, Float, bool, bool, Float, Float, Float, Float, UInt, Float, UInt, Float, Float, Float, UInt, Float, Float, Float, Float, Float, UInt, Float, Float, Float, UInt))
{																	     
	if (model)
		model->NewEpoch = newEpoch;
}

extern "C" DNN_API void DNNModelDispose()
{
	if (model)
		model.reset();
}

//extern "C" DNN_API void DNNPrintModel(const std::string& fileName)
//{
//	if (model)
//	{
//		auto os = std::ofstream(fileName);
//
//		if (os)
//		{
//			for (auto& layer : model->Layers)
//			{
//				os << layer->Name << "  (SharesInput " << std::to_string(layer->SharesInput) << ")  InputLayer " << layer->InputLayer->Name << "  :  ";
//				for (auto input : layer->Inputs)
//					os << input->Name << "  ";
//				os << std::endl;
//			}
//			os.flush();
//			os.close();
//		}
//	}
//}

extern "C" DNN_API Model* DNNModel(const std::string name)
{
	model = std::make_unique<Model>(name, dataprovider.get());

	return model.get();
}

extern "C" DNN_API void DNNDataprovider(const std::string& directory)
{
	dataprovider = std::make_unique<Dataprovider>(directory);
}

extern "C" DNN_API bool DNNLoadDataset()
{
	if (model)
		return dataprovider->LoadDataset(model->Dataset);

	return false;
}

extern "C" DNN_API bool DNNCheckDefinition(std::string& definition, CheckMsg& checkMsg)
{
	return Definition::CheckDefinition(definition, checkMsg);
}

extern "C" DNN_API int DNNReadDefinition(const std::string& definition, CheckMsg& checkMsg)
{
	dnn::Model* ptr = nullptr;

	ptr = Definition::ReadDefinition(definition, dataprovider.get(), checkMsg);

	if (ptr)
	{
		model.reset();
		model = std::unique_ptr<Model>(ptr);
		ptr = nullptr;

		return 1;
	}

	return 0;
}

extern "C" DNN_API int DNNLoadDefinition(const std::string& fileName, CheckMsg& checkMsg)
{
	dnn::Model* ptr = nullptr;

	ptr = Definition::LoadDefinition(fileName, dataprovider.get(), checkMsg);
	
	if (ptr)
	{
		model.reset();
		model = std::unique_ptr<Model>(ptr);
		ptr = nullptr;

		return 1;
	}

	return 0;
}

extern "C" DNN_API void DNNGetLayerInputs(const UInt layerIndex, std::vector<UInt>* inputs)
{
	if (model && layerIndex < model->Layers.size())
	{
		for (auto i = 0ull; i < model->Layers[layerIndex]->Inputs.size(); i++)
		{
			auto inputLayerName = model->Layers[layerIndex]->Inputs[i]->Name;
			for (auto index = 0ull; index < model->Layers.size(); index++)
				if (model->Layers[index]->Name == inputLayerName)
					inputs->push_back(index);
		}
	}
}

extern "C" DNN_API bool DNNBatchNormalizationUsed()
{
	if (model)
		return model->BatchNormalizationUsed();

	return false;
}

extern "C" DNN_API bool DNNStochasticEnabled()
{
#ifdef DNN_STOCHASTIC
	return true;
#else
	return false;
#endif
}

extern "C" DNN_API bool DNNSetFormat(const bool plain)
{
	if (model)
		return model->SetFormat(plain);
		
	return false;
}

extern "C" DNN_API void DNNGetConfusionMatrix(const UInt costLayerIndex, std::vector<std::vector<UInt>>* confusionMatrix)
{
	if (model && costLayerIndex < model->CostLayers.size())
		(*confusionMatrix) = model->CostLayers[costLayerIndex]->ConfusionMatrix;
}

extern "C" DNN_API void DNNPersistOptimizer(const bool persistOptimizer)
{
	if (model)
		model->PersistOptimizer = persistOptimizer;
}

extern "C" DNN_API void DNNResetOptimizer()
{
	if (model)
		model->ResetOptimizer();
}

extern "C" DNN_API void DNNSetOptimizer(const Optimizers optimizer)
{
	if (model)
		model->SetOptimizer(optimizer);
}

extern "C" DNN_API void DNNDisableLocking(const bool disable)
{
	if (model)
		model->DisableLocking = disable;
}

extern "C" DNN_API void DNNResetWeights()
{
	if (model)
		model->ResetWeights();
}

extern "C" DNN_API void DNNResetLayerWeights(const UInt layerIndex)
{
	if (model && layerIndex < model->Layers.size())
		model->Layers[layerIndex]->ResetWeights(model->WeightsFiller, model->WeightsScale, model->BiasesFiller, model->BiasesScale);
}

extern "C" DNN_API void DNNGetImage(const UInt layerIndex, const Byte fillColor, Byte* image)
{
	if (model && layerIndex < model->Layers.size() && !model->BatchSizeChanging.load() && !model->ResettingWeights.load())
	{
		switch (model->Layers[layerIndex]->LayerType)
		{
			case LayerTypes::Activation:
			case LayerTypes::BatchNorm:
			case LayerTypes::BatchNormMish:
			case LayerTypes::BatchNormMishDropout:
			case LayerTypes::BatchNormHardLogistic:
			case LayerTypes::BatchNormHardSwish:
			case LayerTypes::BatchNormHardSwishDropout:
			case LayerTypes::BatchNormRelu:
			case LayerTypes::BatchNormReluDropout:
			case LayerTypes::BatchNormSwish:
			case LayerTypes::BatchNormSwishDropout:
			case LayerTypes::BatchNormTanhExp:
			case LayerTypes::BatchNormTanhExpDropout:
			case LayerTypes::Convolution:
			case LayerTypes::ConvolutionTranspose:
			case LayerTypes::Dense:
			case LayerTypes::DepthwiseConvolution:
			case LayerTypes::PartialDepthwiseConvolution:
			case LayerTypes::LayerNorm:
			{
				auto img = model->Layers[layerIndex]->GetImage(fillColor);
				std::memcpy(image, img.data(), img.size());
			}
			break;

			default:
				return;
		}
	}
}

extern "C" DNN_API bool DNNGetInputSnapShot(std::vector<Float>* snapshot, std::vector<UInt>* label)
{
	if (model)
		if ((model->TaskState.load() == TaskStates::Running) && ((model->State.load() == States::Training) || (model->State.load() == States::Testing)))
			return model->GetInputSnapShot(snapshot, label);

	return false;
}

extern "C" DNN_API void DNNGetLayerWeights(const UInt layerIndex, std::vector<Float>* weights, std::vector<Float>* biases)
{
	if (model && layerIndex < model->Layers.size() && model->Layers[layerIndex]->HasWeights)
	{
		for (auto i = 0ull; i < model->Layers[layerIndex]->WeightCount; i++)
			(*weights)[i] = model->Layers[layerIndex]->Weights[i];
	
		if (model->Layers[layerIndex]->HasBias)
			for (auto i = 0ull; i < model->Layers[layerIndex]->BiasCount; i++)
				(*biases)[i] = model->Layers[layerIndex]->Biases[i];
	}
}

extern "C" DNN_API void DNNAddLearningRate(const bool clear, const UInt gotoEpoch, const Optimizers optimizer, const Float momentum, const Float beta2, const Float L2penalty, const Float eps, const UInt batchSize, const UInt cycles, const UInt epochs, const UInt epochMultiplier, const Float maximumRate, const Float minimumRate, const Float decayFactor, const UInt decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const Interpolations interpolation, const Float scaling, const Float rotation)
{
	if (model)
		model->AddTrainingRate(TrainingRate(optimizer, momentum, beta2, L2penalty, eps, batchSize, cycles, epochs, epochMultiplier, maximumRate, minimumRate, decayAfterEpochs, decayFactor, horizontalFlip, verticalFlip, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, scaling, rotation), clear, gotoEpoch);
}

extern "C" DNN_API void DNNAddLearningRateSGDR(const bool clear, const UInt gotoEpoch, const Optimizers optimizer, const Float momentum, const Float beta2, const Float L2penalty, const Float eps, const UInt batchSize, const UInt cycles, const UInt epochs, const UInt epochMultiplier, const Float maximumRate, const Float minimumRate, const Float decayFactor, const UInt decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const Interpolations interpolation, const Float saling, const Float rotation)
{
	if (model)
		model->AddTrainingRateSGDR(TrainingRate(optimizer, momentum, beta2, L2penalty, eps, batchSize, cycles, epochs, epochMultiplier, maximumRate, minimumRate, decayAfterEpochs, decayFactor, horizontalFlip, verticalFlip, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, saling, rotation), clear, gotoEpoch);
}

extern "C" DNN_API void DNNTraining()
{
	if (model)
	{
		model->State.store(States::Idle);
		model->TrainingAsync();
	}
}

extern "C" DNN_API void DNNTesting()
{
	if (model)
	{
		model->State.store(States::Idle);
		model->TestingAsync();
	}
}

extern "C" DNN_API void DNNStop()
{
	if (model)
		model->StopTask();
}

extern "C" DNN_API void DNNPause()
{
	if (model)
		model->PauseTask();
}

extern "C" DNN_API void DNNResume()
{
	if (model)
		model->ResumeTask();
}

extern "C" DNN_API void DNNSetCostIndex(const UInt costLayerIndex)
{
	if (model && costLayerIndex < model->CostLayers.size())
		model->CostIndex = costLayerIndex;
}

extern "C" DNN_API void DNNGetCostInfo(const UInt costLayerIndex, UInt* trainErrors, Float* trainLoss, Float* avgTrainLoss, Float* trainErrorPercentage, UInt* testErrors, Float* testLoss, Float* avgTestLoss, Float* testErrorPercentage)
{
	if (model && costLayerIndex < model->CostLayers.size())
	{
		*trainErrors = model->CostLayers[costLayerIndex]->TrainErrors;
		*trainLoss = model->CostLayers[costLayerIndex]->TrainLoss;
		*avgTrainLoss = model->CostLayers[costLayerIndex]->AvgTrainLoss;
		*trainErrorPercentage = model->CostLayers[costLayerIndex]->TrainErrorPercentage;

		*testErrors = model->CostLayers[costLayerIndex]->TestErrors;
		*testLoss = model->CostLayers[costLayerIndex]->TestLoss;
		*avgTestLoss = model->CostLayers[costLayerIndex]->AvgTestLoss;
		*testErrorPercentage = model->CostLayers[costLayerIndex]->TestErrorPercentage;
	}
}

extern "C" DNN_API void DNNGetModelInfo(std::string* name, UInt* costIndex, UInt* costLayerCount, UInt* groupIndex, UInt* labelIndex, UInt* hierarchies, bool* meanStdNormalization, Costs* lossFunction, Datasets* dataset, UInt* layerCount, UInt* trainingSamples, UInt* testingSamples, std::vector<Float>* meanTrainSet, std::vector<Float>* stdTrainSet)
{
	if (model)
	{
		*name = model->Name;
		*costIndex = model->CostIndex;
		*costLayerCount = model->CostLayers.size();
		*groupIndex = model->GroupIndex;
		*labelIndex = model->LabelIndex;
		*meanStdNormalization = model->MeanStdNormalization;
		*hierarchies = dataprovider->Hierarchies;
		*lossFunction = model->CostFuction;
		*dataset = dataprovider->Dataset;
		*layerCount = model->Layers.size();
		*trainingSamples = dataprovider->TrainingSamplesCount;
		*testingSamples = dataprovider->TestingSamplesCount;
		
		(*meanTrainSet).clear();
		(*stdTrainSet).clear();
		
		switch (dataprovider->Dataset)
		{
		case Datasets::tinyimagenet:
		case Datasets::cifar10:
		case Datasets::cifar100:
			for (auto c = 0ull; c < 3ull; c++)
			{
				(*meanTrainSet).push_back(dataprovider->Mean[c]);
				(*stdTrainSet).push_back(dataprovider->StdDev[c]);
			}
			break;
		case Datasets::fashionmnist:
		case Datasets::mnist:
			(*meanTrainSet).push_back(dataprovider->Mean[0]);
			(*stdTrainSet).push_back(dataprovider->StdDev[0]);
			break;
		}
	}
}

extern "C" DNN_API void DNNGetTrainingInfo(UInt* currentCycle, UInt* totalCycles, UInt* currentEpoch, UInt* totalEpochs, bool* horizontalFlip, bool* verticalFlip, Float* dropout, Float* cutout, Float* autoAugment, Float* colorCast, UInt* colorAngle, Float* distortion, Interpolations* interpolation, Float* scaling, Float* rotation, UInt* sampleIndex, UInt* batchSize, Float* maximumRate, Optimizers* optimizer, Float* momentum, Float* l2Penalty, Float* avgTrainLoss, Float* trainErrorPercentage, UInt* trainErrors, Float* avgTestLoss, Float* testErrorPercentage, UInt* testErrors, Float* sampleSpeed, States* networkState, TaskStates* taskState)
{
	if (model)
	{
		const auto sampleIdx = model->SampleIndex + model->BatchSize;

		switch (model->State)
		{
		case States::Training:
		{
			model->TrainLoss = model->CostLayers[model->CostIndex]->TrainLoss;
			model->TrainErrors = model->CostLayers[model->CostIndex]->TrainErrors;
			model->TrainErrorPercentage = Float(model->TrainErrors * 100) / sampleIdx;
			model->AvgTrainLoss = model->TrainLoss / sampleIdx;
   
			*avgTrainLoss = model->AvgTrainLoss;
			*trainErrorPercentage = model->TrainErrorPercentage;
			*trainErrors = model->TrainErrors;
		}
		break;

		case States::Testing:
		{
			const auto adjustedsampleIndex = sampleIdx > dataprovider->TestingSamplesCount ? dataprovider->TestingSamplesCount : sampleIdx;
						
			model->TestLoss = model->CostLayers[model->CostIndex]->TestLoss;
			model->TestErrors = model->CostLayers[model->CostIndex]->TestErrors;
			model->TestErrorPercentage = Float(model->TestErrors * 100) / adjustedsampleIndex;
			model->AvgTestLoss = model->TestLoss / adjustedsampleIndex;

			*avgTestLoss = model->AvgTestLoss;
			*testErrorPercentage = model->TestErrorPercentage;
			*testErrors = model->TestErrors;
		}
		break;

		case States::Idle:
		case States::NewEpoch:
		case States::SaveWeights:
		case States::Completed:
		{
			// Do nothing
		}
		break;
		}
		
		*currentCycle = model->CurrentCycle;
		*totalCycles = model->TotalCycles;
		*currentEpoch = model->CurrentEpoch;
		*totalEpochs = model->TotalEpochs;
		*sampleIndex = model->SampleIndex;
		*horizontalFlip = model->CurrentTrainingRate.HorizontalFlip;
		*verticalFlip = model->CurrentTrainingRate.VerticalFlip;
		*dropout = model->CurrentTrainingRate.Dropout;
		*cutout = model->CurrentTrainingRate.Cutout;
		*autoAugment = model->CurrentTrainingRate.AutoAugment;
		*colorCast = model->CurrentTrainingRate.ColorCast;
		*colorAngle = model->CurrentTrainingRate.ColorAngle;
		*distortion = model->CurrentTrainingRate.Distortion;
		*interpolation = model->CurrentTrainingRate.Interpolation;
		*scaling = model->CurrentTrainingRate.Scaling;
		*rotation = model->CurrentTrainingRate.Rotation;
		*maximumRate = model->CurrentTrainingRate.MaximumRate;
		*optimizer = model->Optimizer;
		*momentum = model->CurrentTrainingRate.Momentum;
		*l2Penalty = model->CurrentTrainingRate.L2Penalty;
		*batchSize = model->BatchSize;
		*sampleSpeed = model->SampleSpeed;
		
		*networkState = model->State.load();
		*taskState = model->TaskState.load();
	}
}

extern "C" DNN_API void DNNGetTestingInfo(UInt* batchSize, UInt* sampleIndex, Float* avgTestLoss, Float* testErrorPercentage, UInt* testErrors, Float* sampleSpeed, States* networkState, TaskStates* taskState)
{
	if (model)
	{
		const auto sampleIdx = model->SampleIndex + model->BatchSize;
		const auto adjustedsampleIndex = sampleIdx > dataprovider->TestingSamplesCount ? dataprovider->TestingSamplesCount : sampleIdx;
							
		model->TestLoss = model->CostLayers[model->CostIndex]->TestLoss;
		model->TestErrors = model->CostLayers[model->CostIndex]->TestErrors;
		model->TestErrorPercentage = Float(model->TestErrors * 100) / adjustedsampleIndex;
		model->AvgTestLoss = model->TestLoss / adjustedsampleIndex;

		*batchSize = model->BatchSize;
		*sampleIndex = model->SampleIndex;
		*avgTestLoss = model->AvgTestLoss;
		*testErrorPercentage = model->TestErrorPercentage;
		*testErrors = model->TestErrors;
		*sampleSpeed = model->SampleSpeed;
				
		*networkState = model->State.load();
		*taskState = model->TaskState.load();
	}
}

extern "C" DNN_API void DNNRefreshStatistics(const UInt layerIndex, std::string* description, Stats* neuronsStats, Stats* weightsStats, Stats* biasesStats, Float* fpropLayerTime, Float* bpropLayerTime, Float* updateLayerTime, Float* fpropTime, Float* bpropTime, Float* updateTime, bool* locked)
{
	if (model && layerIndex < model->Layers.size())
	{
		while (model->BatchSizeChanging.load() || model->ResettingWeights.load())
			std::this_thread::sleep_for(std::chrono::milliseconds(250));

		auto statsOK = false;
		if (!model->BatchSizeChanging.load() && !model->ResettingWeights.load())
			statsOK = model->Layers[layerIndex]->RefreshStatistics(model->BatchSize);

		if (!statsOK)
		{
			model->StopTask();
			return;
		}

		*description = model->Layers[layerIndex]->GetDescription();

		*neuronsStats = model->Layers[layerIndex]->NeuronsStats;
		*weightsStats = model->Layers[layerIndex]->WeightsStats;
		*biasesStats = model->Layers[layerIndex]->BiasesStats;

		*fpropLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->Layers[layerIndex]->fpropTime).count()) / 1000;
		*bpropLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->Layers[layerIndex]->bpropTime).count()) / 1000;
		*updateLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->Layers[layerIndex]->updateTime).count()) / 1000;
		*fpropTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->fpropTime).count()) / 1000;
		*bpropTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->bpropTime).count()) / 1000;
		*updateTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->updateTime).count()) / 1000;

		*locked = model->Layers[layerIndex]->Lockable() ? model->Layers[layerIndex]->LockUpdate.load() : false;
	}
}

extern "C" DNN_API void DNNGetLayerInfo(const UInt layerIndex, UInt* inputsCount, LayerTypes* layerType, Activations* activationFunction, Costs* cost, std::string* name, std::string* description, UInt* neuronCount, UInt* weightCount, UInt* biasesCount, UInt* multiplier, UInt* groups, UInt* group, UInt* localSize, UInt* c, UInt* d, UInt* h, UInt* w, UInt* kernelH, UInt* kernelW, UInt* strideH, UInt* strideW, UInt* dilationH, UInt* dilationW, UInt* padD, UInt* padH, UInt* padW, Float* dropout, Float* labelTrue, Float* labelFalse, Float* weight, UInt* groupIndex, UInt* labelIndex, UInt* inputC, Float* alpha, Float* beta, Float* k, Algorithms* algorithm, Float* fH, Float* fW, bool* hasBias, bool* scaling, bool* acrossChannels, bool* locked, bool* lockable)
{
	if (model && layerIndex < model->Layers.size())
	{
		*inputsCount = model->Layers[layerIndex]->Inputs.size();
		*layerType = model->Layers[layerIndex]->LayerType;
		*name = model->Layers[layerIndex]->Name;
		*description = model->Layers[layerIndex]->GetDescription();
		*neuronCount = model->Layers[layerIndex]->CDHW;
		*weightCount = model->Layers[layerIndex]->WeightCount;
		*biasesCount = model->Layers[layerIndex]->BiasCount;
		*multiplier = 1;
		*groups = 1;
		*group = 1;
		*c = model->Layers[layerIndex]->C;
		*d = model->Layers[layerIndex]->D;
		*h = model->Layers[layerIndex]->H;
		*w = model->Layers[layerIndex]->W;
		*padD = model->Layers[layerIndex]->PadD;
		*padH = model->Layers[layerIndex]->PadH;
		*padW = model->Layers[layerIndex]->PadW;
		*dilationH = 1;
		*dilationW = 1;
		*kernelH = 0;
		*kernelW = 0;
		*strideH = 1;
		*strideW = 1;
		*algorithm = Algorithms::Linear;
		*fH = 1;
		*fW = 1;
		*dropout = Float(0);
		*labelTrue = Float(1);
		*labelFalse = Float(0);
		*weight = Float(1);
		*groupIndex = 0;
		*labelIndex = 0;
		*inputC = model->Layers[layerIndex]->InputLayer != nullptr ? model->Layers[layerIndex]->InputLayer->C : 0;
		*hasBias = model->Layers[layerIndex]->HasBias;
		*locked = model->Layers[layerIndex]->Lockable() ? model->Layers[layerIndex]->LockUpdate.load() : false;
		*lockable = model->Layers[layerIndex]->Lockable();

		switch (model->Layers[layerIndex]->LayerType)
		{
		case LayerTypes::Resampling:
		{
			auto resampling = dynamic_cast<Resampling*>(model->Layers[layerIndex].get());
			if (resampling)
			{
				*algorithm = resampling->Algorithm;
				*fH = resampling->FactorH;
				*fW = resampling->FactorW;
			}
		}
		break;

		case LayerTypes::LocalResponseNorm:
		{
			auto lrn = dynamic_cast<LocalResponseNorm*>(model->Layers[layerIndex].get());
			if (lrn)
			{
				*acrossChannels = lrn->AcrossChannels;
				*localSize = lrn->LocalSize;
				*alpha = lrn->Alpha;
				*beta = lrn->Beta;
				*k = lrn->K;
			}
		}
		break;

		case LayerTypes::Activation:
		{
			auto activation = dynamic_cast<Activation*>(model->Layers[layerIndex].get());
			if (activation)
			{
				*activationFunction = activation->ActivationFunction;
				*alpha = activation->Alpha;
				*beta = activation->Beta;
			}
		}
		break;

		case LayerTypes::BatchNorm:
		{
			auto bn = dynamic_cast<BatchNorm*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormMish:
		{
			auto bn = dynamic_cast<BatchNormActivation<Mish, LayerTypes::BatchNormMish>*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormMishDropout:
		{
			auto bn = dynamic_cast<BatchNormActivationDropout<Mish, LayerTypes::BatchNormMishDropout>*>(model->Layers[layerIndex].get());
			if (bn)
			{
				*scaling = bn->Scaling;
				*dropout = Float(1) - bn->Keep;
			}
		}
		break;

		case LayerTypes::BatchNormHardLogistic:
		{
			auto bn = dynamic_cast<BatchNormActivation<HardLogistic, LayerTypes::BatchNormHardLogistic>*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormHardSwish:
		{
			auto bn = dynamic_cast<BatchNormActivation<HardSwish, LayerTypes::BatchNormHardSwish>*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormHardSwishDropout:
		{
			auto bn = dynamic_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(model->Layers[layerIndex].get());
			if (bn)
			{
				*scaling = bn->Scaling;
				*dropout = Float(1) - bn->Keep;
			}
		}
		break;

		case LayerTypes::BatchNormRelu:
		{
			auto bn = dynamic_cast<BatchNormRelu*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormReluDropout:
		{
			auto bn = dynamic_cast<BatchNormActivationDropout<Relu, LayerTypes::BatchNormReluDropout>*>(model->Layers[layerIndex].get());
			if (bn)
			{
				*scaling = bn->Scaling;
				*dropout = Float(1) - bn->Keep;
			}
		}
		break;

		case LayerTypes::BatchNormSwish:
		{
			auto bn = dynamic_cast<BatchNormActivation<Swish, LayerTypes::BatchNormSwish>*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormSwishDropout:
		{
			auto bn = dynamic_cast<BatchNormActivationDropout<Swish, LayerTypes::BatchNormSwishDropout>*>(model->Layers[layerIndex].get());
			if (bn)
			{
				*scaling = bn->Scaling;
				*dropout = Float(1) - bn->Keep;
			}
		}
		break;

		case LayerTypes::BatchNormTanhExp:
		{
			auto bn = dynamic_cast<BatchNormActivation<TanhExp, LayerTypes::BatchNormTanhExp>*>(model->Layers[layerIndex].get());
			if (bn)
				*scaling = bn->Scaling;
		}
		break;

		case LayerTypes::BatchNormTanhExpDropout:
		{
			auto bn = dynamic_cast<BatchNormActivationDropout<TanhExp, LayerTypes::BatchNormTanhExpDropout>*>(model->Layers[layerIndex].get());
			if (bn)
			{
				*scaling = bn->Scaling;
				*dropout = Float(1) - bn->Keep;
			}
		}
		break;

		case LayerTypes::Dropout:
		{
			auto drop = dynamic_cast<Dropout*>(model->Layers[layerIndex].get());
			if (drop)
				*dropout = Float(1) - drop->Keep;
		}
		break;

		case LayerTypes::AvgPooling:
		{
			auto pool = dynamic_cast<AvgPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				*kernelH = pool->KernelH;
				*kernelW = pool->KernelW;
				*strideH = pool->StrideH;
				*strideW = pool->StrideW;
			}
		}
		break;

		case LayerTypes::MaxPooling:
		{
			auto pool = dynamic_cast<MaxPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				*kernelH = pool->KernelH;
				*kernelW = pool->KernelW;
				*strideH = pool->StrideH;
				*strideW = pool->StrideW;
			}
		}
		break;

		case LayerTypes::GlobalAvgPooling:
		{
			auto pool = dynamic_cast<GlobalAvgPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				*kernelH = pool->KernelH;
				*kernelW = pool->KernelW;
			}
		}
		break;

		case LayerTypes::GlobalMaxPooling:
		{
			auto pool = dynamic_cast<GlobalMaxPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				*kernelH = pool->KernelH;
				*kernelW = pool->KernelW;
			}
		}
		break;

		case LayerTypes::Convolution:
		{
			auto conv = dynamic_cast<Convolution*>(model->Layers[layerIndex].get());
			if (conv)
			{
				*groups = conv->Groups;
				*kernelH = conv->KernelH;
				*kernelW = conv->KernelW;
				*strideH = conv->StrideH;
				*strideW = conv->StrideW;
				*dilationH = conv->DilationH;
				*dilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::DepthwiseConvolution:
		{
			auto conv = dynamic_cast<DepthwiseConvolution*>(model->Layers[layerIndex].get());
			if (conv)
			{
				*multiplier = conv->Multiplier;
				*kernelH = conv->KernelH;
				*kernelW = conv->KernelW;
				*strideH = conv->StrideH;
				*strideW = conv->StrideW;
				*dilationH = conv->DilationH;
				*dilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::ConvolutionTranspose:
		{
			auto conv = dynamic_cast<ConvolutionTranspose*>(model->Layers[layerIndex].get());
			if (conv)
			{
				*kernelH = conv->KernelH;
				*kernelW = conv->KernelW;
				*strideH = conv->StrideH;
				*strideW = conv->StrideW;
				*dilationH = conv->DilationH;
				*dilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::ChannelShuffle:
		{
			auto channel = dynamic_cast<ChannelShuffle*>(model->Layers[layerIndex].get());
			if (channel)
				*groups = channel->Groups;
		}
		break;

		case LayerTypes::ChannelSplit:
		{
			auto channel = dynamic_cast<ChannelSplit*>(model->Layers[layerIndex].get());
			if (channel)
			{
				*group = channel->Group;
				*groups = channel->Groups;
			}
		}
		break;

		case LayerTypes::Cost:
		{
			auto loss = dynamic_cast<Cost*>(model->Layers[layerIndex].get());
			if (loss)
			{
				*cost = loss->CostFunction;
				*labelTrue = loss->LabelTrue;
				*labelFalse = loss->LabelFalse;
				*groupIndex = loss->GroupIndex;
				*labelIndex = loss->LabelIndex;
				*weight = loss->Weight;
			}
		}
		break;

		case LayerTypes::PartialDepthwiseConvolution:
		{
			auto conv = dynamic_cast<PartialDepthwiseConvolution*>(model->Layers[layerIndex].get());
			if (conv)
			{
				*group = conv->Group;
				*groups = conv->Groups;
				*multiplier = conv->Multiplier;
				*kernelH = conv->KernelH;
				*kernelW = conv->KernelW;
				*strideH = conv->StrideH;
				*strideW = conv->StrideW;
				*dilationH = conv->DilationH;
				*dilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::LayerNorm:
		{
			auto ln = dynamic_cast<LayerNorm*>(model->Layers[layerIndex].get());
			if (ln)
				*scaling = ln->Scaling;
		}
		break;

		default:
			return;
		}
	}
}

extern "C" DNN_API Optimizers GetOptimizer()
{
	if (model)
		return model->Optimizer;

	return Optimizers::SGD;
}

extern "C" DNN_API int DNNLoadWeights(const std::string& fileName, const bool persistOptimizer)
{
	if (model)
		return model->LoadWeights(fileName, persistOptimizer);
	
	return -10;
}

extern "C" DNN_API int DNNSaveWeights(const std::string& fileName, const bool persistOptimizer)
{
	if (model)
		return model->SaveWeights(fileName, persistOptimizer);
	
	return -10;
}

extern "C" DNN_API int DNNLoadLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer)
{
	if (model)
	{
		if (GetFileSize(fileName) == model->Layers[layerIndex]->GetWeightsSize(persistOptimizer, model->Optimizer))
			return model->LoadLayerWeights(fileName, layerIndex, persistOptimizer);
		else
			return -1;
	}
	
	return -10;
}

extern "C" DNN_API int DNNSaveLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer)
{
	if (model && layerIndex < model->Layers.size())
		return model->SaveLayerWeights(fileName, layerIndex, persistOptimizer);

	return -10;
}

extern "C" DNN_API void DNNSetLocked(const bool locked)
{
	if (model)
		model->SetLocking(locked);
}

extern "C" DNN_API void DNNSetLayerLocked(const UInt layerIndex, const bool locked)
{
	if (model)
		 model->SetLayerLocking(layerIndex, locked);
}