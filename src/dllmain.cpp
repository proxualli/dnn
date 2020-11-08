#include "Definition.h"

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

using namespace dnn;

std::unique_ptr<dnn::Model> model;
std::unique_ptr<dnn::Dataprovider> dataprovider;

extern "C" DNN_API void DNNSetNewEpochDelegate(void(*newEpoch)(size_t, size_t, size_t, bool, bool, Float, Float, Float, Float, size_t, Float, size_t, Float, Float, Float, size_t, Float, Float, Float, Float, Float, size_t, Float, Float, Float, size_t))
{																	     
	if (model)
		model->NewEpoch = newEpoch;
}

extern "C" DNN_API void DNNModelDispose()
{
	if (model.get())
		model->~Model();
}

extern "C" DNN_API Model* DNNModel(const char* name)
{
	if (model.get())
		DNNModelDispose();

	model = std::make_unique<Model>(name, dataprovider.get());

	return model.get();
}

extern "C" DNN_API void DNNDataprovider(const char* directory)
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

extern "C" DNN_API int DNNReadDefinition(const char* definition, const Optimizers optimizer, CheckMsg& checkMsg)
{
	dnn::Model *ptr = nullptr;
	
	ptr = Definition::ReadDefinition(definition, optimizer, dataprovider.get(), checkMsg);

	if (ptr)
	{
		DNNModelDispose();

		model = std::unique_ptr<Model>(ptr);
	
		return 1;
	}

	return 0;
}

extern "C" DNN_API int DNNLoadDefinition(const char* fileName, const Optimizers optimizer, CheckMsg& checkMsg)
{
	dnn::Model *ptr = nullptr;
	
	ptr = Definition::LoadDefinition(fileName, optimizer, dataprovider.get(), checkMsg);

	if (ptr)
	{
		DNNModelDispose();

		model = std::unique_ptr<Model>(ptr);
		
		return 1;
	}

	return 0;
}

extern "C" DNN_API void DNNGetLayerInputs(const size_t layerIndex, std::vector<size_t> * inputs)
{
	if (model && layerIndex < model->Layers.size())
	{
		for (size_t i = 0; i < model->Layers[layerIndex]->Inputs.size(); i++)
		{
			auto inputLayerName = model->Layers[layerIndex]->Inputs[i]->Name;
			for (size_t index = 0; index < model->Layers.size(); index++)
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

extern "C" DNN_API void DNNGetConfusionMatrix(const size_t costLayerIndex, std::vector<std::vector<size_t>>* confusionMatrix)
{
	if (model && costLayerIndex < model->CostLayers.size())
		(*confusionMatrix) = model->CostLayers[costLayerIndex]->ConfusionMatrix;
}

extern "C" DNN_API void DNNSetOptimizersHyperParameters(const Float adaDeltaEps, const Float adaGradEps, const Float adamEps, const Float adamBeta2, const Float adamaxEps, const Float adamaxBeta2, const Float rmsPropEps, const Float radamEps, const Float radamBeta1, const Float radamBeta2)
{
	if (model)
		model->SetHyperParameters(adaDeltaEps, adaGradEps, adamEps, adamBeta2, adamaxEps, adamaxBeta2, rmsPropEps, radamEps, radamBeta1, radamBeta2);
}

extern "C" DNN_API void DNNPersistOptimizer(const bool persistOptimizer)
{
	if (model)
		model->PersistOptimizer = persistOptimizer;
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

extern "C" DNN_API void DNNResetLayerWeights(const size_t layerIndex)
{
	if (model && layerIndex < model->Layers.size())
		model->Layers[layerIndex]->ResetWeights(model->WeightsFiller, model->WeightsScale, model->BiasesFiller, model->BiasesScale);
}

extern "C" DNN_API void DNNGetImage(const size_t layerIndex, const unsigned char fillColor, unsigned char* image)
{
	if (model && layerIndex < model->Layers.size() && !model->BatchSizeChanging.load() && !model->ResettingWeights.load())
	{
		switch (model->Layers[layerIndex]->LayerType)
		{
			case LayerTypes::BatchNorm:
			case LayerTypes::BatchNormHardLogistic:
			case LayerTypes::BatchNormHardSwish:
			case LayerTypes::BatchNormHardSwishDropout:
			case LayerTypes::BatchNormRelu:
			case LayerTypes::BatchNormReluDropout:
			case LayerTypes::BatchNormSwish:
			case LayerTypes::Convolution:
			case LayerTypes::ConvolutionTranspose:
			case LayerTypes::Dense:
			case LayerTypes::DepthwiseConvolution:
			case LayerTypes::PartialDepthwiseConvolution:
			{
				ByteVector img = model->Layers[layerIndex]->GetImage(fillColor);
				std::memcpy(image, img.data(), img.size());
			}
			break;
		}
	}
}

extern "C" DNN_API bool DNNGetInputSnapShot(std::vector<Float>* snapshot, std::vector<size_t>* label)
{
	if (model)
		if ((model->TaskState.load() == TaskStates::Running) && ((model->State.load() == States::Training) || (model->State.load() == States::Testing)))
			return model->GetInputSnapShot(snapshot, label);

	return false;
}

extern "C" DNN_API void DNNGetLayerWeights(const size_t layerIndex, std::vector<Float>* weights, std::vector<Float>* biases)
{
	if (model)
		if (layerIndex < model->Layers.size() && model->Layers[layerIndex]->HasWeights)
		{
			for (size_t i = 0; i < model->Layers[layerIndex]->WeightCount; i++)
				(*weights)[i] = model->Layers[layerIndex]->Weights[i];
	
			if (model->Layers[layerIndex]->HasBias)
				for (size_t i = 0; i < model->Layers[layerIndex]->BiasCount; i++)
					(*biases)[i] = model->Layers[layerIndex]->Biases[i];
		}
}

extern "C" DNN_API void DNNSetOptimizer(const Optimizers optimizer)
{
	if (model)
		model->SetOptimizer(optimizer);
}

extern "C" DNN_API void DNNAddLearningRate(const bool clear, const size_t gotoEpoch, const Float maximumRate, const size_t batchSize, const size_t cycles, const size_t epochs, const size_t epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const size_t decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const size_t colorAngle, const Float distortion, const size_t interpolation, const Float maxScaling, const Float maxRotation)
{
	if (model)
		model->AddTrainingRate(TrainingRate(maximumRate, batchSize, cycles, epochs, epochMultiplier, minimumRate, L2penalty, momentum, decayFactor, decayAfterEpochs, horizontalFlip, verticalFlip, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, maxScaling, maxRotation), clear, gotoEpoch);
}

extern "C" DNN_API void DNNAddLearningRateSGDR(const bool clear, const size_t gotoEpoch, const Float maximumRate, const size_t batchSize, const size_t cycles, const size_t epochs, const size_t epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const size_t decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const size_t colorAngle, const Float distortion, const size_t interpolation, const Float maxScaling, const Float maxRotation)
{
	if (model)
		model->AddTrainingRateSGDR(TrainingRate(maximumRate, batchSize, cycles, epochs, epochMultiplier, minimumRate, L2penalty, momentum, decayFactor, decayAfterEpochs, horizontalFlip, verticalFlip, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, maxScaling, maxRotation), clear, gotoEpoch);
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

extern "C" DNN_API void DNNSetCostIndex(const size_t costLayerIndex)
{
	if (model && costLayerIndex < model->CostLayers.size())
		model->CostIndex = costLayerIndex;
}

extern "C" DNN_API void DNNGetCostInfo(const size_t costLayerIndex, size_t* trainErrors, Float* trainLoss, Float* avgTrainLoss, Float* trainErrorPercentage, size_t* testErrors, Float* testLoss, Float* avgTestLoss, Float* testErrorPercentage)
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

extern "C" DNN_API void DNNGetNetworkInfo(std::string* name, size_t* costIndex, size_t* costLayerCount, size_t* groupIndex, size_t* labelIndex, size_t* hierarchies, bool* meanStdNormalization, Costs* lossFunction, Datasets* dataset, size_t* layerCount, size_t* trainingSamples, size_t* testingSamples, std::vector<Float>* meanTrainSet, std::vector<Float>* stdTrainSet)
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
			for (size_t i = 0; i < 3; i++)
			{
				(*meanTrainSet).push_back(dataprovider->Mean[i]);
				(*stdTrainSet).push_back(dataprovider->StdDev[i]);
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

extern "C" DNN_API void DNNGetTrainingInfo(size_t* currentCycle, size_t* totalCycles, size_t* currentEpoch, size_t* totalEpochs, bool* horizontalFlip, bool* verticalFlip, Float* inputDropOut, Float* inputCutout, Float* autoAugment, Float* colorCast, size_t* colorAngle, Float* distortion, size_t* interpolation, Float* scaling, Float* rotation, size_t* sampleIndex, size_t* batchSize, Float* maximumRate, Float* momentum, Float* l2Penalty, Float* avgTrainLoss, Float* trainErrorPercentage, size_t* trainErrors, Float* avgTestLoss, Float* testErrorPercentage, size_t* testErrors, States* networkState, TaskStates* taskState)
{
	if (model)
	{
		switch (model->State)
		{
		case States::Training:
		{
			const size_t sampleIdx = ((model->SampleIndex + model->BatchSize) >= dataprovider->TrainingSamplesCount) ? dataprovider->TrainingSamplesCount : model->SampleIndex + model->BatchSize;

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
			const size_t sampleIdx = ((model->SampleIndex + model->BatchSize) >= dataprovider->TestingSamplesCount) ? dataprovider->TestingSamplesCount : model->SampleIndex + model->BatchSize;
						
			model->TestLoss = model->CostLayers[model->CostIndex]->TestLoss;
			model->TestErrors = model->CostLayers[model->CostIndex]->TestErrors;
			model->TestErrorPercentage = Float(model->TestErrors * 100) / sampleIdx;
			model->AvgTestLoss = model->TestLoss / sampleIdx;

			*avgTestLoss = model->AvgTestLoss;
			*testErrorPercentage = model->TestErrorPercentage;
			*testErrors = model->TestErrors;
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
		*inputDropOut = model->CurrentTrainingRate.Dropout;
		*inputCutout = model->CurrentTrainingRate.Cutout;
		*autoAugment = model->CurrentTrainingRate.AutoAugment;
		*colorCast = model->CurrentTrainingRate.ColorCast;
		*colorAngle = model->CurrentTrainingRate.ColorAngle;
		*distortion = model->CurrentTrainingRate.Distortion;
		*interpolation = model->CurrentTrainingRate.Interpolation;
		*scaling = model->CurrentTrainingRate.Scaling;
		*rotation = model->CurrentTrainingRate.Rotation;
		*maximumRate = model->CurrentTrainingRate.MaximumRate;
		*momentum = model->CurrentTrainingRate.Momentum;
		*l2Penalty = model->CurrentTrainingRate.L2Penalty;
		*batchSize = model->BatchSize;
		
		*networkState = model->State.load();
		*taskState = model->TaskState.load();
	}
}

extern "C" DNN_API void DNNGetTestingInfo(size_t* batchSize, size_t* sampleIndex, Float* avgTestLoss, Float* testErrorPercentage, size_t* testErrors, States* networkState, TaskStates* taskState)
{
	if (model)
	{
		const size_t sampleIdx = ((model->SampleIndex + model->BatchSize) >= dataprovider->TestingSamplesCount) ? dataprovider->TestingSamplesCount : model->SampleIndex + model->BatchSize;
					
		model->TestLoss = model->CostLayers[model->CostIndex]->TestLoss;
		model->TestErrors = model->CostLayers[model->CostIndex]->TestErrors;
		model->TestErrorPercentage = Float(model->TestErrors * 100) / sampleIdx;
		model->AvgTestLoss = model->TestLoss / sampleIdx;

		*batchSize = model->BatchSize;
		*sampleIndex = model->SampleIndex;
		*avgTestLoss = model->AvgTestLoss;
		*testErrorPercentage = model->TestErrorPercentage;
		*testErrors = model->TestErrors;
				
		*networkState = model->State.load();
		*taskState = model->TaskState.load();
	}
}

extern "C" DNN_API void DNNRefreshStatistics(const size_t layerIndex, std::string* description, Float* neuronsStdDev, Float* neuronsMean, Float* neuronsMin, Float* neuronsMax, Float* weightsStdDev, Float* weightsMean, Float* weightsMin, Float* weightsMax, Float* biasesStdDev, Float* biasesMean, Float* biasesMin, Float* biasesMax, Float* fpropLayerTime, Float* bpropLayerTime, Float* updateLayerTime, Float* fpropTime, Float* bpropTime, Float* updateTime, bool* locked)
{
	if (model)
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

		Layer* layer = model->Layers[layerIndex];
		
		*description = layer->GetDescription();

		*neuronsStdDev = layer->NeuronsStdDev;
		*neuronsMean = layer->NeuronsMean;
		*neuronsMin = layer->NeuronsMin;
		*neuronsMax = layer->NeuronsMax;

		*weightsStdDev = layer->WeightsStdDev;
		*weightsMean = layer->WeightsMean;
		*weightsMin = layer->WeightsMin;
		*weightsMax = layer->WeightsMax;
		*biasesStdDev = layer->BiasesStdDev;
		*biasesMean = layer->BiasesMean;
		*biasesMin = layer->BiasesMin;
		*biasesMax = layer->BiasesMax;

		*fpropLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(layer->fpropTime).count()) / 1000;
		*bpropLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(layer->bpropTime).count()) / 1000;
		*updateLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(layer->updateTime).count()) / 1000;
		*fpropTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->fpropTime).count()) / 1000;
		*bpropTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->bpropTime).count()) / 1000;
		*updateTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->updateTime).count()) / 1000;

		*locked = layer->Lockable() ? layer->LockUpdate.load() : false;
	}
}

extern "C" DNN_API void DNNGetLayerInfo(const size_t layerIndex, size_t* inputsCount, LayerTypes* layerType, Activations* activationFunction, Costs* cost, std::string* name, std::string* description, size_t* neuronCount, size_t* weightCount, size_t* biasesCount, size_t* multiplier, size_t* groups, size_t* group, size_t* localSize, size_t* c, size_t* d, size_t* h, size_t* w, size_t* kernelH, size_t* kernelW, size_t* strideH, size_t* strideW, size_t* dilationH, size_t* dilationW, size_t* padD, size_t* padH, size_t* padW, Float* dropout, Float* labelTrue, Float* labelFalse, Float* weight, size_t* groupIndex, size_t* labelIndex, size_t* inputC, Float* alpha, Float* beta, Float* k, Algorithms* algorithm, Float* fH, Float* fW, bool* hasBias, bool* scaling, bool* acrossChannels, bool* locked, bool* lockable)
{
	if (model && layerIndex < model->Layers.size())
	{
		Layer* layer = model->Layers[layerIndex];

		*inputsCount = layer->Inputs.size();
		*layerType = layer->LayerType;
		*name = layer->Name;
		*description = layer->GetDescription();
		*neuronCount = layer->CDHW;
		*weightCount = layer->WeightCount;
		*biasesCount = layer->BiasCount;
		*multiplier = 1;
		*groups = 1;
		*group = 1;
		*c = layer->C;
		*d = layer->D;
		*h = layer->H;
		*w = layer->W;
		*padD = layer->PadD;
		*padH = layer->PadH;
		*padW = layer->PadW;
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
		*labelTrue = Float(1.0);
		*labelFalse = Float(0.0);
		*weight = Float(1.0);
		*groupIndex = 0;
		*labelIndex = 0;
		*inputC = layer->InputLayer != nullptr ? layer->InputLayer->C : 0;
		*hasBias = layer->HasBias;
		*locked = layer->Lockable() ? layer->LockUpdate.load() : false;
		*lockable = layer->Lockable();

		switch (layer->LayerType)
		{
		case LayerTypes::Resampling:
			*algorithm = static_cast<Resampling*>(layer)->Algorithm;
			*fH = static_cast<Resampling*>(layer)->FactorH;
			*fW = static_cast<Resampling*>(layer)->FactorW;
			break;

		case LayerTypes::LocalResponseNormalization:
			*acrossChannels = static_cast<LocalResponseNormalization*>(layer)->AcrossChannels;
			*localSize = static_cast<LocalResponseNormalization*>(layer)->LocalSize;
			*alpha = static_cast<LocalResponseNormalization*>(layer)->Alpha;
			*beta = static_cast<LocalResponseNormalization*>(layer)->Beta;
			*k = static_cast<LocalResponseNormalization*>(layer)->K;
			break;

		case LayerTypes::Activation:
			*activationFunction = static_cast<Activation*>(layer)->ActivationFunction;
			*alpha = static_cast<Activation*>(layer)->Alpha;
			*beta = static_cast<Activation*>(layer)->Beta;
			break;

		case LayerTypes::BatchNorm:
			*scaling = static_cast<BatchNorm*>(layer)->Scaling;
			break;

		case LayerTypes::BatchNormHardLogistic:
			*scaling = static_cast<BatchNormActivation<HardLogistic, LayerTypes::BatchNormHardLogistic>*>(layer)->Scaling;
			break;

		case LayerTypes::BatchNormHardSwish:
			*scaling = static_cast<BatchNormActivation<HardSwish, LayerTypes::BatchNormHardSwish>*>(layer)->Scaling;
			break;

		case LayerTypes::BatchNormHardSwishDropout:
			*scaling = static_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer)->Scaling;
			*dropout = Float(1) - static_cast<BatchNormActivationDropout<HardSwish, LayerTypes::BatchNormHardSwishDropout>*>(layer)->Keep;
			break;

		case LayerTypes::BatchNormRelu:
			*scaling = static_cast<BatchNormRelu*>(layer)->Scaling;
			break;

		case LayerTypes::BatchNormReluDropout:
			*scaling = static_cast<BatchNormActivationDropout<Relu, LayerTypes::BatchNormReluDropout>*>(layer)->Scaling;
			*dropout = Float(1) - static_cast<BatchNormActivationDropout<Relu, LayerTypes::BatchNormReluDropout>*>(layer)->Keep;
			break;

		case LayerTypes::BatchNormSwish:
			*scaling = static_cast<BatchNormActivation<Swish, LayerTypes::BatchNormSwish>*>(layer)->Scaling;
			break;

		case LayerTypes::Dropout:
			*dropout = Float(1) - static_cast<Dropout*>(layer)->Keep;
			break;

		case LayerTypes::AvgPooling:
			*kernelH = static_cast<AvgPooling*>(layer)->KernelH;
			*kernelW = static_cast<AvgPooling*>(layer)->KernelW;
			*strideH = static_cast<AvgPooling*>(layer)->StrideH;
			*strideW = static_cast<AvgPooling*>(layer)->StrideW;
			break;

		case LayerTypes::MaxPooling:
			*kernelH = static_cast<MaxPooling*>(layer)->KernelH;
			*kernelW = static_cast<MaxPooling*>(layer)->KernelW;
			*strideH = static_cast<MaxPooling*>(layer)->StrideH;
			*strideW = static_cast<MaxPooling*>(layer)->StrideW;
			break;

		case LayerTypes::GlobalAvgPooling:
			*kernelH = static_cast<GlobalAvgPooling*>(layer)->KernelH;
			*kernelW = static_cast<GlobalAvgPooling*>(layer)->KernelW;
			break;

		case LayerTypes::GlobalMaxPooling:
			*kernelH = static_cast<GlobalMaxPooling*>(layer)->KernelH;
			*kernelW = static_cast<GlobalMaxPooling*>(layer)->KernelW;
			break;

		case LayerTypes::Convolution:
			*groups = static_cast<Convolution*>(layer)->Groups;
			*kernelH = static_cast<Convolution*>(layer)->KernelH;
			*kernelW = static_cast<Convolution*>(layer)->KernelW;
			*strideH = static_cast<Convolution*>(layer)->StrideH;
			*strideW = static_cast<Convolution*>(layer)->StrideW;
			*dilationH = static_cast<Convolution*>(layer)->DilationH;
			*dilationW = static_cast<Convolution*>(layer)->DilationW;
			break;

		case LayerTypes::DepthwiseConvolution:
			*multiplier = static_cast<DepthwiseConvolution*>(layer)->Multiplier;
			*kernelH = static_cast<DepthwiseConvolution*>(layer)->KernelH;
			*kernelW = static_cast<DepthwiseConvolution*>(layer)->KernelW;
			*strideH = static_cast<DepthwiseConvolution*>(layer)->StrideH;
			*strideW = static_cast<DepthwiseConvolution*>(layer)->StrideW;
			*dilationH = static_cast<DepthwiseConvolution*>(layer)->DilationH;
			*dilationW = static_cast<DepthwiseConvolution*>(layer)->DilationW;
			break;

		case LayerTypes::ConvolutionTranspose:
			*kernelH = static_cast<ConvolutionTranspose*>(layer)->KernelH;
			*kernelW = static_cast<ConvolutionTranspose*>(layer)->KernelW;
			*strideH = static_cast<ConvolutionTranspose*>(layer)->StrideH;
			*strideW = static_cast<ConvolutionTranspose*>(layer)->StrideW;
			*dilationH = static_cast<ConvolutionTranspose*>(layer)->DilationH;
			*dilationW = static_cast<ConvolutionTranspose*>(layer)->DilationW;
			break;

		case LayerTypes::ChannelShuffle:
			*groups = size_t(static_cast<ChannelShuffle*>(layer)->Groups);
			break;

		case LayerTypes::ChannelSplit:
			*group = size_t(static_cast<ChannelSplit*>(layer)->Group);
			*groups = size_t(static_cast<ChannelSplit*>(layer)->Groups);
			break;

		case LayerTypes::Cost:
			*cost = static_cast<Cost*>(layer)->CostFunction;
			*labelTrue = static_cast<Cost*>(layer)->LabelTrue;
			*labelFalse = static_cast<Cost*>(layer)->LabelFalse;
			*groupIndex = static_cast<Cost*>(layer)->GroupIndex;
			*labelIndex = static_cast<Cost*>(layer)->LabelIndex;
			*weight = static_cast<Cost*>(layer)->Weight;
			break;

		case LayerTypes::PartialDepthwiseConvolution:
			*group = static_cast<PartialDepthwiseConvolution*>(layer)->Group;
			*groups = static_cast<PartialDepthwiseConvolution*>(layer)->Groups;
			*multiplier = static_cast<PartialDepthwiseConvolution*>(layer)->Multiplier;
			*kernelH = static_cast<PartialDepthwiseConvolution*>(layer)->KernelH;
			*kernelW = static_cast<PartialDepthwiseConvolution*>(layer)->KernelW;
			*strideH = static_cast<PartialDepthwiseConvolution*>(layer)->StrideH;
			*strideW = static_cast<PartialDepthwiseConvolution*>(layer)->StrideW;
			*dilationH = static_cast<PartialDepthwiseConvolution*>(layer)->DilationH;
			*dilationW = static_cast<PartialDepthwiseConvolution*>(layer)->DilationW;
			break;
		}
	}
}

extern "C" DNN_API int DNNLoadNetworkWeights(const char* fileName, const bool persistOptimizer)
{
	if (model)
		return model->LoadWeights(fileName, persistOptimizer);
	
	return -10;
}

extern "C" DNN_API int DNNSaveNetworkWeights(const char* fileName, const bool persistOptimizer)
{
	if (model)
		return model->SaveWeights(fileName, persistOptimizer);
	
	return -10;
}

extern "C" DNN_API int DNNLoadLayerWeights(const char* fileName, const size_t layerIndex, const bool persistOptimizer)
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

extern "C" DNN_API int DNNSaveLayerWeights(const char* fileName, const size_t layerIndex, const bool persistOptimizer)
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

extern "C" DNN_API void DNNSetLayerLocked(const size_t layerIndex, const bool locked)
{
	if (model)
		 model->SetLayerLocking(layerIndex, locked);
}
