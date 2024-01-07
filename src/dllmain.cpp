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


extern "C" DNN_API void DNNSetNewEpochDelegate(void(*newEpoch)(UInt, UInt, UInt, UInt, Float, Float, Float, bool, bool, Float, Float, bool, Float, Float, UInt, Float, UInt, Float, Float, Float, UInt, UInt, UInt, UInt, UInt, UInt, UInt, Float, Float, Float, Float, Float, Float, UInt, Float, Float, Float, UInt, UInt))
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

extern "C" DNN_API Model* DNNModel(const std::string definition)
{
	model = std::make_unique<Model>(definition, dataprovider.get());

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

extern "C" DNN_API bool DNNSetShuffleCount(const UInt count)
{
	if (dataprovider)
	{
		if (count > 0ull)
		{
			dataprovider->ShuffleCount = count;
			return true;
		}
	}

	return false;
}

extern "C" DNN_API void DNNDataproviderDispose()
{
	if (dataprovider)
		dataprovider.reset();
}

extern "C" DNN_API bool DNNCheck(std::string& definition, CheckMsg& checkMsg)
{
	return Check(definition, checkMsg);
}

extern "C" DNN_API int DNNRead(const std::string& definition, CheckMsg& checkMsg)
{
	dnn::Model* ptr = nullptr;

	ptr = Read(definition, dataprovider.get(), checkMsg);

	if (ptr)
	{
		model.reset();
		model = std::unique_ptr<Model>(ptr);
		ptr = nullptr;

		return 1;
	}

	return 0;
}

extern "C" DNN_API int DNNLoad(const std::string& fileName, CheckMsg& checkMsg)
{
	dnn::Model* ptr = nullptr;

	ptr = Load(fileName, dataprovider.get(), checkMsg);
	
	if (ptr)
	{
		model.reset();
		model = std::unique_ptr<Model>(ptr);
		ptr = nullptr;

		return 1;
	}

	return 0;
}

extern "C" DNN_API bool DNNLoadModel(const std::string& fileName)
{
	if (model)
		return model->LoadModel(fileName);
	
	return false;
}

extern "C" DNN_API bool DNNSaveModel(const std::string& fileName)
{
	if (model)
		return model->SaveModel(fileName);

	return false;
}

extern "C" DNN_API bool DNNClearLog()
{
	if (model)
	{
		model->ClearLog();
		return true;
	}

	return false;
}

extern "C" DNN_API bool DNNLoadLog(const std::string& fileName)
{
	if (model)
		return model->LoadLog(fileName);

	return false;
}

extern "C" DNN_API bool DNNSaveLog(const std::string& fileName)
{
	if (model)
		return model->SaveLog(fileName);

	return false;
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

extern "C" DNN_API bool DNNBatchNormUsed()
{
	if (model)
		return model->BatchNormUsed();

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

extern "C" DNN_API void DNNSetUseTrainingStrategy(const bool enable)
{
	if (model)
		model->UseTrainingStrategy = enable;
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
		model->Layers[layerIndex]->ResetWeights(model->WeightsFiller, model->WeightsFillerMode, model->WeightsGain, model->WeightsScale, model->BiasesFiller, model->BiasesFillerMode, model->BiasesGain, model->BiasesScale);
}

extern "C" DNN_API void DNNGetImage(const UInt layerIndex, const Byte fillColor, Byte* image)
{
	if (model && layerIndex < model->Layers.size() && !model->BatchSizeChanging.load() && !model->ResettingWeights.load())
	{
		switch (model->Layers[layerIndex]->LayerType)
		{
			case LayerTypes::BatchNorm:
			case LayerTypes::BatchNormActivation:
			case LayerTypes::BatchNormActivationDropout:
			case LayerTypes::BatchNormRelu:
			case LayerTypes::Convolution:
			case LayerTypes::ConvolutionTranspose:
			case LayerTypes::Dense:
			case LayerTypes::DepthwiseConvolution:
			case LayerTypes::GroupNorm:
			case LayerTypes::LayerNorm:
			case LayerTypes::PartialDepthwiseConvolution:
			case LayerTypes::PRelu:
			{
				auto img = model->Layers[layerIndex]->GetImage(fillColor);
				std::memcpy(image, img.data(), img.size());
				img.release();
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
		if (model->TaskState.load() == TaskStates::Running && (model->State.load() == States::Training || model->State.load() == States::Testing))
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

extern "C" DNN_API void DNNAddTrainingRate(const TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples)
{
	if (model)
		model->AddTrainingRate(rate, clear, gotoEpoch, trainSamples);
}

extern "C" DNN_API void DNNAddTrainingRateSGDR(const TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt gotoCycle, const UInt trainSamples)
{
	if (model)
		model->AddTrainingRateSGDR(rate, clear, gotoEpoch, gotoCycle, trainSamples);
}

extern "C" DNN_API void DNNClearTrainingStrategies()
{
	if (model)
		model->TrainingStrategies = std::vector<TrainingStrategy>();
}

extern "C" DNN_API void DNNAddTrainingStrategy(const TrainingStrategy& strategy)
{
	if (model)
		model->TrainingStrategies.push_back(strategy);
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

extern "C" DNN_API void DNNGetCostInfo(const UInt index, CostInfo* info)
{
	if (model && index < model->CostLayers.size())
	{
		info->TrainErrors = model->CostLayers[index]->TrainErrors;
		info->TrainLoss = model->CostLayers[index]->TrainLoss;
		info->AvgTrainLoss = model->CostLayers[index]->AvgTrainLoss;
		info->TrainErrorPercentage = model->CostLayers[index]->TrainErrorPercentage;

		info->TestErrors = model->CostLayers[index]->TestErrors;
		info->TestLoss = model->CostLayers[index]->TestLoss;
		info->AvgTestLoss = model->CostLayers[index]->AvgTestLoss;
		info->TestErrorPercentage = model->CostLayers[index]->TestErrorPercentage;
	}
}

extern "C" DNN_API void DNNGetModelInfo(ModelInfo* info)
{
	if (model)
	{
		info->Name = model->Name;
		info->Dataset = dataprovider->Dataset;
		info->CostFunction = model->CostFunc;
		info->LayerCount = model->Layers.size();
		info->CostLayerCount = model->CostLayers.size();
		info->CostIndex = model->CostIndex;
		info->GroupIndex = model->GroupIndex;
		info->LabelIndex = model->LabelIndex;
		info->Hierarchies = dataprovider->Hierarchies;
		info->TrainSamplesCount = dataprovider->TrainSamplesCount;
		info->TestSamplesCount = dataprovider->TestSamplesCount;
		info->MeanStdNormalization = model->MeanStdNormalization;
		info->MeanTrainSet.clear();
		info->StdTrainSet.clear();
		
		switch (dataprovider->Dataset)
		{
		case Datasets::cifar10:
		case Datasets::cifar100:
		case Datasets::tinyimagenet:
			for (auto c = 0ull; c < 3ull; c++)
			{
				info->MeanTrainSet.push_back(dataprovider->Mean[c]);
				info->StdTrainSet.push_back(dataprovider->StdDev[c]);
			}
			break;
		case Datasets::fashionmnist:
		case Datasets::mnist:
			info->MeanTrainSet.push_back(dataprovider->Mean[0]);
			info->StdTrainSet.push_back(dataprovider->StdDev[0]);
			break;
		}
	}
}

extern "C" DNN_API void DNNGetLayerInfo(const UInt layerIndex, LayerInfo* info)
{
	if (model && layerIndex < model->Layers.size())
	{
		info->LayerIndex = layerIndex;
		info->Name = model->Layers[layerIndex]->Name;
		info->Description = model->Layers[layerIndex]->GetDescription();
		info->LayerType = model->Layers[layerIndex]->LayerType;
		info->Algorithm = Algorithms::Linear;
		info->InputsCount = model->Layers[layerIndex]->Inputs.size();
		info->NeuronCount = model->Layers[layerIndex]->CDHW();
		info->WeightCount = model->Layers[layerIndex]->WeightCount;
		info->BiasesCount = model->Layers[layerIndex]->BiasCount;
		info->Multiplier = 1;
		info->Groups = 1;
		info->Group = 1;
		info->C = model->Layers[layerIndex]->C;
		info->D = model->Layers[layerIndex]->D;
		info->H = model->Layers[layerIndex]->H;
		info->W = model->Layers[layerIndex]->W;
		info->PadD = model->Layers[layerIndex]->PadD;
		info->PadH = model->Layers[layerIndex]->PadH;
		info->PadW = model->Layers[layerIndex]->PadW;
		info->DilationH = 1;
		info->DilationW = 1;
		info->KernelH = 0;
		info->KernelW = 0;
		info->StrideH = 1;
		info->StrideW = 1;
		info->fH = 1;
		info->fW = 1;
		info->Dropout = Float(0);
		info->LabelTrue = Float(1);
		info->LabelFalse = Float(0);
		info->Weight = Float(1);
		info->GroupIndex = 0;
		info->LabelIndex = 0;
		info->InputC = model->Layers[layerIndex]->InputLayer != nullptr ? model->Layers[layerIndex]->InputLayer->C : 0;
		info->HasBias = model->Layers[layerIndex]->HasBias;
		info->Locked = model->Layers[layerIndex]->Lockable() ? model->Layers[layerIndex]->LockUpdate.load() : false;
		info->Lockable = model->Layers[layerIndex]->Lockable();

		switch (model->Layers[layerIndex]->LayerType)
		{
		case LayerTypes::Activation:
		{
			auto activation = dynamic_cast<Activation*>(model->Layers[layerIndex].get());
			if (activation)
			{
				info->Activation = activation->ActivationFunction;
				info->Alpha = activation->Alpha;
				info->Beta = activation->Beta;
			}
		}
		break;

		case LayerTypes::AvgPooling:
		{
			auto pool = dynamic_cast<AvgPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				info->KernelH = pool->KernelH;
				info->KernelW = pool->KernelW;
				info->StrideH = pool->StrideH;
				info->StrideW = pool->StrideW;
				info->DilationH = pool->DilationH;
				info->DilationW = pool->DilationW;
			}
		}
		break;

		case LayerTypes::BatchNorm:
		{
			auto bn = dynamic_cast<BatchNorm*>(model->Layers[layerIndex].get());
			if (bn)
			{
				info->Scaling = bn->Scaling;
			}
		}
		break;

		case LayerTypes::BatchNormActivation:
		{
			auto bn = dynamic_cast<BatchNormActivation*>(model->Layers[layerIndex].get());
			if (bn)
			{
				info->Scaling = bn->Scaling;
				info->Alpha = bn->Alpha;
				info->Beta = bn->Beta;
				info->Activation = bn->ActivationFunction;
			}
		}
		break;

		case LayerTypes::BatchNormActivationDropout:
		{
			auto bn = dynamic_cast<BatchNormActivationDropout*>(model->Layers[layerIndex].get());
			if (bn)
			{
				info->Scaling = bn->Scaling;
				info->Dropout = Float(1) - bn->Keep;
				info->Alpha = bn->Alpha;
				info->Beta = bn->Beta;
				info->Activation = bn->ActivationFunction;
			}
		}
		break;

		case LayerTypes::BatchNormRelu:
		{
			auto bn = dynamic_cast<BatchNormRelu*>(model->Layers[layerIndex].get());
			if (bn)
				info->Scaling = bn->Scaling;
		}
		break;

		case LayerTypes::ChannelSplit:
		{
			auto split = dynamic_cast<ChannelSplit*>(model->Layers[layerIndex].get());
			if (split)
			{
				info->Group = split->Group;
				info->Groups = split->Groups;
			}
		}
		break;

		case LayerTypes::Convolution:
		{
			auto conv = dynamic_cast<Convolution*>(model->Layers[layerIndex].get());
			if (conv)
			{
				info->Groups = conv->Groups;
				info->KernelH = conv->KernelH;
				info->KernelW = conv->KernelW;
				info->StrideH = conv->StrideH;
				info->StrideW = conv->StrideW;
				info->DilationH = conv->DilationH;
				info->DilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::ConvolutionTranspose:
		{
			auto conv = dynamic_cast<ConvolutionTranspose*>(model->Layers[layerIndex].get());
			if (conv)
			{
				info->KernelH = conv->KernelH;
				info->KernelW = conv->KernelW;
				info->StrideH = conv->StrideH;
				info->StrideW = conv->StrideW;
				info->DilationH = conv->DilationH;
				info->DilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::Cost:
		{
			auto cost = dynamic_cast<Cost*>(model->Layers[layerIndex].get());
			if (cost)
			{
				info->Cost = cost->CostFunction;
				info->LabelTrue = cost->LabelTrue;
				info->LabelFalse = cost->LabelFalse;
				info->GroupIndex = cost->GroupIndex;
				info->LabelIndex = cost->LabelIndex;
				info->Weight = cost->Weight;
			}
		}
		break;

		case LayerTypes::DepthwiseConvolution:
		{
			auto conv = dynamic_cast<DepthwiseConvolution*>(model->Layers[layerIndex].get());
			if (conv)
			{
				info->Multiplier = conv->Multiplier;
				info->KernelH = conv->KernelH;
				info->KernelW = conv->KernelW;
				info->StrideH = conv->StrideH;
				info->StrideW = conv->StrideW;
				info->DilationH = conv->DilationH;
				info->DilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::Dropout:
		{
			auto dropout = dynamic_cast<dnn::Dropout*>(model->Layers[layerIndex].get());
			if (dropout)
				info->Dropout = Float(1) - dropout->Keep;
		}
		break;

		case LayerTypes::GlobalAvgPooling:
		{
			auto pool = dynamic_cast<GlobalAvgPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				info->KernelH = pool->KernelH;
				info->KernelW = pool->KernelW;
			}
		}
		break;

		case LayerTypes::GlobalMaxPooling:
		{
			auto pool = dynamic_cast<GlobalMaxPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				info->KernelH = pool->KernelH;
				info->KernelW = pool->KernelW;
			}
		}
		break;

		case LayerTypes::GroupNorm:
		{
			auto gn = dynamic_cast<GroupNorm*>(model->Layers[layerIndex].get());
			if (gn)
			{
				info->Scaling = gn->Scaling;
			}
		}
		break;

		case LayerTypes::LayerNorm:
		{
			auto ln = dynamic_cast<LayerNorm*>(model->Layers[layerIndex].get());
			if (ln)
				info->Scaling = ln->Scaling;
		}
		break;

		case LayerTypes::LocalResponseNorm:
		{
			auto lrn = dynamic_cast<LocalResponseNorm*>(model->Layers[layerIndex].get());
			if (lrn)
			{
				info->AcrossChannels = lrn->AcrossChannels;
				info->LocalSize = lrn->LocalSize;
				info->Alpha = lrn->Alpha;
				info->Beta = lrn->Beta;
				info->K = lrn->K;
			}
		}
		break;

		case LayerTypes::MaxPooling:
		{
			auto pool = dynamic_cast<MaxPooling*>(model->Layers[layerIndex].get());
			if (pool)
			{
				info->KernelH = pool->KernelH;
				info->KernelW = pool->KernelW;
				info->StrideH = pool->StrideH;
				info->StrideW = pool->StrideW;
				info->DilationH = pool->DilationH;
				info->DilationW = pool->DilationW;
			}
		}
		break;

		case LayerTypes::PartialDepthwiseConvolution:
		{
			auto conv = dynamic_cast<PartialDepthwiseConvolution*>(model->Layers[layerIndex].get());
			if (conv)
			{
				info->Group = conv->Group;
				info->Groups = conv->Groups;
				info->Multiplier = conv->Multiplier;
				info->KernelH = conv->KernelH;
				info->KernelW = conv->KernelW;
				info->StrideH = conv->StrideH;
				info->StrideW = conv->StrideW;
				info->DilationH = conv->DilationH;
				info->DilationW = conv->DilationW;
			}
		}
		break;

		case LayerTypes::PRelu:
		{
			auto prelu = dynamic_cast<PRelu*>(model->Layers[layerIndex].get());
			if (prelu)
				info->Alpha = prelu->Alpha;
		}
		break;

		case LayerTypes::ReductionAvg:
		{
		}
		break;

		case LayerTypes::ReductionMax:
		{
		}
		break;

		case LayerTypes::Resampling:
		{
			auto resampling = dynamic_cast<Resampling*>(model->Layers[layerIndex].get());
			if (resampling)
			{
				info->Algorithm = resampling->Algorithm;
				info->fH = resampling->FactorH;
				info->fW = resampling->FactorW;
			}
		}
		break;
		
		case LayerTypes::Shuffle:
		{
			auto shuffle = dynamic_cast<Shuffle*>(model->Layers[layerIndex].get());
			if (shuffle)
				info->Groups = shuffle->Groups;
		}
		break;

		default:
			return;
		}
	}
}

extern "C" DNN_API void DNNGetResolution(UInt* N, UInt* C, UInt* D, UInt* H, UInt* W)
{
	if (model)
	{
		*N = model->N;
		*C = model->C;
		*D = model->D;
		*H = model->H;
		*W = model->W;
	}
}

extern "C" DNN_API void DNNRefreshStatistics(const UInt layerIndex, StatsInfo* info)
{
	if (model && layerIndex < model->Layers.size())
	{
		while (model->BatchSizeChanging.load() || model->ResettingWeights.load())
			std::this_thread::yield();

		if (model->Layers[layerIndex]->RefreshStatistics(model->N))
		{
			info->Description = model->Layers[layerIndex]->GetDescription();
			info->NeuronsStats = model->Layers[layerIndex]->NeuronsStats;
			info->WeightsStats = model->Layers[layerIndex]->WeightsStats;
			info->BiasesStats = model->Layers[layerIndex]->BiasesStats;
			info->FPropLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->Layers[layerIndex]->fpropTime).count()) / 1000;
			info->BPropLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->Layers[layerIndex]->bpropTime).count()) / 1000;
			info->UpdateLayerTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->Layers[layerIndex]->updateTime).count()) / 1000;
			info->FPropTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->fpropTime).count()) / 1000;
			info->BPropTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->bpropTime).count()) / 1000;
			info->UpdateTime = Float(std::chrono::duration_cast<std::chrono::microseconds>(model->updateTime).count()) / 1000;
			info->Locked = model->Layers[layerIndex]->Lockable() ? model->Layers[layerIndex]->LockUpdate.load() : false;
		}
		else
			model->StopTask();
	}
}

extern "C" DNN_API void DNNGetTrainingInfo(TrainingInfo* info)
{
	if (model)
	{
		const auto sampleIdx = model->SampleIndex + model->N;
		const auto costIdx = model->CostIndex;

		switch (model->State)
		{
		case States::Training:
		{
			const auto adjustedsampleIndex = sampleIdx > dataprovider->TrainSamplesCount ? dataprovider->TrainSamplesCount : sampleIdx;

			model->TrainLoss = model->CostLayers[costIdx]->TrainLoss;
			model->TrainErrors = model->CostLayers[costIdx]->TrainErrors;
			model->TrainErrorPercentage = Float(model->CostLayers[costIdx]->TrainErrors * 100) / adjustedsampleIndex;
			model->AvgTrainLoss = model->CostLayers[costIdx]->TrainLoss / adjustedsampleIndex;

			info->AvgTrainLoss = model->AvgTrainLoss;
			info->TrainErrorPercentage = model->TrainErrorPercentage;
			info->TrainErrors = model->TrainErrors;
		}
		break;

		case States::Testing:
		{
			const auto adjustedsampleIndex = sampleIdx > dataprovider->TestSamplesCount ? dataprovider->TestSamplesCount : sampleIdx;

			model->TestLoss = model->CostLayers[costIdx]->TestLoss;
			model->TestErrors = model->CostLayers[costIdx]->TestErrors;
			model->TestErrorPercentage = Float(model->CostLayers[costIdx]->TestErrors * 100) / adjustedsampleIndex;
			model->AvgTestLoss = model->CostLayers[costIdx]->TestLoss / adjustedsampleIndex;

			info->AvgTestLoss = model->AvgTestLoss;
			info->TestErrorPercentage = model->TestErrorPercentage;
			info->TestErrors = model->TestErrors;
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

		info->TotalCycles = model->TotalCycles;
		info->TotalEpochs = model->TotalEpochs;
		info->Cycle = model->CurrentCycle;
		info->Epoch = model->CurrentEpoch;
		info->SampleIndex = model->SampleIndex;

		info->Rate = model->CurrentTrainingRate.MaximumRate;
		info->Optimizer = model->Optimizer;

		info->Momentum = model->CurrentTrainingRate.Momentum;
		info->Beta2 = model->CurrentTrainingRate.Beta2;
		info->Gamma = model->CurrentTrainingRate.Gamma;
		info->L2Penalty = model->CurrentTrainingRate.L2Penalty;
		info->Dropout = model->CurrentTrainingRate.Dropout;

		info->BatchSize = model->N;
		info->Height = model->H;
		info->Width = model->W;
		info->PadH = model->PadH;
		info->PadW = model->PadW;

		info->HorizontalFlip = model->CurrentTrainingRate.HorizontalFlip;
		info->VerticalFlip = model->CurrentTrainingRate.VerticalFlip;
		info->InputDropout = model->CurrentTrainingRate.InputDropout;
		info->Cutout = model->CurrentTrainingRate.Cutout;
		info->CutMix = model->CurrentTrainingRate.CutMix;
		info->AutoAugment = model->CurrentTrainingRate.AutoAugment;
		info->ColorCast = model->CurrentTrainingRate.ColorCast;
		info->ColorAngle = model->CurrentTrainingRate.ColorAngle;
		info->Distortion = model->CurrentTrainingRate.Distortion;
		info->Interpolation = model->CurrentTrainingRate.Interpolation;
		info->Scaling = model->CurrentTrainingRate.Scaling;
		info->Rotation = model->CurrentTrainingRate.Rotation;

		info->SampleSpeed = model->SampleSpeed;
		info->State = model->State.load();
		info->TaskState = model->TaskState.load();
	}
}

extern "C" DNN_API void DNNGetTestingInfo(TestingInfo* info)
{
	if (model)
	{
		const auto sampleIdx = model->SampleIndex + model->N;
		const auto costIdx = model->CostIndex;
		const auto adjustedsampleIndex = sampleIdx > dataprovider->TestSamplesCount ? dataprovider->TestSamplesCount : sampleIdx;

		model->TestLoss = model->CostLayers[costIdx]->TestLoss;
		model->TestErrors = model->CostLayers[costIdx]->TestErrors;
		model->TestErrorPercentage = Float(model->CostLayers[costIdx]->TestErrors * 100) / adjustedsampleIndex;
		model->AvgTestLoss = model->CostLayers[costIdx]->TestLoss / adjustedsampleIndex;

		info->SampleIndex = model->SampleIndex;

		info->BatchSize = model->N;
		info->Height = model->H;
		info->Width = model->W;
		info->PadH = model->PadH;
		info->PadW = model->PadW;

		info->AvgTestLoss = model->AvgTestLoss;
		info->TestErrorPercentage = model->TestErrorPercentage;
		info->TestErrors = model->TestErrors;

		info->SampleSpeed = model->SampleSpeed;

		info->State = model->State.load();
		info->TaskState = model->TaskState.load();
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