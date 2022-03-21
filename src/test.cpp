#ifndef _WIN32
  #include <stdlib.h>
  #define DNN_API extern "C" 
#else
#ifdef DNN_DLL
  #define DNN_API extern "C" __declspec(dllimport)
#else
  #define DNN_API extern "C"
#endif
#endif

#include "Model.h"
#include "Scripts.h"

#ifdef _WIN32
static std::string path = std::string(getenv("USERPROFILE")) + std::string("\\Documents\\convnet\\");
#else
static std::string path = std::string(getenv("HOME")) + std::string("/convnet/");
#endif

using namespace dnn;

DNN_API bool DNNStochasticEnabled();
DNN_API void DNNSetLocked(const bool locked);
DNN_API bool DNNSetLayerLocked(const UInt layerIndex, const bool locked);
DNN_API void DNNPersistOptimizer(const bool persist);
DNN_API void DNNDisableLocking(const bool disable);
DNN_API void DNNGetConfusionMatrix(const UInt costLayerIndex, std::vector<std::vector<UInt>>* confusionMatrix);
DNN_API void DNNGetLayerInputs(const UInt layerIndex, std::vector<UInt>* inputs);
DNN_API void DNNGetLayerInfo(const UInt layerIndex, dnn::LayerInfo* info);
DNN_API void DNNSetNewEpochDelegate(void(*newEpoch)(UInt, UInt, UInt, UInt, Float, Float, Float, bool, bool, Float, Float, bool, Float, Float, UInt, Float, UInt, Float, Float, Float, UInt, UInt, UInt, Float, Float, Float, Float, Float, Float, UInt, Float, Float, Float, UInt));
DNN_API void DNNModelDispose();
DNN_API bool DNNBatchNormalizationUsed();
DNN_API void DNNResetWeights();
DNN_API void DNNResetLayerWeights(const UInt layerIndex);
DNN_API void DNNAddTrainingRate(const dnn::TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples);
DNN_API void DNNAddTrainingRateSGDR(const dnn::TrainingRate& rate, const bool clear, const UInt gotoEpoch, const UInt trainSamples);
DNN_API void DNNClearTrainingStrategies();
DNN_API void DNNSetUseTrainingStrategy(const bool enable);
DNN_API void DNNAddTrainingStrategy(const dnn::TrainingStrategy& strategy);
DNN_API bool DNNLoadDataset();
DNN_API void DNNDataproviderDispose();
DNN_API void DNNTraining();
DNN_API void DNNStop();
DNN_API void DNNPause();
DNN_API void DNNResume();
DNN_API void DNNTesting();
DNN_API void DNNGetTrainingInfo(dnn::TrainingInfo* info);
DNN_API void DNNGetTestingInfo(dnn::TestingInfo* info);
DNN_API void DNNGetModelInfo(dnn::ModelInfo* info);
DNN_API void DNNSetOptimizer(const dnn::Optimizers strategy);
DNN_API void DNNResetOptimizer();
DNN_API void DNNRefreshStatistics(const UInt layerIndex, dnn::StatsInfo* info);
DNN_API bool DNNGetInputSnapShot(std::vector<Float>* snapshot, std::vector<UInt>* label);
DNN_API bool DNNCheck(std::string& definition, dnn::CheckMsg& checkMsg);
DNN_API int DNNLoad(const std::string& fileName, dnn::CheckMsg& checkMsg);
DNN_API int DNNRead(const std::string& definition, dnn::CheckMsg& checkMsg);
DNN_API void DNNDataprovider(const std::string& directory);
DNN_API int DNNLoadWeights(const std::string& fileName, const bool persistOptimizer);
DNN_API int DNNSaveWeights(const std::string& fileName, const bool persistOptimizer);
DNN_API int DNNLoadLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer);
DNN_API int DNNSaveLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer);
DNN_API void DNNGetLayerWeights(const UInt layerIndex, std::vector<Float>* weights, std::vector<Float>* biases);
DNN_API void DNNSetCostIndex(const UInt index);
DNN_API void DNNGetCostInfo(const UInt costIndex, dnn::CostInfo* info);
DNN_API void DNNGetImage(const UInt layer, const Byte fillColor, Byte* image);
DNN_API bool DNNSetFormat(const bool plain);
DNN_API dnn::Optimizers GetOptimizer();
//DNN_API void DNNPrintModel(const std::string& fileName);


void NewEpoch(UInt CurrentCycle, UInt CurrentEpoch, UInt TotalEpochs, UInt Optimizer, Float Beta2, Float Gamma, Float Eps, bool HorizontalFlip, bool VerticalFlip, Float InputDropout, Float Cutout, bool CutMix, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, UInt Interpolation, Float Scaling, Float Rotation, Float MaximumRate, UInt BatchSize, UInt Height, UInt Width, Float Momentum, Float L2Penalty, Float Dropout, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors)
{
    std::cout << std::string("Cycle: ") << std::to_string(CurrentCycle) << std::string("  Epoch: ") << std::to_string(CurrentEpoch) << std::string("  Test Accuracy: ") << FloatToStringFixed(TestAccuracy, 2) << std::string("%                                                                           ") << std::endl;
    std::cout.flush();

    DNN_UNREF_PAR(TotalEpochs);
    DNN_UNREF_PAR(Optimizer);
    DNN_UNREF_PAR(Beta2);
    DNN_UNREF_PAR(Eps);
    DNN_UNREF_PAR(HorizontalFlip);
    DNN_UNREF_PAR(VerticalFlip);
    DNN_UNREF_PAR(InputDropout);
    DNN_UNREF_PAR(Cutout);
    DNN_UNREF_PAR(CutMix);
    DNN_UNREF_PAR(AutoAugment);
    DNN_UNREF_PAR(ColorCast);
    DNN_UNREF_PAR(ColorAngle);
    DNN_UNREF_PAR(Distortion);
    DNN_UNREF_PAR(Interpolation);
    DNN_UNREF_PAR(Scaling);
    DNN_UNREF_PAR(Rotation);
    DNN_UNREF_PAR(MaximumRate);
    DNN_UNREF_PAR(BatchSize);
    DNN_UNREF_PAR(Momentum);
    DNN_UNREF_PAR(Height);
    DNN_UNREF_PAR(Width);
    DNN_UNREF_PAR(L2Penalty);
    DNN_UNREF_PAR(Gamma);
    DNN_UNREF_PAR(Dropout);
    DNN_UNREF_PAR(AvgTrainLoss);
    DNN_UNREF_PAR(TrainErrorPercentage);
    DNN_UNREF_PAR(TrainAccuracy);
    DNN_UNREF_PAR(TrainErrors);
    DNN_UNREF_PAR(AvgTestLoss);
    DNN_UNREF_PAR(TestErrorPercentage);
    DNN_UNREF_PAR(TestErrors);
}

void GetTrainingProgress(int seconds = 5, UInt trainingSamples = 50000, UInt testingSamples = 10000)
{
    auto info = new dnn::TrainingInfo();
   
    info->State = States::Idle;
    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        DNNGetTrainingInfo(info);
    } 
    while (info->State == States::Idle);

    int barWidth = 40;
    float progress = 0.0;
  
    while (info->State != States::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        
        DNNGetTrainingInfo(info);
       
        if (info->State == States::Testing || info->State == States::Training)
        {
            if (info->State == States::Testing)
                progress = Float(info->SampleIndex) / testingSamples;
            else
                progress = Float(info->SampleIndex) / trainingSamples;

            std::cout << std::string("[");
            int pos = int(barWidth * progress);
            for (int i = 0; i < barWidth; ++i)
            {
                if (i < pos)
                    std::cout << std::string("=");
                else
                    if (i == pos)
                        std::cout << std::string(">");
                    else
                        std::cout << std::string(" ");
            }
            std::cout << std::string("] ") << FloatToStringFixed(progress * 100.0f, 2) << std::string("%  Cycle:") << std::to_string(info->Cycle) << std::string("  Epoch:") << std::to_string(info->Epoch) << std::string("  Error:");

            if (info->State == States::Testing)
                std::cout << FloatToStringFixed(info->TestErrorPercentage, 2);
            else
                std::cout << FloatToStringFixed(info->TrainErrorPercentage, 2);

            std::cout << std::string("%  ") << FloatToStringFixed(info->SampleSpeed, 2) << std::string(" samples/s   \r");
            std::cout.flush();
        }
    }
   
    delete info;
}


#ifdef _WIN32
int __cdecl wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    DNN_UNREF_PAR(argc);
    DNN_UNREF_PAR(argv);

    CheckMsg msg;

    scripts::ScriptParameters p;

    p.Script = scripts::Scripts::mobilenetv3;
    p.Dataset = scripts::Datasets::cifar10;
    p.C = 3;
    p.H = 32;
    p.W = 32;
    p.PadH = 8;
    p.PadW = 8;
    p.MirrorPad = false;
    p.Groups = 3;
    p.Iterations = 4;
    p.Width = 4;
    p.Activation = scripts::Activations::HardSwish;
    p.Dropout = Float(0);
    p.Bottleneck = false;
    p.SqueezeExcitation = true;
    p.ChannelZeroPad = false;
    p.EfficientNet = { { 1, 24, 2, 1, false }, { 4, 48, 4, 2, false }, { 4, 64, 4, 2, false }, { 4, 128, 6, 2, true }, { 6, 160, 9, 1, true }, { 6, 256, 15, 2, true } };
    p.ShuffleNet = { { 7, 3, 1, 2, false }, { 7, 3, 1, 2, true }, { 7, 3, 1, 2, true } };
    p.WeightsFiller = scripts::Fillers::HeNormal;
    p.WeightsFillerMode = scripts::FillerModes::In;
    p.StrideHFirstConv = 1;
    p.StrideWFirstConv = 1;

    auto model = scripts::ScriptsCatalog::Generate(p);

    const auto optimizer = Optimizers::NAG;
    const auto persistOptimizer = true;
       
    dnn::TrainingRate rate;

    rate.Optimizer = dnn::Optimizers::NAG;
    rate.Momentum = 0.9f;
    rate.Beta2 = 0.999f;
    rate.L2Penalty = 0.0005f;
    rate.Dropout = 0.0f;
    rate.Eps = 0.00001f,
    rate.BatchSize = 128;
    rate.Height = 32;
    rate.Width = 32;
    rate.PadH = 8;
    rate.PadW = 8;
    rate.Cycles = 1;
    rate.Epochs = 200;
    rate.EpochMultiplier = 1;
    rate.MaximumRate = 0.05;
    rate.MinimumRate = 0.0001f;
    rate.FinalRate = 0.1f;
    rate.Gamma = 0.003f;
    rate.DecayAfterEpochs = 200;
    rate.DecayFactor = 1.0f;
    rate.HorizontalFlip = true;
    rate.VerticalFlip = false;
    rate.InputDropout = 0.0f;
    rate.Cutout = 0.7f;
    rate.CutMix = true;
    rate.AutoAugment = 0.7f;
    rate.Distortion = 0.7f;
    rate.Interpolation = dnn::Interpolations::Cubic;
    rate.Scaling = 10.0f;
    rate.Rotation = 12.0f;
    
    DNNDataprovider(path);
    
    if (DNNRead(model, msg) == 1)
    {
        if (DNNLoadDataset())
        {
            //DNNPrintModel(path + "Normal.txt");
            auto info = new ModelInfo();
            DNNGetModelInfo(info);

            std::cout << std::string("Training ") << info->Name << std::string(" on ") << std::string(magic_enum::enum_name<Datasets>(info->Dataset)) << std::string(" with ") +  std::string(magic_enum::enum_name<Optimizers>(optimizer)) + std::string(" optimizer") << std::endl << std::endl;
            std::cout.flush();

            DNNSetNewEpochDelegate(&NewEpoch);
            DNNPersistOptimizer(persistOptimizer);
            DNNAddTrainingRateSGDR(rate, true, 1, info->TrainingSamplesCount);
            DNNTraining();

            GetTrainingProgress(5, info->TrainingSamplesCount, info->TestingSamplesCount);
            
            delete info;
                   
            DNNStop();
        }
        else
            std::cout << std::endl << std::string("Could not load dataset") << std::endl;
    }
    else
        std::cout << std::endl << std::string("Could not load model") << std::endl << msg.Message << std::endl << model << std::endl;

    DNNDataproviderDispose();
}