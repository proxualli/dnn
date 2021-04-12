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

#include "Utils.h"
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
DNN_API void DNNGetLayerInfo(const UInt layerIndex, UInt* inputsCount, dnn::LayerTypes* layerType, dnn::Activations* activationFunction, dnn::Costs* cost, std::string* name, std::string* description, UInt* neuronCount, UInt* weightCount, UInt* biasesCount, UInt* multiplier, UInt* groups, UInt* group, UInt* localSize, UInt* c, UInt* d, UInt* h, UInt* w, UInt* kernelH, UInt* kernelW, UInt* strideH, UInt* strideW, UInt* dilationH, UInt* dilationW, UInt* padD, UInt* padH, UInt* padW, Float* dropout, Float* labelTrue, Float* labelFalse, Float* weight, UInt* groupIndex, UInt* labelIndex, UInt* inputC, Float* alpha, Float* beta, Float* k, Algorithms* algorithm, Float* fH, Float* fW, bool* hasBias, bool* scaling, bool* acrossChannels, bool* locked, bool* lockable);
DNN_API void DNNSetNewEpochDelegate(void(*newEpoch)(UInt, UInt, UInt, bool, bool, Float, Float, Float, Float, UInt, Float, UInt, Float, Float, Float, UInt, Float, Float, Float, Float, Float, UInt, Float, Float, Float, UInt));
DNN_API void DNNModelDispose();
DNN_API bool DNNBatchNormalizationUsed();
DNN_API void DNNSetOptimizersHyperParameters(const Float adaDeltaEps, const Float adaGradEps, const Float adamEps, const Float adamBeta2, const Float adamaxEps, const Float adamaxBeta2, const Float rmsPropEps, const Float radamEps, const Float radamBeta1, const Float radamBeta2);
DNN_API void DNNResetWeights();
DNN_API void DNNResetLayerWeights(const UInt layerIndex);
DNN_API void DNNAddLearningRate(const bool clear, const UInt gotoEpoch, const Float maximumRate, const UInt bachSize, const UInt cycles, const UInt epochs, const UInt epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const UInt decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const UInt interpolation, const Float scaling, const Float rotation);
DNN_API void DNNAddLearningRateSGDR(const bool clear, const UInt gotoEpoch, const Float maximumRate, const UInt bachSize, const UInt cycles, const UInt epochs, const UInt epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const UInt decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const UInt interpolation, const Float scaling, const Float rotation);
DNN_API bool DNNLoadDataset();
DNN_API void DNNTraining();
DNN_API void DNNStop();
DNN_API void DNNPause();
DNN_API void DNNResume();
DNN_API void DNNTesting();
DNN_API void DNNGetTrainingInfo(UInt* currentCycle, UInt* totalCycles, UInt* currentEpoch, UInt* totalEpochs, bool* verticalMirror, bool* horizontalMirror, Float* dropout, Float* cutout, Float* autoAugment, Float* colorCast, UInt* colorAngle, Float* distortion, UInt* interpolation, Float* scaling, Float* rotation, UInt* sampleIndex, UInt* batchSize, Float* rate, Float* momentum, Float* l2Penalty, Float* avgTrainLoss, Float* trainErrorPercentage, UInt* trainErrors, Float* avgTestLoss, Float* testErrorPercentage, UInt* testErrors, Float* sampleSpeed, dnn::States* networkState, dnn::TaskStates* taskState);
DNN_API void DNNGetTestingInfo(UInt* batchSize, UInt* sampleIndex, Float* avgTestLoss, Float* testErrorPercentage, UInt* testErrors, Float* sampleSpeed, dnn::States* networkState, dnn::TaskStates* taskState);
DNN_API void DNNGetNetworkInfo(std::string* name, UInt* costIndex, UInt* costLayerCount, UInt* groupIndex, UInt* labelIndex, UInt* hierarchies, bool* meanStdNormalization, dnn::Costs* lossFunction, dnn::Datasets* dataset, UInt* layerCount, UInt* trainingSamples, UInt* testingSamples, std::vector<Float>* meanTrainSet, std::vector<Float>* stdTrainSet);
DNN_API bool DNNSetFormat(const bool plain);
DNN_API void DNNSetOptimizer(const dnn::Optimizers strategy);
DNN_API void DNNRefreshStatistics(const UInt layerIndex, std::string* description, dnn::Stats* neuronsStats, dnn::Stats* weightsStats, dnn::Stats* biasesStats, Float* fpropLayerTime, Float* bpropLayerTime, Float* updateLayerTime, Float* fpropTime, Float* bpropTime, Float* updateTime, bool* locked);
DNN_API bool DNNGetInputSnapShot(std::vector<Float>* snapshot, std::vector<UInt>* label);
DNN_API bool DNNCheckDefinition(std::string& definition, dnn::CheckMsg& checkMsg);
DNN_API int DNNLoadDefinition(const std::string& fileName, const dnn::Optimizers optimizer, dnn::CheckMsg& checkMsg);
DNN_API int DNNReadDefinition(const std::string& definition, const dnn::Optimizers optimizer, dnn::CheckMsg& checkMsg);
DNN_API void DNNDataprovider(const std::string& directory);
DNN_API int DNNLoadNetworkWeights(const std::string& fileName, const bool persistOptimizer);
DNN_API int DNNSaveNetworkWeights(const std::string& fileName, const bool persistOptimizer);
DNN_API int DNNLoadLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer);
DNN_API int DNNSaveLayerWeights(const std::string& fileName, const UInt layerIndex, const bool persistOptimizer);
DNN_API void DNNGetLayerWeights(const UInt layerIndex, std::vector<Float>* weights, std::vector<Float>* biases);
DNN_API void DNNSetCostIndex(const UInt index);
DNN_API void DNNGetCostInfo(const UInt costIndex, UInt* trainErrors, Float* trainLoss, Float* avgTrainLoss, Float* trainErrorPercentage, UInt* testErrors, Float* testLoss, Float* avgTestLoss, Float* testErrorPercentage);
DNN_API void DNNGetImage(const UInt layer, const dnn::Byte fillColor, dnn::Byte* image);


void NewEpoch(UInt CurrentCycle, UInt CurrentEpoch, UInt TotalEpochs, bool HorizontalFlip, bool VerticalFlip, Float Dropout, Float Cutout, Float AutoAugment, Float ColorCast, UInt ColorAngle, Float Distortion, UInt Interpolation, Float Scaling, Float Rotation, Float MaximumRate, UInt BatchSize, Float Momentum, Float L2Penalty, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, UInt TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, UInt TestErrors)
{
    std::cout << "Cycle: " << std::to_string(CurrentCycle) << "  Epoch: " << std::to_string(CurrentEpoch) << "  Test Accuracy: " << FloatToStringFixed(TestAccuracy, 2) << std::string("%                                                                           ") << std::endl;
    std::cout.flush();

    DNN_UNREF_PAR(TotalEpochs);
    DNN_UNREF_PAR(HorizontalFlip);
    DNN_UNREF_PAR(VerticalFlip);
    DNN_UNREF_PAR(Dropout);
    DNN_UNREF_PAR(Cutout);
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
    DNN_UNREF_PAR(L2Penalty);
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
    auto cycle = new UInt();
    auto totalCycles = new UInt();
    auto epoch = new UInt();
    auto totalEpochs = new UInt();
    auto horizontalMirror = new bool();
    auto verticalMirror = new bool();
    auto dropout = new Float();
    auto cutout = new Float();
    auto autoAugment = new Float();
    auto colorCast = new Float();
    auto colorAngle = new UInt();
    auto distortion = new Float();
    auto interpolation = new UInt();
    auto scaling = new Float();
    auto rotation = new Float();
    auto sampleIndex = new UInt();
    auto rate = new Float();
    auto momentum = new Float();
    auto l2Penalty = new Float();
    auto batchSize = new UInt();
    auto avgTrainLoss = new Float();
    auto trainErrorPercentage = new Float();
    auto trainErrors = new UInt();
    auto avgTestLoss = new Float();
    auto testErrorPercentage = new Float();
    auto testErrors = new UInt();
    auto sampleSpeed = new Float();
    auto state = new States();
    auto taskState = new TaskStates();

    *state = States::Idle;
    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        DNNGetTrainingInfo(cycle, totalCycles, epoch, totalEpochs, horizontalMirror, verticalMirror, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, scaling, rotation, sampleIndex, batchSize, rate, momentum, l2Penalty, avgTrainLoss, trainErrorPercentage, trainErrors, avgTestLoss, testErrorPercentage, testErrors, sampleSpeed, state, taskState);
    } 
    while (*state == States::Idle);

    int barWidth = 40;
    float progress = 0.0;
  
    while (*state != dnn::States::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(*state == dnn::States::Testing ? 1 : seconds));
        
        DNNGetTrainingInfo(cycle, totalCycles, epoch, totalEpochs, horizontalMirror, verticalMirror, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, scaling, rotation, sampleIndex, batchSize, rate, momentum, l2Penalty, avgTrainLoss, trainErrorPercentage, trainErrors, avgTestLoss, testErrorPercentage, testErrors, sampleSpeed, state, taskState);
       
        if (*state == dnn::States::Testing)
            progress = Float(*sampleIndex) / testingSamples; 
        else
            progress = Float(*sampleIndex) / trainingSamples; 

        if (*state != dnn::States::Completed)
        {
            std::cout << "[";
            int pos = int(barWidth * progress);
            for (int i = 0; i < barWidth; ++i)
            {
                if (i < pos)
                    std::cout << "=";
                else
                    if (i == pos)
                        std::cout << ">";
                    else
                        std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "%  Cycle:" << std::to_string(*cycle) << "  Epoch:" << std::to_string(*epoch) << "  Error:";
            if (*state == dnn::States::Testing)
                std::cout << FloatToStringFixed(*testErrorPercentage, 2);
            else
                std::cout << FloatToStringFixed(*trainErrorPercentage, 2);
            std::cout << "%  " << FloatToStringFixed(*sampleSpeed, 2) << " samples/s   \r";
            std::cout.flush();
        }
    }
   
    delete cycle;
    delete totalCycles;
    delete epoch;
    delete totalEpochs;
    delete horizontalMirror;
    delete verticalMirror;
    delete dropout;
    delete cutout;
    delete autoAugment;
    delete colorCast;
    delete colorAngle;
    delete distortion;
    delete interpolation;
    delete scaling;
    delete rotation;
    delete sampleIndex;
    delete rate;
    delete momentum;
    delete l2Penalty;
    delete batchSize;
    delete avgTrainLoss;
    delete trainErrorPercentage;
    delete trainErrors;
    delete avgTestLoss;
    delete testErrorPercentage;
    delete testErrors;
    delete sampleSpeed;
    delete state;
    delete taskState;
}


#ifdef _WIN32
int __cdecl wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    CheckMsg msg;

    scripts::ScriptParameters p;

    p.Script = scripts::Scripts::shufflenetv2;
    p.Dataset = scripts::Datasets::cifar10;
    p.C = 3;
    p.H = 32;
    p.W = 32;
    p.PadH = 4;
    p.PadW = 4;
    p.MirrorPad = false;
    p.Groups = 3;
    p.Iterations = 6;
    p.Width = 10;
    p.Activation = scripts::Activations::HardSwish;
    p.Dropout = Float(0);
    p.Bottleneck = false;
    p.SqueezeExcitation = true;
    p.ChannelZeroPad = false;

    auto model = ScriptsCatalog::Generate(p);

    const auto optimizer = Optimizers::NAG;
    const auto persistOptimizer = true;
   
    DNNDataprovider(path);
    
    if (DNNReadDefinition(model, optimizer, msg) == 1)
    {
        if (DNNLoadDataset())
        {
            auto name = new std::string();
            auto costIndex = new UInt(); 
            auto costLayerCount = new UInt(); 
            auto groupIndex = new UInt(); 
            auto labelIndex = new UInt(); 
            auto hierarchies = new UInt(); 
            auto meanStdNormalization = new bool(); 
            auto lossFunction = new Costs(); 
            auto dataset = new Datasets(); 
            auto layerCount = new UInt(); 
            auto trainingSamples = new UInt(); 
            auto testingSamples = new UInt(); 
            auto meanTrainSet = new std::vector<Float>();
            auto stdTrainSet = new std::vector<Float>();
            
            DNNGetNetworkInfo(name, costIndex, costLayerCount, groupIndex, labelIndex, hierarchies, meanStdNormalization, lossFunction, dataset, layerCount, trainingSamples, testingSamples, meanTrainSet, stdTrainSet);
            std::cout << std::string("Training ") << *name << std::string(" on ") << std::string(magic_enum::enum_name<Datasets>(*dataset)) << std::string(" with " +  std::string(magic_enum::enum_name<Optimizers>(optimizer)) + " optimizer") << std::endl << std::endl;
            std::cout.flush();

            DNNSetNewEpochDelegate(&NewEpoch);
            DNNPersistOptimizer(persistOptimizer);
            DNNAddLearningRateSGDR(true, 1, 0.05f, 128, 1, 200, 1, 0.0001f, 0.0005f, 0.9f, 1.0f, 20, true, false, 0.0f, 0.7f, 0.7f, 0.7f, 20, 0.7f, 0, 10.0f, 12.0f);
            DNNTraining();

            GetTrainingProgress(1, *trainingSamples, *testingSamples);
            
            delete name;
            delete costIndex; 
            delete costLayerCount; 
            delete groupIndex; 
            delete labelIndex; 
            delete hierarchies; 
            delete meanStdNormalization; 
            delete lossFunction; 
            delete dataset; 
            delete layerCount; 
            delete trainingSamples; 
            delete testingSamples; 
            delete meanTrainSet;
            delete stdTrainSet;
        
            DNNStop();
        }
        else
            std::cout << std::endl << "Could not load dataset" << std::endl;
    }
    else
        std::cout << std::endl << "Could not load model" << std::endl << msg.Message << std::endl << model << std::endl;
}