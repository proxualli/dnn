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
#include <chrono>

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
DNN_API bool DNNSetLayerLocked(const size_t layerIndex, const bool locked);
DNN_API void DNNPersistOptimizer(const bool persist);
DNN_API void DNNDisableLocking(const bool disable);
DNN_API void DNNGetConfusionMatrix(const size_t costLayerIndex, std::vector<std::vector<size_t>>* confusionMatrix);
DNN_API void DNNGetLayerInputs(const size_t layerIndex, std::vector<size_t>* inputs);
DNN_API void DNNGetLayerInfo(const size_t layerIndex, size_t* inputsCount, LayerTypes* layerType, Activations* activationFunction, Costs* cost, std::string* name, std::string* description, size_t* neuronCount, size_t* weightCount, size_t* biasesCount, size_t* multiplier, size_t* groups, size_t* group, size_t* localSize, size_t* c, size_t* d, size_t* h, size_t* w, size_t* kernelH, size_t* kernelW, size_t* strideH, size_t* strideW, size_t* dilationH, size_t* dilationW, size_t* padD, size_t* padH, size_t* padW, Float* dropout, Float* labelTrue, Float* labelFalse, Float* weight, size_t* groupIndex, size_t* labelIndex, size_t* inputC, Float* alpha, Float* beta, Float* k, Algorithms* algorithm, Float* fH, Float* fW, bool* hasBias, bool* scaling, bool* acrossChannels, bool* locked, bool* lockable);
DNN_API void DNNSetNewEpochDelegate(void(*newEpoch)(size_t, size_t, size_t, bool, bool, Float, Float, Float, Float, size_t, Float, size_t, Float, Float, Float, size_t, Float, Float, Float, Float, Float, size_t, Float, Float, Float, size_t));
DNN_API void DNNModelDispose();
DNN_API bool DNNBatchNormalizationUsed();
DNN_API void DNNSetOptimizersHyperParameters(const Float adaDeltaEps, const Float adaGradEps, const Float adamEps, const Float adamBeta2, const Float adamaxEps, const Float adamaxBeta2, const Float rmsPropEps, const Float radamEps, const Float radamBeta1, const Float radamBeta2);
DNN_API void DNNResetWeights();
DNN_API void DNNResetLayerWeights(const size_t layerIndex);
DNN_API void DNNAddLearningRate(const bool clear, const size_t gotoEpoch, const Float maximumRate, const size_t bachSize, const size_t cycles, const size_t epochs, const size_t epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const size_t decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const size_t colorAngle, const Float distortion, const size_t interpolation, const Float scaling, const Float rotation);
DNN_API void DNNAddLearningRateSGDR(const bool clear, const size_t gotoEpoch, const Float maximumRate, const size_t bachSize, const size_t cycles, const size_t epochs, const size_t epochMultiplier, const Float minimumRate, const Float L2penalty, const Float momentum, const Float decayFactor, const size_t decayAfterEpochs, const bool horizontalFlip, const bool verticalFlip, const Float dropout, const Float cutout, const Float autoAugment, const Float colorCast, const size_t colorAngle, const Float distortion, const size_t interpolation, const Float scaling, const Float rotation);
DNN_API bool DNNLoadDataset();
DNN_API void DNNTraining();
DNN_API void DNNStop();
DNN_API void DNNPause();
DNN_API void DNNResume();
DNN_API void DNNTesting();
DNN_API void DNNGetTrainingInfo(size_t* currentCycle, size_t* totalCycles, size_t* currentEpoch, size_t* totalEpochs, bool* verticalMirror, bool* horizontalMirror, Float* dropout, Float* cutout, Float* autoAugment, Float* colorCast, size_t* colorAngle, Float* distortion, size_t* interpolation, Float* scaling, Float* rotation, size_t* sampleIndex, size_t* batchSize, Float* rate, Float* momentum, Float* l2Penalty, Float* avgTrainLoss, Float* trainErrorPercentage, size_t* trainErrors, Float* avgTestLoss, Float* testErrorPercentage, size_t* testErrors, States* networkState, TaskStates* taskState);
DNN_API void DNNGetTestingInfo(size_t* batchSize, size_t* sampleIndex, Float* avgTestLoss, Float* testErrorPercentage, size_t* testErrors, States* networkState, TaskStates* taskState);
DNN_API void DNNGetNetworkInfo(std::string* name, size_t* costIndex, size_t* costLayerCount, size_t* groupIndex, size_t* labelindex, size_t* hierarchies, bool* meanStdNormalization, Costs* lossFunction, Datasets* dataset, size_t* layerCount, size_t* trainingSamples, size_t* testingSamples, std::vector<Float>* meanTrainSet, std::vector<Float>* stdTrainSet);
DNN_API void DNNSetOptimizer(const Optimizers strategy);
DNN_API void DNNRefreshStatistics(const size_t layerIndex, std::string* description, Float* neuronsStdDev, Float* neuronsMean, Float* neuronsMin, Float* neuronsMax, Float* weightsStdDev, Float* weightsMean, Float* weightsMin, Float* weightsMax, Float* biasesStdDev, Float* biasesMean, Float* biasesMin, Float* biasesMax, Float* fpropLayerTime, Float* bpropLayerTime, Float* updateLayerTime, Float* fpropTime, Float* bpropTime, Float* updateTime, bool* locked);
DNN_API bool DNNGetInputSnapShot(std::vector<Float>* snapshot, std::vector<size_t>* label);
DNN_API bool DNNCheckDefinition(std::string& definition, CheckMsg& checkMsg);
DNN_API int DNNLoadDefinition(const char* fileName, const Optimizers optimizer, CheckMsg& checkMsg);
DNN_API int DNNReadDefinition(const char* definition, const Optimizers optimizer, CheckMsg& checkMsg);
DNN_API void DNNDataprovider(const char* directory);
DNN_API int DNNLoadNetworkWeights(const char* fileName, const bool persistOptimizer);
DNN_API int DNNSaveNetworkWeights(const char* fileName, const bool persistOptimizer);
DNN_API int DNNLoadLayerWeights(const char* fileName, const size_t layerIndex, const bool persistOptimizer);
DNN_API int DNNSaveLayerWeights(const char* fileName, const size_t layerIndex, const bool persistOptimizer);
// DNN_API void DNNGetLayerWeights(const size_t layerIndex, std::vector<Float>* weights, std::vector<Float>* biases);
DNN_API void DNNSetCostIndex(const size_t index);
DNN_API void DNNGetCostInfo(const size_t costIndex, size_t* trainErrors, Float* trainLoss, Float* avgTrainLoss, Float* trainErrorPercentage, size_t* testErrors, Float* testLoss, Float* avgTestLoss, Float* testErrorPercentage);
DNN_API void DNNGetImage(const size_t layer, const unsigned char fillColor, unsigned char* image);

static size_t oldSampleIndex = 0;

void NewEpoch(size_t CurrentCycle, size_t CurrentEpoch, size_t TotalEpochs, bool HorizontalFlip, bool VerticalFlip, Float Dropout, Float Cutout, Float AutoAugment, Float ColorCast, size_t ColorAngle, Float Distortion, size_t Interpolation, Float Scaling, Float Rotation, Float MaximumRate, size_t BatchSize, Float Momentum, Float L2Penalty, Float AvgTrainLoss, Float TrainErrorPercentage, Float TrainAccuracy, size_t TrainErrors, Float AvgTestLoss, Float TestErrorPercentage, Float TestAccuracy, size_t TestErrors)
{
    std::cout << "Cycle: " << std::to_string(CurrentCycle) << "  Epoch: " << std::to_string(CurrentEpoch) << "  Test Accuracy: " << std::to_string(TestAccuracy) << std::string("                                                                                 ") << std::endl;
}

void GetProgress(int seconds = 10, size_t trainingSamples = 50000, size_t testingSamples = 10000)
{
    auto cycle = new size_t();
    auto totalCycles = new size_t();
    auto epoch = new size_t();
    auto totalEpochs = new size_t();
    auto horizontalMirror = new bool();
    auto verticalMirror = new bool();
    auto dropout = new Float();
    auto cutout = new Float();
    auto autoAugment = new Float();
    auto colorCast = new Float();
    auto colorAngle = new size_t();
    auto distortion = new Float();
    auto interpolation = new size_t();
    auto scaling = new Float();
    auto rotation = new Float();
    auto sampleIndex = new size_t();
    auto rate = new Float();
    auto momentum = new Float();
    auto l2Penalty = new Float();
    auto batchSize = new size_t();
    auto avgTrainLoss = new Float();
    auto trainErrorPercentage = new Float();
    auto trainErrors = new size_t();
    auto avgTestLoss = new Float();
    auto testErrorPercentage = new Float();
    auto testErrors = new size_t();
    auto state = new States();
    auto taskState = new TaskStates();

    *state = States::Idle;
    do
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        DNNGetTrainingInfo(cycle, totalCycles, epoch, totalEpochs, horizontalMirror, verticalMirror, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, scaling, rotation, sampleIndex, batchSize, rate, momentum, l2Penalty, avgTrainLoss, trainErrorPercentage, trainErrors, avgTestLoss, testErrorPercentage, testErrors, state, taskState);
    } 
    while (*state == States::Idle);

    int barWidth = 52;
    float progress = 0.0;
    while (*state != States::Completed)
    {
        std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock().now();
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        DNNGetTrainingInfo(cycle, totalCycles, epoch, totalEpochs, horizontalMirror, verticalMirror, dropout, cutout, autoAugment, colorCast, colorAngle, distortion, interpolation, scaling, rotation, sampleIndex, batchSize, rate, momentum, l2Penalty, avgTrainLoss, trainErrorPercentage, trainErrors, avgTestLoss, testErrorPercentage, testErrors, state, taskState);

        if (*state == States::Testing)
            progress = Float(*sampleIndex) / testingSamples; 
        else
            progress = Float(*sampleIndex) / trainingSamples; 

        if (oldSampleIndex > *sampleIndex)
            oldSampleIndex = 0;

        std::chrono::duration<Float> elapsedTime = std::chrono::high_resolution_clock().now() - startTime;
        const Float elapsedSeconds = Float(std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count()) / 1000000;
        const Float samplesPerSecond = Float(*sampleIndex - oldSampleIndex) / elapsedSeconds;

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%  Cycle:" << std::to_string(*cycle) << "  Epoch:" << std::to_string(*epoch) << "  Error:";
        if (*state == States::Testing)
            std::cout << std::to_string(*testErrorPercentage);
        else
            std::cout << std::to_string(*trainErrorPercentage);
        std::cout << "%  " << std::to_string(samplesPerSecond) << " samples/s   \r";
        std::cout.flush();

        if (*sampleIndex > oldSampleIndex)
           oldSampleIndex = SampleIndex;
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
    delete state;
    delete taskState;
}


int main()
{
    CheckMsg msg;

    ScriptParameters p;

    p.Script = Scripts::shufflenetv2;
    p.Dataset = Datasets::cifar10;
    p.C = 3;
    p.H = 32;
    p.W = 32;
    p.PadH = 4;
    p.PadW = 4;
    p.MirrorPad = false;
    p.Groups = 3;
    p.Iterations = 2;
    p.Width = 4;
    p.Relu = true;
    p.SqueezeExcitation = false;

    auto model = ScriptsCatalog::Generate(p);

    DNNDataprovider(path.c_str());
    if (DNNReadDefinition(model.c_str(), Optimizers::NAG, msg) == 1)
    {
        if (DNNLoadDataset())
        {
            DNNSetNewEpochDelegate(&NewEpoch);

            DNNAddLearningRateSGDR(true, 1, 0.05f, 128, 1, 200, 1, 0.0001f, 0.0005f, 0.9f, 1.0f, 200, true, false, 0.0f, 0.7f, 0.7f, 0.7f, 20, 0.7f, 0, 10.0f, 12.0f);
            DNNTraining();

            auto name = new std::string();
            auto costIndex = new size_t(); 
            auto costLayerCount = new size_t(); 
            auto groupIndex = new size_t(); 
            auto labelIndex = new size_t(); 
            auto hierarchies = new size_t(); 
            auto meanStdNormalization = new bool(); 
            auto lossFunction = new Costs(); 
            auto dataset = new Datasets(); 
            auto layerCount = new size_t(); 
            auto trainingSamples = new size_t(); 
            auto testingSamples = new size_t(); 
            auto meanTrainSet = new std::vector<Float>();
            auto stdTrainSet = new std::vector<Float>();
            
            DNNGetNetworkInfo(name, costIndex, costLayerCount, groupIndex, labelIndex, hierarchies, meanStdNormalization, lossFunction, dataset, layerCount, trainingSamples, testingSamples, meanTrainSet, stdTrainSet);

            GetProgress(5, *trainingSamples, *testingSamples);
            
            DNNStop();

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

            DNNModelDispose();
        }
        else
            std::cout << std::endl << "Could not load dataset" << std::endl;
    }
    else
        std::cout << std::endl <<  "Could not load model" << std::endl << msg.Message << std::endl << model << std::endl;
}