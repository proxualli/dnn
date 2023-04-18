#pragma once
#include "Activation.h"
#include "Add.h"
#include "Average.h"
#include "AvgPooling.h"
#include "BatchNorm.h"
#include "BatchNormActivation.h"
#include "BatchNormActivationDropout.h"
#include "BatchNormRelu.h"
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
#include "Shuffle.h"
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
		UInt PadH;
		UInt PadW;
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

		bool operator==(const TrainingRate& o) const
		{
			return 
				std::tie(Epochs, BatchSize, Height, Width, PadH, PadW, Momentum, Beta2, Gamma, L2Penalty, Dropout, HorizontalFlip, VerticalFlip, InputDropout, Cutout, CutMix, AutoAugment, ColorCast, ColorAngle, Distortion, Interpolation, Scaling, Rotation) 
				== 
				std::tie(o.Epochs, o.BatchSize, o.Height, o.Width, o.PadH, o.PadW, o.Momentum, o.Beta2, o.Gamma, o.L2Penalty, o.Dropout, o.HorizontalFlip, o.VerticalFlip, o.InputDropout, o.Cutout, o.CutMix, o.AutoAugment, o.ColorCast, o.ColorAngle, o.Distortion, o.Interpolation, o.Scaling, o.Rotation);
		}

		TrainingStrategy() :
			Epochs(Float(1)),
			BatchSize(128),
			Height(32),
			Width(32),
			PadH(4),
			PadW(4),
			Momentum(Float(0.9)),
			Beta2(Float(0.999)),
			Gamma(Float(0.9)),
			L2Penalty(Float(0.0005)),
			Dropout(Float(0)),
			HorizontalFlip(true),
			VerticalFlip(false),
			InputDropout(Float(0)),
			Cutout(Float(0)),
			CutMix(false),
			AutoAugment(Float(0)),
			ColorCast(Float(0)),
			Distortion(Float(0)),
			Interpolation(Interpolations::Cubic),
			Scaling(Float(10)),
			Rotation(Float(12))
		{
		}

		TrainingStrategy(const Float epochs, const UInt batchSize, const UInt height, const UInt width, const UInt padH, const UInt padW, const Float momentum, const Float beta2, const Float gamma, const Float l2penalty, const Float dropout, const bool horizontalFlip, const bool verticalFlip, const Float inputDropout, const Float cutout, const bool cutMix, const Float autoAugment, const Float colorCast, const UInt colorAngle, const Float distortion, const Interpolations interpolation, const Float scaling, const Float rotation) :
			Epochs(epochs),
			BatchSize(batchSize),
			Height(height),
			Width(width),
			PadH(padH),
			PadW(padW),
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
			if (Epochs <= 0 || Epochs > 1)
				throw std::invalid_argument("Epochs out of range in TrainingStrategy");
			if (BatchSize == 0)
				throw std::invalid_argument("BatchSize cannot be zero in TrainingStrategy");
			if (Height == 0)
				throw std::invalid_argument("Height cannot be zero in TrainingStrategy");
			if (Width == 0)
				throw std::invalid_argument("Width cannot be zero in TrainingStrategy");
			if (PadH == 0)
				throw std::invalid_argument("PadH cannot be zero in TrainingStrategy");
			if (PadW == 0)
				throw std::invalid_argument("PadW cannot be zero in TrainingStrategy");
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

	/*
	private:
		friend bitsery::Access;
		template<typename S>
		void serialize(S& s, TrainingStrategy& o)
		{
			s.ext(*this, bitsery::ext::Growable{}, [](S& s, TrainingStrategy& o)
			{
				s.value4b(o.Epochs);
				s.value8b(o.BatchSize);
				s.value8b(o.Height);
				s.value8b(o.Width);
				s.value8b(o.PadH);
				s.value8b(o.PadW);
				s.value4b(o.Momentum);
				s.value4b(o.Beta2);
				s.value4b(o.Gamma);
				s.value4b(o.L2Penalty);
				s.value4b(o.Dropout);
				s.boolValue(o.HorizontalFlip);
				s.boolValue(o.VerticalFlip);
				s.value4b(o.InputDropout);
				s.value4b(o.Cutout);
				s.boolValue(o.CutMix);
				s.value4b(o.AutoAugment);
				s.value4b(o.ColorCast);
				s.value8b(o.ColorAngle);
				s.value4b(o.Distortion);
				s.value4b(o.Interpolation);
				s.value4b(o.Scaling);
				s.value4b(o.Rotation);
			});
		}
	*/

	};

	struct TrainingInfo
	{
		UInt TotalCycles;
		UInt TotalEpochs;
		UInt Cycle;
		UInt Epoch;
		UInt SampleIndex;
		Float Rate;
		Optimizers Optimizer;
		Float Momentum;
		Float Beta2;
		Float Gamma;
		Float L2Penalty;
		Float Dropout;
		UInt BatchSize;
		UInt Height;
		UInt Width;
		UInt PadH;
		UInt PadW;
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
		Float AvgTrainLoss;
		Float TrainErrorPercentage;
		UInt TrainErrors;
		Float AvgTestLoss;
		Float TestErrorPercentage;
		UInt TestErrors;
		Float SampleSpeed;
		States State;
		TaskStates TaskState;
	};

	struct TestingInfo
	{
		UInt TotalCycles;
		UInt TotalEpochs;
		UInt Cycle;
		UInt Epoch;
		UInt SampleIndex;
		UInt BatchSize;
		UInt Height;
		UInt Width;
		UInt PadH;
		UInt PadW;
		Float AvgTestLoss;
		Float TestErrorPercentage;
		UInt TestErrors;
		Float SampleSpeed;
		States State;
		TaskStates TaskState;
	};

	struct LayerInfo
	{
		std::string Name;
		std::string Description;
		LayerTypes LayerType;
		Activations Activation;
		Algorithms Algorithm;
		Costs Cost;
		UInt NeuronCount;
		UInt WeightCount;
		UInt BiasesCount;
		UInt LayerIndex;
		UInt InputsCount;
		UInt C;
		UInt D;
		UInt H;
		UInt W;
		UInt PadD;
		UInt PadH;
		UInt PadW;
		UInt KernelH;
		UInt KernelW;
		UInt StrideH;
		UInt StrideW;
		UInt DilationH;
		UInt DilationW;
		UInt Multiplier;
		UInt Groups;
		UInt Group;
		UInt LocalSize;
		Float Dropout;
		Float LabelTrue;
		Float LabelFalse;
		Float Weight;
		UInt GroupIndex;
		UInt LabelIndex;
		UInt InputC;
		Float Alpha;
		Float Beta;
		Float K;
		Float fH;
		Float fW;
		bool HasBias;
		bool Scaling;
		bool AcrossChannels;
		bool Locked;
		bool Lockable;
	};

	struct ModelInfo
	{
		std::string Name;
		Datasets Dataset;
		Costs LossFunction;
		UInt LayerCount;
		UInt CostLayerCount;
		UInt CostIndex;
		UInt GroupIndex;
		UInt LabelIndex;
		UInt Hierarchies;
		UInt TrainingSamplesCount;
		UInt TestingSamplesCount;
		bool MeanStdNormalization;
		std::vector<Float> MeanTrainSet;
		std::vector<Float> StdTrainSet;
	};

	struct CostInfo
	{
		UInt TrainErrors;
		Float TrainLoss;
		Float AvgTrainLoss;
		Float TrainErrorPercentage;
		UInt TestErrors;
		Float TestLoss;
		Float AvgTestLoss;
		Float TestErrorPercentage;
	};

	struct StatsInfo
	{
		std::string Description;
		Stats NeuronsStats;
		Stats WeightsStats;
		Stats BiasesStats;
		Float FPropLayerTime;
		Float BPropLayerTime;
		Float UpdateLayerTime;
		Float FPropTime;
		Float BPropTime;
		Float UpdateTime;
		bool Locked;
	};

	static const bool IsBatchNorm(const LayerTypes& type)
	{
		return std::string(magic_enum::enum_name<LayerTypes>(type)).find("BatchNorm", 0) != std::string::npos;
	}

	
	class Model
	{
	private:
		std::future<void> Task;

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
		bool FixedDepthDrop;
		Float DepthDrop;
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
		std::vector<bool> TrainingSamplesHFlip;
		std::vector<bool> TrainingSamplesVFlip;
		std::vector<bool> TestingSamplesHFlip;
		std::vector<bool> TestingSamplesVFlip;
		std::vector<UInt> RandomTrainingSamples;
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

		auto GetModelName(const std::string& definition) const
		{
			if (definition.length() > 2)
			{
				auto first = definition.find_first_of("]");
				if (first != std::string::npos)
					return definition.substr(1, first - 1);
			}

			return std::string("");
		}

		Model(const std::string& definition, Dataprovider* dataprovider) :
			Name(GetModelName(definition)),
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
			H(32),
			W(32),
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
			DepthDrop(Float(0.0)),					// DepthDrop			(Stochastic Depth: https://www.arxiv-vanity.com/papers/1603.09382/)
			FixedDepthDrop(false),					// FixedDepthDrop
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
			TrainingStrategies(std::vector<TrainingStrategy>()),
			UseTrainingStrategy(false),
			SampleSpeed(Float(0)),
			NewEpoch(nullptr),
			AdjustedTrainingSamplesCount(0),
			AdjustedTestingSamplesCount(0),
			TrainOverflowCount(0),
			TestOverflowCount(0),
			TrainSkipCount(0),
			TestSkipCount(0),
			fpropTime(std::chrono::duration<Float>(Float(0))),
			bpropTime(std::chrono::duration<Float>(Float(0))),
			updateTime(std::chrono::duration<Float>(Float(0))),
			FirstUnlockedLayer(1),
			BatchSizeChanging(false),
			ResettingWeights(false)
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
			//dnnl::set_primitive_cache_capacity(1000);
			//dnnl::set_default_fpmath_mode(dnnl::fpmath_mode::any);
		}

		virtual ~Model() = default;
		
		void Load(const std::string& fileName)
		{
			auto is = std::ifstream(fileName, std::ios::in | std::ios::binary);

			if (!is.bad() && is.is_open())
			{
				auto state = bitsery::quickDeserialization<bitsery::InputStreamAdapter>(is, *this);
				assert(state.first == bitsery::ReaderError::NoError && state.second);
				is.close();
			}
		}

		void Save(const std::string& fileName)
		{
			auto os = std::fstream{ fileName, std::ios::binary | std::ios::trunc | std::ios::out };

			if (!os.bad() && os.is_open())
			{
				bitsery::Serializer<bitsery::OutputBufferedStreamAdapter> serializer{ os };
				serializer.object(*this);
				serializer.adapter().flush();
				os.close();
			}
		}

		auto GetWeightsSize(const bool persistOptimizer, const Optimizers optimizer) const
		{
			std::streamsize weightsSize = 0;

			for (const auto& layer : Layers)
				weightsSize += layer->GetWeightsSize(persistOptimizer, optimizer);

			return weightsSize;
		}

		auto GetNeuronsSize(const UInt batchSize) const
		{
			UInt neuronsSize = 0;

			for (const auto& layer : Layers)
				neuronsSize += layer->GetNeuronsSize(batchSize);

			return neuronsSize;
		}

		void SaveDefinition(const std::string& fileName)
		{
			std::fstream file{ fileName, file.trunc | file.out };

			if (file.is_open())
			{
				file << CaseInsensitiveReplace(Definition.begin(), Definition.end(), nwl, "\n");
				file.close();
			}
		}

		bool BatchNormalizationUsed() const
		{
			for (const auto& layer : Layers)
				if (IsBatchNorm(layer->LayerType))
					return true;

			return false;
		}

		bool ChangeResolution(const UInt batchSize, const UInt h, const UInt w, const UInt padH, const UInt padW)
		{
			if (batchSize < 1 || h < 1 || w < 1 || padH < 1 || padW < 1)
				throw false;

			if (batchSize == BatchSize && h == H && w == W)
			{
				PadH = padH;
				PadW = padW;

				return true;
			}
						
			const auto currentSize = GetNeuronsSize(BatchSize) + GetWeightsSize(PersistOptimizer, Optimizer);

			while (BatchSizeChanging.load() || ResettingWeights.load())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				std::this_thread::yield();
			}

			BatchSizeChanging.store(true);

			if (Layers[0]->H != h && Layers[0]->W != w)
			{
				Layers[0]->H = h;
				Layers[0]->W = w;
				for (auto& layer : Layers)
					layer->UpdateResolution();
			}

			if (batchSize * h * w > BatchSize * H * W)
			{
				auto requestedSize = GetNeuronsSize(batchSize) + GetWeightsSize(PersistOptimizer, Optimizer);

				if (GetTotalFreeMemory() + currentSize < requestedSize)
				{
					std::cout << std::string("Memory required: ") << std::to_string(requestedSize / 1024 / 1024) << std::string(" MB with resolution") << std::to_string(batchSize) + std::string("x") + std::to_string(h) + std::string("x") + std::to_string(w) << std::endl << std::endl;

					Layers[0]->H = H;
					Layers[0]->W = W;
					for (auto& layer : Layers)
						layer->UpdateResolution();

					BatchSizeChanging.store(false);

					return false;
				}
			}

			for (auto& layer : Layers)
				layer->SetBatchSize(batchSize);
				
			AdjustedTrainingSamplesCount = (DataProv->TrainingSamplesCount % batchSize == 0) ? DataProv->TrainingSamplesCount : ((DataProv->TrainingSamplesCount / batchSize) + 1) * batchSize;
			AdjustedTestingSamplesCount = (DataProv->TestingSamplesCount % batchSize == 0) ? DataProv->TestingSamplesCount : ((DataProv->TestingSamplesCount / batchSize) + 1) * batchSize;
			TrainSkipCount = batchSize - (AdjustedTrainingSamplesCount - DataProv->TrainingSamplesCount);
			TestSkipCount = batchSize - (AdjustedTestingSamplesCount - DataProv->TestingSamplesCount);
			TrainOverflowCount = AdjustedTrainingSamplesCount - batchSize;
			TestOverflowCount = AdjustedTestingSamplesCount - batchSize;;

			BatchSize = batchSize;
			H = h;
			W = w;
			PadH = padH;
			PadW = padW;

			BatchSizeChanging.store(false);

			return true;
		}

		void ChangeDropout(const Float dropout, const UInt batchSize)
		{
			if (dropout < 0 || dropout >= 1)
				throw std::invalid_argument("Invalid dropout value in ChangeDropout function");

			for (auto& layer : Layers)
			{
				switch (layer->LayerType)
				{
				case LayerTypes::BatchNormActivationDropout:
				{
					auto bn = dynamic_cast<BatchNormActivationDropout*>(layer.get());
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
					layer->NeuronsFormat = Format;
				
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
					{
						if (UseTrainingStrategy && TrainingStrategies.size() > 0)
						{
							TrainingRate tmpRate;

							auto sum = Float(0);
							for (const auto& strategy : TrainingStrategies)
							{
								tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, strategy.PadH, strategy.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

								sum += strategy.Epochs * rate.Epochs;
								if ((i + 1) <= sum)
									break;
							}

							TrainingRates.push_back(tmpRate);
						}
						else
							TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, rate.PadH, rate.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
					}
				}
				else
				{
					if ((i + 1) >= gotoEpoch)
					{
						if (UseTrainingStrategy && TrainingStrategies.size() > 0)
						{
							TrainingRate tmpRate;

							auto sum = Float(0);
							for (const auto& strategy : TrainingStrategies)
							{
								tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, strategy.PadH, strategy.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

								sum += strategy.Epochs * rate.Epochs;
								if ((i + 1) <= sum)
									break;
							}

							TrainingRates.push_back(tmpRate);
						}
						else
							TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, rate.PadH, rate.PadW, 1, rate.Epochs, 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
					}
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
				{
					if (UseTrainingStrategy && TrainingStrategies.size() > 0)
					{
						TrainingRate tmpRate;

						auto sum = Float(0);
						for (const auto& strategy : TrainingStrategies)
						{
							tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, strategy.PadH, strategy.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

							sum += strategy.Epochs * rate.Epochs;
							if (totIteration * decayAfterEpochs <= sum)
								break;
						}

						TrainingRates.push_back(tmpRate);
					}
					else
						TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, rate.PadH, rate.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				}
			}
			else
			{
				if ((totIteration * decayAfterEpochs) < rate.Epochs)
				{
					if (UseTrainingStrategy && TrainingStrategies.size() > 0)
					{
						TrainingRate tmpRate;

						auto sum = Float(0);
						for (const auto& strategy : TrainingStrategies)
						{
							tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, strategy.PadH, strategy.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs),1, newRate, rate.MinimumRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

							sum += strategy.Epochs * rate.Epochs;
							if (totIteration * decayAfterEpochs <= sum)
								break;
						}

						TrainingRates.push_back(tmpRate);
					}
					else
						TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, rate.PadH, rate.PadW, 1, rate.Epochs - (totIteration * decayAfterEpochs), 1, newRate, rate.MinimumRate, rate.FinalRate / LR, rate.Gamma, decayAfterEpochs, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
				}
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
							if (UseTrainingStrategy && TrainingStrategies.size() > 0)
							{
								TrainingRate tmpRate;

								auto sum = Float(0);
								for (const auto& strategy : TrainingStrategies)
								{
									tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, weightDecayMultiplier * weightDecayNormalized, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, strategy.PadH, strategy.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);

									sum += strategy.Epochs * totalEpochs;
									if (epoch <= sum)
										break;
								}

								TrainingRates.push_back(tmpRate);
							}
							else
								TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, weightDecayMultiplier * weightDecayNormalized, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, rate.PadH, rate.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, rate.Gamma, 1, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
						}
					}
					else
					{
						if (epoch >= gotoEpoch)
						{
							if (UseTrainingStrategy && TrainingStrategies.size() > 0)
							{
								TrainingRate tmpRate;
								
								auto sum = Float(0);
								for (const auto& strategy : TrainingStrategies)
								{
									tmpRate = TrainingRate(rate.Optimizer, strategy.Momentum, strategy.Beta2, strategy.L2Penalty, strategy.Dropout, rate.Eps, strategy.BatchSize, strategy.Height, strategy.Width, strategy.PadH, strategy.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, strategy.Gamma, 1, Float(1), strategy.HorizontalFlip, strategy.VerticalFlip, strategy.InputDropout, strategy.Cutout, strategy.CutMix, strategy.AutoAugment, strategy.ColorCast, strategy.ColorAngle, strategy.Distortion, strategy.Interpolation, strategy.Scaling, strategy.Rotation);
									
									sum += strategy.Epochs * totalEpochs;
									if (epoch <= sum)
										break;
								}

								TrainingRates.push_back(tmpRate);
							}
							else
								TrainingRates.push_back(TrainingRate(rate.Optimizer, rate.Momentum, rate.Beta2, rate.L2Penalty, rate.Dropout, rate.Eps, rate.BatchSize, rate.Height, rate.Width, rate.PadH, rate.PadW, c + 1, 1, rate.EpochMultiplier, newRate, minRate, rate.FinalRate / LR, rate.Gamma, 1, Float(1), rate.HorizontalFlip, rate.VerticalFlip, rate.InputDropout, rate.Cutout, rate.CutMix, rate.AutoAugment, rate.ColorCast, rate.ColorAngle, rate.Distortion, rate.Interpolation, rate.Scaling, rate.Rotation));
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
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::this_thread::yield(); 
			}
			
			return TaskState.load() == TaskStates::Running;
		}

		void TrainingAsync()
		{
			Task = std::async(std::launch::async, [=] { Training(); });
			//Task = std::async(std::launch::async, [=] { CheckValid(); });
		}

		void TestingAsync()
		{
			Task = std::async(std::launch::async, [=] { Testing(); });
		}

		void StopTask()
		{
			if (TaskState.load() != TaskStates::Stopped)
			{
				TaskState.store(TaskStates::Stopped);

				if (Task.valid())
					try
				    {
					    Task.get();
				    }
				    catch (const std::future_error& e)
				    {
					    std::cout << std::string("StopTask exception: ") << std::string(e.what()) << std::endl << std::string("code: ") << e.code().message() << std::endl;
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

		/*
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
		*/

		void SwitchInplaceBwd(const bool enable)
		{
			if constexpr (Inplace)
			{
				if (enable)
					for (auto& layer : Layers)
					{
						layer->Inputs = std::vector<Layer*>(layer->InputsBwd);
						layer->InputLayer = layer->InputLayerBwd;
						layer->SharesInput = layer->SharesInputInplace;
					}
				else
					for (auto& layer : Layers)
					{
						layer->Inputs = std::vector<Layer*>(layer->InputsFwd);
						layer->InputLayer = layer->InputLayerFwd;
						layer->SharesInput = layer->SharesInputOriginal;
					}
			}
		}

		auto IsSkippable(const Layer& layer) const
		{
			return layer.LayerType == LayerTypes::Add || layer.LayerType == LayerTypes::Average || layer.LayerType == LayerTypes::Substract; // || layer.LayerType == LayerTypes::Multiply || layer.LayerType == LayerTypes::Divide;
		}

		auto GetTotalSkipConnections() const
		{
			auto totalSkipConnections = 0ull;
		
			for (const auto& layer : Layers)
				if (IsSkippable(*layer.get()))
					for (const auto& l : layer->Outputs)
						if (IsSkippable(*l))
							totalSkipConnections++;

			return totalSkipConnections;
		}

		void StochasticDepth(const UInt totalSkipConnections, const Float dropRate = Float(0.5), const bool fixed = false)
		{
			auto isSkipConnection = false;
			auto endLayer = std::string("");
			auto survive = true;
			
			auto skipConnection = 0ull;
			for (auto& layer : Layers)
			{
				if (IsSkippable(*layer.get()))
				{
					if (isSkipConnection && layer->Name == endLayer)
					{
						auto survivalProb = Float(1);

						for (auto inputLayer = 0ull; inputLayer < layer->Inputs.size(); inputLayer++)
						{
							if (!IsSkippable(*layer->Inputs[inputLayer]))
								survivalProb = fixed ? Float(1) / (Float(1) - dropRate) : Float(1) / (Float(1) - (dropRate * Float(skipConnection) / Float(totalSkipConnections)));
							else
								survivalProb = Float(1);
							
							switch (layer->LayerType)
							{
							case LayerTypes::Add:
								dynamic_cast<Add*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							case LayerTypes::Average:
								dynamic_cast<Average*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							case LayerTypes::Substract:
								dynamic_cast<Substract*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							/*case LayerTypes::Multiply:
								dynamic_cast<Multiply*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;
							case LayerTypes::Divide:
								dynamic_cast<Divide*>(layer.get())->SurvivalProbability[inputLayer] = survivalProb;
								break;*/
							default:
								break;
							}
						}
					}
					
					isSkipConnection = false;
					for (const auto& outputLayer : layer->Outputs)
					{
						if (IsSkippable(*outputLayer))
						{
							isSkipConnection = true;
							endLayer = outputLayer->Name;
							skipConnection++;
							survive = Bernoulli<bool>(fixed ? Float(1) / (Float(1) - dropRate) : (Float(1) - (dropRate * Float(skipConnection) / Float(totalSkipConnections))));
						}
					}
				}
				else if (isSkipConnection)
					layer->Skip = !survive;
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
					throw std::invalid_argument((std::string("Invalid input layer: ") + name).c_str());
			}

			return list;
		}

		std::vector<Layer*> GetLayerOutputs(const Layer* parentLayer, const bool inplace = false) const
		{
			auto outputs = std::vector<Layer*>();

			for (auto& layer : Layers)
				if (layer->Name != parentLayer->Name)
					for (auto input : inplace ? layer->InputsBwd : layer->InputsFwd)
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
				layer->Outputs = GetLayerOutputs(layer.get());
				auto outputsCount = layer->Outputs.size();

				if (outputsCount > 1)
				{
					for (auto& l : Layers)
					{
						if (l->Name == layer->Name)
							continue;

						for (auto input : l->InputsFwd)
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

						for (auto input : l->InputsBwd)
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

				auto msg = std::string();
				if (!Activation::CheckActivations(msg))
				{
					cimg_library::cimg::dialog("Activations Sanity Check", msg.c_str(), "OK");

					State.store(States::Completed);
					return;
				};

				const auto totalSkipConnections = GetTotalSkipConnections();

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
			
				if (!ChangeResolution(CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
					return;
				
				if (Dropout != CurrentTrainingRate.Dropout)
					ChangeDropout(CurrentTrainingRate.Dropout, BatchSize);

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
					TrainingSamplesHFlip.push_back(Bernoulli<bool>(Float(0.5)));
					TrainingSamplesVFlip.push_back(Bernoulli<bool>(Float(0.5)));
				}
				for (auto index = 0ull; index < DataProv->TestingSamplesCount; index++)
				{
					TestingSamplesHFlip.push_back(Bernoulli<bool>(Float(0.5)));
					TestingSamplesVFlip.push_back(Bernoulli<bool>(Float(0.5)));
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
						
						if (!ChangeResolution(CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
							return;
								
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

						const auto shuffleCount = UniformInt<UInt>(DataProv->ShuffleCount / 2ull, DataProv->ShuffleCount);
						for (auto shuffle = 0ull; shuffle < shuffleCount; shuffle++)
							std::shuffle(std::begin(RandomTrainingSamples), std::end(RandomTrainingSamples), std::mt19937(Seed<unsigned>()));

						for (auto cost : CostLayers)
							cost->Reset();

#ifdef DNN_STOCHASTIC				
						if (BatchSize == 1)
						{
							for (SampleIndex = 0; SampleIndex < DataProv->TrainingSamplesCount; SampleIndex++)
							{
								if (DepthDrop > 0)
									StochasticDepth(totalSkipConnections, DepthDrop, FixedDepthDrop);

								// Forward
								timePointGlobal = timer.now();
								auto SampleLabel = TrainSample(SampleIndex);
								Layers[0]->fpropTime = timer.now() - timePointGlobal;

								for (auto cost : CostLayers)
									cost->SetSampleLabel(SampleLabel);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									timePoint = timer.now();
									if (!Layers[i]->Skip)
										Layers[i]->ForwardProp(1, true);
									Layers[i]->fpropTime = timer.now() - timePoint;
								}

								CostFunction(State.load());
								Recognized(State.load(), SampleLabel);
								fpropTime = timer.now() - timePointGlobal;

								// Backward
								bpropTimeCount = std::chrono::duration<Float>(Float(0));
								updateTimeCount = std::chrono::duration<Float>(Float(0));
								SwitchInplaceBwd(true);
								for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
								{
									if (Layers[i]->HasWeights && TaskState.load() == TaskStates::Running)
									{
										timePoint = timer.now();
										if (!Layers[i]->Skip)
										{
											Layers[i]->ResetGradients();
											Layers[i]->BackwardProp(BatchSize);
										}
										Layers[i]->bpropTime = timer.now() - timePoint;
										timePoint = timer.now();
										if (!Layers[i]->Skip)
											Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
										Layers[i]->updateTime = timer.now() - timePoint;
										updateTimeCount += Layers[i]->updateTime;
									}
									else
									{
										timePoint = timer.now();
										if (!Layers[i]->Skip && TaskState.load() == TaskStates::Running)
											Layers[i]->BackwardProp(1);
										Layers[i]->bpropTime = timer.now() - timePoint;
									}
									bpropTimeCount += Layers[i]->bpropTime;
								}
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
								if (DepthDrop > 0)
									StochasticDepth(totalSkipConnections, DepthDrop, FixedDepthDrop);

								while (Layers[0]->RefreshingStats.load()) {	std::this_thread::yield(); }
								Layers[0]->Fwd.store(true);
								timePointGlobal = timer.now();
								auto SampleLabels = TrainBatch(SampleIndex, BatchSize);
								Layers[0]->fpropTime = timer.now() - timePointGlobal;
								Layers[0]->Fwd.store(false);

								for (auto cost : CostLayers)
									cost->SetSampleLabels(SampleLabels);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									if (!Layers[i]->Skip && TaskState.load() == TaskStates::Running)
									{
										while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
										Layers[i]->Fwd.store(true);
										timePoint = timer.now();
										Layers[i]->ForwardProp(BatchSize, true);
										Layers[i]->fpropTime = timer.now() - timePoint;
										Layers[i]->Fwd.store(false);
									}
									else
										Layers[i]->fpropTime = std::chrono::duration<Float>(Float(0));
								}
								
								overflow = SampleIndex >= TrainOverflowCount;
								CostFunctionBatch(State.load(), BatchSize, overflow, TrainSkipCount);
								RecognizedBatch(State.load(), BatchSize, overflow, TrainSkipCount, SampleLabels);
								fpropTime = timer.now() - timePointGlobal;

								// Backward
								bpropTimeCount = std::chrono::duration<Float>(Float(0));
								updateTimeCount = std::chrono::duration<Float>(Float(0));
								SwitchInplaceBwd(true);
								for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
								{
									if (TaskState.load() == TaskStates::Running)
									{
										Layers[i]->bpropTime = std::chrono::duration<Float>(Float(0));
										Layers[i]->updateTime = std::chrono::duration<Float>(Float(0));

										if (!Layers[i]->Skip)
										{
											while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
											Layers[i]->Bwd.store(true);
											timePoint = timer.now();

											if (Layers[i]->HasWeights)
											{
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
												Layers[i]->BackwardProp(BatchSize);
												Layers[i]->bpropTime = timer.now() - timePoint;
											}

											bpropTimeCount += Layers[i]->bpropTime;
											Layers[i]->Bwd.store(false);
										}										
									}
								}
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

								while (Layers[0]->RefreshingStats.load()) { std::this_thread::yield(); }
								Layers[0]->Fwd.store(true);
								timePoint = timer.now();
								auto SampleLabels = TestBatch(SampleIndex, BatchSize);
								Layers[0]->fpropTime = timer.now() - timePoint;
								Layers[0]->Fwd.store(false);

								for (auto cost : CostLayers)
									cost->SetSampleLabels(SampleLabels);

								for (auto i = 1ull; i < Layers.size(); i++)
								{
									while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
									Layers[i]->Fwd.store(true);
									timePoint = timer.now();
									Layers[i]->ForwardProp(BatchSize, false);
									Layers[i]->fpropTime = timer.now() - timePoint;
									Layers[i]->Fwd.store(false);
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

							// save the weights and definition
							State.store(States::SaveWeights);
							const auto fileName = PersistOptimizer ? (std::string("(") + StringToLower(std::string(magic_enum::enum_name<Datasets>(Dataset))) + std::string(")(") + StringToLower(std::string(magic_enum::enum_name<Optimizers>(Optimizer))) + std::string(").bin")) : (std::string("(") + StringToLower(std::string(magic_enum::enum_name<Datasets>(Dataset))) + std::string(").bin"));
							const auto dir = DataProv->StorageDirectory / std::string("definitions") / Name;
							std::filesystem::create_directories(dir);
							const auto epoch = std::string("(") + StringToLower(std::string(magic_enum::enum_name<Datasets>(Dataset))) + std::string(")(") + StringToLower(std::string(magic_enum::enum_name<Optimizers>(Optimizer))) + std::string(")") + std::to_string(CurrentEpoch) + std::string("-") + std::to_string(CurrentCycle) + std::string("-") + std::to_string(TrainErrors) + std::string("-") + std::to_string(TestErrors);
							const auto subdir = dir / epoch;
							std::filesystem::current_path(dir);
							std::filesystem::create_directories(subdir);
							SaveWeights((subdir / fileName).string(), PersistOptimizer);
							SaveDefinition((subdir / std::string("model.txt")).string());
							Save((subdir / std::string("model.bin")).string());
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

#ifndef NDEBUG
		void CheckValid()
		{
			if (TaskState == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);
				const auto totalSkipConnections = GetTotalSkipConnections();

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

				if (!ChangeResolution(CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
					return;

				if (Dropout != CurrentTrainingRate.Dropout)
					ChangeDropout(CurrentTrainingRate.Dropout, BatchSize);

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
					TrainingSamplesHFlip.push_back(false);
					TrainingSamplesVFlip.push_back(false);
				}
				for (auto index = 0ull; index < DataProv->TestingSamplesCount; index++)
				{
					TestingSamplesHFlip.push_back(false);
					TestingSamplesVFlip.push_back(false);
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

				State.store(States::Training);

				if (CheckTaskState())
				{
					if constexpr (Inplace)
						SwitchInplaceBwd(false);

					for (auto cost : CostLayers)
						cost->Reset();


					auto overflow = false;
					SampleIndex = 0;

					timePointGlobal = timer.now();

					auto SampleLabels = TrainCheckBatch(SampleIndex, BatchSize);
					std::filesystem::path path = std::filesystem::path("C:\\Users\\dhaen\\");

					auto os = std::ofstream((path / "testB.bin").string(), std::ios::out | std::ios::binary | std::ios::trunc);
					auto oss = std::ofstream((path / "testB.txt").string(), std::ios::out | std::ios::trunc);

					size_t point = 0;
					if (!os.bad() && os.is_open())
					{
						Layers[0]->SaveNeurons(os);
						//oss <<  std::to_string(point) + " Input\n";
						//point += Layers[0]->Neurons.size();

						Layers[0]->fpropTime = timer.now() - timePointGlobal;

						//point += Layers[0]->Neurons.size();

						for (auto cost : CostLayers)
							cost->SetSampleLabels(SampleLabels);

						for (auto i = 1ull; i < Layers.size(); i++)
						{
							timePoint = timer.now();
							Layers[i]->ForwardProp(BatchSize, true);
							if (i < 3ull)
								Layers[i]->SaveNeurons(os);
							//oss << std::to_string(point) + " " + Layers[i]->Name + "\n";
							//point += Layers[i]->Neurons.size();
							Layers[i]->fpropTime = timer.now() - timePoint;
						}
						fpropTime = timer.now() - timePointGlobal;
						overflow = SampleIndex >= TrainOverflowCount;
						CostFunctionBatch(State.load(), BatchSize, overflow, TrainSkipCount);
						RecognizedBatch(State.load(), BatchSize, overflow, TrainSkipCount, SampleLabels);

						for (auto i = Layers.size() - 1; i >= FirstUnlockedLayer.load(); --i)
						{
							if (Layers[i]->HasWeights)
							{
								//Layers[i]->ResetGradients();
								Layers[i]->BackwardProp(BatchSize);
								//Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
								//Layers[i]->SaveGradients(os);
							}
							else
								Layers[i]->BackwardProp(BatchSize);

							//Layers[i]->SaveNeuronsD1(os);
							//oss << std::to_string(point) + " " + Layers[i]->Name + "\n";
							//point += Layers[i]->NeuronsD1.size();
						}

						os.flush();
						os.close();

						oss.flush();
						oss.close();
					}

					SampleSpeed = BatchSize / (Float(std::chrono::duration_cast<std::chrono::microseconds>(fpropTime).count()) / 1000000);

					for (auto cost : CostLayers)
					{
						cost->AvgTestLoss = cost->TrainLoss / DataProv->TrainingSamplesCount;
						cost->TestErrorPercentage = cost->TrainErrors / Float(DataProv->TrainingSamplesCount / 100);
					}
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
#endif

		void Testing()
		{
			if (TaskState == TaskStates::Stopped && !BatchSizeChanging.load() && !ResettingWeights.load())
			{
				TaskState.store(TaskStates::Running);
				State.store(States::Idle);

				auto timer = std::chrono::high_resolution_clock();
				auto timePoint = timer.now();
				auto timePointGlobal = timer.now();
				//auto elapsedTime = std::chrono::duration<Float>(Float(0));

				CurrentTrainingRate = TrainingRates[0];
				Rate = CurrentTrainingRate.MaximumRate;

				if (!ChangeResolution(CurrentTrainingRate.BatchSize, CurrentTrainingRate.Height, CurrentTrainingRate.Width, CurrentTrainingRate.PadH, CurrentTrainingRate.PadW))
					return;

				if (Dropout != CurrentTrainingRate.Dropout)
					ChangeDropout(CurrentTrainingRate.Dropout, BatchSize);


				TrainingSamplesHFlip = std::vector<bool>();
				TrainingSamplesVFlip = std::vector<bool>();
				TestingSamplesHFlip = std::vector<bool>();
				TestingSamplesVFlip = std::vector<bool>();
				for (auto index = 0ull; index < DataProv->TrainingSamplesCount; index++)
				{
					TrainingSamplesHFlip.push_back(Bernoulli<bool>(Float(0.5)));
					TrainingSamplesVFlip.push_back(Bernoulli<bool>(Float(0.5)));
				}
				for (auto index = 0ull; index < DataProv->TestingSamplesCount; index++)
				{
					TestingSamplesHFlip.push_back(Bernoulli<bool>(Float(0.5)));
					TestingSamplesVFlip.push_back(Bernoulli<bool>(Float(0.5)));
				}

				State.store(States::Testing);

				if (CheckTaskState())
				{
					SwitchInplaceBwd(false);

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
#endif
						auto overflow = false;
						for (SampleIndex = 0; SampleIndex < AdjustedTestingSamplesCount; SampleIndex += BatchSize)
						{
							timePointGlobal = timer.now();

							while (Layers[0]->RefreshingStats.load()) { std::this_thread::yield(); }
							Layers[0]->Fwd.store(true);
							auto SampleLabels = TestAugmentedBatch(SampleIndex, BatchSize);
							Layers[0]->fpropTime = timer.now() - timePointGlobal;
							Layers[0]->Fwd.store(false);

							for (auto cost : CostLayers)
								cost->SetSampleLabels(SampleLabels);

							for (auto i = 1ull; i < Layers.size(); i++)
							{
								while (Layers[i]->RefreshingStats.load()) { std::this_thread::yield(); }
								Layers[i]->Fwd.store(true);
								timePoint = timer.now();
								Layers[i]->ForwardProp(BatchSize, false);
								Layers[i]->fpropTime = timer.now() - timePoint;
								Layers[i]->Fwd.store(false);
							}

							overflow = SampleIndex >= TestOverflowCount;
							CostFunctionBatch(State.load(), BatchSize, overflow, TestSkipCount);
							RecognizedBatch(State.load(), BatchSize, overflow, TestSkipCount, SampleLabels);

							fpropTime = timer.now() - timePointGlobal;

							SampleSpeed = BatchSize / (Float(std::chrono::duration_cast<std::chrono::microseconds>(fpropTime).count()) / 1000000);

							if (TaskState.load() != TaskStates::Running && !CheckTaskState())
								break;
						}
#ifdef DNN_STOCHASTIC
					}
#endif
					for (auto cost : CostLayers)
					{
						cost->AvgTestLoss = cost->TestLoss / DataProv->TestingSamplesCount;
						cost->TestErrorPercentage = cost->TestErrors / Float(DataProv->TestingSamplesCount / 100);
					}
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
			if (!Layers[0]->Neurons.empty() && !BatchSizeChanging.load() && !ResettingWeights.load() && !Layers[0]->Fwd.load())
			{
				const auto idx = UniformInt<UInt>(0ull, BatchSize - 1ull) + SampleIndex;
				const auto size = Layers[0]->CDHW();
				const auto offset = (idx - SampleIndex) * size;

				if (State.load() == States::Training && idx < DataProv->TrainingSamplesCount)
				{
					*label = DataProv->TrainingLabels[RandomTrainingSamples[idx]];
					
					for (auto i = 0ull; i < size; i++)
						(*snapshot)[i] = Layers[0]->Neurons[i + offset];

					return true;
				}
				else if (State.load() == States::Testing && idx < DataProv->TestingSamplesCount)
				{
					*label = DataProv->TestingLabels[idx];
					
					for (auto i = 0ull; i < size; i++)
						(*snapshot)[i] = Layers[0]->Neurons[i + offset];

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
					Image<Byte>::RandomCutMix(imgByte, imgByteMix, &lambda);
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
				Image<Byte>::HorizontalMirror(imgByte);

			if (CurrentTrainingRate.VerticalFlip && TrainingSamplesVFlip[rndIndex])
				Image<Byte>::VerticalMirror(imgByte);

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

			if (imgByteMix.D() != D || imgByte.H() != H || imgByte.W() != W)
				Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (cutout)
				Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

			if (CurrentTrainingRate.InputDropout > Float(0))
				Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

			if (RandomCrop)
				imgByte = Image<Byte>::RandomCrop(imgByte, D, H, W, DataProv->Mean);

			for (auto c = 0u; c < imgByte.C(); c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);

				for (auto d = 0u; d < imgByte.D(); d++)
					for (auto h = 0u; h < imgByte.H(); h++)
						for (auto w = 0u; w < imgByte.W(); w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<LabelInfo> TestSample(const UInt index)
		{
			auto label = DataProv->TestingLabels[index];
			auto SampleLabel = GetLabelInfo(label);

			auto imgByte = DataProv->TestingSamples[index];

			if (imgByte.D() != D || imgByte.H() != H || imgByte.W() != W)
				Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

			imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (RandomCrop)
				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

			for (auto c = 0u; c < imgByte.C(); c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);

				for (auto d = 0u; d < imgByte.D(); d++)
					for (auto h = 0u; h < imgByte.H(); h++)
						for (auto w = 0u; w < imgByte.W(); w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}

		std::vector<LabelInfo> TestAugmentedSample(const UInt index)
		{
			auto label = DataProv->TestingLabels[index];
			auto SampleLabel = GetLabelInfo(label);

			auto imgByte = DataProv->TestingSamples[index];

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
				Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

			if (CurrentTrainingRate.HorizontalFlip && TestingSamplesHFlip[index])
				Image<Byte>::HorizontalMirror(imgByte);

			if (CurrentTrainingRate.VerticalFlip && TestingSamplesVFlip[index])
				Image<Byte>::VerticalMirror(imgByte);

			if (imgByte.D() != D || imgByte.H() != H || imgByte.W() != W)
				Image<Byte>::Resize(imgByte, D, H, W, static_cast<Interpolations>(CurrentTrainingRate.Interpolation));

			if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
				imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
			else
				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

			if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
				imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, static_cast<Interpolations>(CurrentTrainingRate.Interpolation), DataProv->Mean);

			if (Bernoulli<bool>(CurrentTrainingRate.Cutout) && !CurrentTrainingRate.CutMix)
				Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

			if (RandomCrop)
				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

			if (CurrentTrainingRate.InputDropout > Float(0))
				Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

			for (auto c = 0u; c < imgByte.C(); c++)
			{
				const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
				const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);

				for (auto d = 0u; d < imgByte.D(); d++)
					for (auto h = 0u; h < imgByte.H(); h++)
						for (auto w = 0u; w < imgByte.W(); w++)
							Layers[0]->Neurons[(c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
			}

			return SampleLabel;
		}
#endif

		std::vector<std::vector<LabelInfo>> TrainCheckBatch(const UInt index, const UInt batchSize)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TrainingSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = DataProv->TrainingLabels[sampleIndex];
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = DataProv->TrainingSamples[sampleIndex];

				if (resize)
					Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);
					
					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TrainBatch(const UInt index, const UInt batchSize)
		{
			const auto hierarchies = DataProv->Hierarchies;
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;
			
			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i_dynamic(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto randomIndex = (index + batchIndex >= DataProv->TrainingSamplesCount) ? RandomTrainingSamples[batchIndex] : RandomTrainingSamples[index + batchIndex];
				auto imgByte = DataProv->TrainingSamples[randomIndex];

				const auto randomIndexMix = (index + batchSize - (batchIndex + 1) >= DataProv->TrainingSamplesCount) ? RandomTrainingSamples[batchSize - (batchIndex + 1)] : RandomTrainingSamples[index + batchSize - (batchIndex + 1)];
				auto imgByteMix = DataProv->TrainingSamples[randomIndexMix];

				auto labels = DataProv->TrainingLabels[randomIndex];
				auto mixLabels = DataProv->TrainingLabels[randomIndexMix];
				
				auto cutout = false;
				if (Bernoulli<bool>(CurrentTrainingRate.Cutout))
				{
					if (CurrentTrainingRate.CutMix)
					{
						double lambda = BetaDistribution<double>(1, 1);
						Image<Byte>::RandomCutMix(imgByte, imgByteMix, &lambda);
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
					Image<Byte>::HorizontalMirror(imgByte);

				if (CurrentTrainingRate.VerticalFlip && TrainingSamplesVFlip[randomIndex])
					Image<Byte>::VerticalMirror(imgByte);

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

				if (resize)
					Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (cutout)
					Image<Byte>::RandomCutout(imgByte, DataProv->Mean);

				if (RandomCrop)
					imgByte = Image<Byte>::RandomCrop(imgByte, D, H, W, DataProv->Mean);

				if (CurrentTrainingRate.InputDropout > Float(0))
					Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);
				
				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);
					
					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TestBatch(const UInt index, const UInt batchSize)
		{
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(DataProv->Hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i_dynamic(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestingSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = DataProv->TestingLabels[sampleIndex];
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = DataProv->TestingSamples[sampleIndex];

				if (resize)
					Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);
					
					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
				}
			});

			return SampleLabels;
		}

		std::vector<std::vector<LabelInfo>> TestAugmentedBatch(const UInt index, const UInt batchSize)
		{
			auto SampleLabels = std::vector<std::vector<LabelInfo>>(batchSize, std::vector<LabelInfo>(DataProv->Hierarchies));
			const auto resize = DataProv->D != D || DataProv->H != H || DataProv->W != W;

			const auto elements = batchSize * C * D * H * W;
			const auto threads = GetThreads(elements, Float(10));

			for_i_dynamic(batchSize, threads, [=, &SampleLabels](const UInt batchIndex)
			{
				const auto sampleIndex = ((index + batchIndex) >= DataProv->TestingSamplesCount) ? batchIndex : index + batchIndex;

				auto labels = DataProv->TestingLabels[sampleIndex];
				SampleLabels[batchIndex] = GetLabelInfo(labels);

				auto imgByte = DataProv->TestingSamples[sampleIndex];

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.ColorCast))
					Image<Byte>::ColorCast(imgByte, CurrentTrainingRate.ColorAngle);

				if (CurrentTrainingRate.HorizontalFlip && TestingSamplesHFlip[sampleIndex])
					Image<Byte>::HorizontalMirror(imgByte);

				if (CurrentTrainingRate.VerticalFlip && TestingSamplesVFlip[sampleIndex])
					Image<Byte>::VerticalMirror(imgByte);

				if (resize)
					Image<Byte>::Resize(imgByte, D, H, W, Interpolations(CurrentTrainingRate.Interpolation));

				if (DataProv->C == 3 && Bernoulli<bool>(CurrentTrainingRate.AutoAugment))
					imgByte = Image<Byte>::AutoAugment(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);
				else
					imgByte = Image<Byte>::Padding(imgByte, PadD, PadH, PadW, DataProv->Mean, MirrorPad);

				if (Bernoulli<bool>(CurrentTrainingRate.Distortion))
					imgByte = Image<Byte>::Distorted(imgByte, CurrentTrainingRate.Scaling, CurrentTrainingRate.Rotation, Interpolations(CurrentTrainingRate.Interpolation), DataProv->Mean);

				if (Bernoulli<bool>(CurrentTrainingRate.Cutout) && !CurrentTrainingRate.CutMix)
					Image<Byte>::RandomCutout(imgByte, DataProv->Mean);
				
				if (RandomCrop)
					imgByte = Image<Byte>::Crop(imgByte, Positions::Center, D, H, W, DataProv->Mean);

				if (CurrentTrainingRate.InputDropout > Float(0))
					Image<Byte>::Dropout(imgByte, CurrentTrainingRate.InputDropout, DataProv->Mean);

				for (auto c = 0u; c < imgByte.C(); c++)
				{
					const auto mean = MeanStdNormalization ? DataProv->Mean[c] : imgByte.GetChannelMean(c);
					const auto stddev = MeanStdNormalization ? DataProv->StdDev[c] : imgByte.GetChannelStdDev(c);
					
					for (auto d = 0u; d < imgByte.D(); d++)
						for (auto h = 0u; h < imgByte.H(); h++)
							for (auto w = 0u; w < imgByte.W(); w++)
								Layers[0]->Neurons[batchIndex * imgByte.Size() + (c * imgByte.ChannelSize()) + (d * imgByte.Area()) + (h * imgByte.W()) + w] = (imgByte(c, d, h, w) - mean) / stddev;
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
			if constexpr (Inplace)
				SwitchInplaceBwd(true);

			for (auto i = Layers.size() - 1; i > 0ull; --i)
			{
				if (Layers[i]->HasWeights && TaskState.load() == TaskStates::Running)
				{
					Layers[i]->ResetGradients();
					Layers[i]->BackwardProp(batchSize);
					if (!DisableLocking)
						Layers[i]->UpdateWeights(CurrentTrainingRate, Optimizer, DisableLocking);
				}
				else
					Layers[i]->BackwardProp(batchSize);
			}

			if constexpr (Inplace)
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

	template<typename S>
	void serialize(S& s, TrainingStrategy& o)
	{
		s.value4b(o.Epochs);
		s.value8b(o.BatchSize);
		s.value8b(o.Height);
		s.value8b(o.Width);
		s.value8b(o.PadH);
		s.value8b(o.PadW);
		s.value4b(o.Momentum);
		s.value4b(o.Beta2);
		s.value4b(o.Gamma);
		s.value4b(o.L2Penalty);
		s.value4b(o.Dropout);
		s.boolValue(o.HorizontalFlip);
		s.boolValue(o.VerticalFlip);
		s.value4b(o.InputDropout);
		s.value4b(o.Cutout);
		s.boolValue(o.CutMix);
		s.value4b(o.AutoAugment);
		s.value4b(o.ColorCast);
		s.value8b(o.ColorAngle);
		s.value4b(o.Distortion);
		s.value4b(o.Interpolation);
		s.value4b(o.Scaling);
		s.value4b(o.Rotation);
	}
	
	template<typename S>
	void serialize(S& s, TrainingRate& o)
	{
		s.value4b(o.Optimizer);
		s.value4b(o.Momentum);
		s.value4b(o.Beta2);
		s.value4b(o.L2Penalty);
		s.value4b(o.Eps);
		s.value8b(o.BatchSize);
		s.value8b(o.Height);
		s.value8b(o.Width);
		s.value8b(o.PadH);
		s.value8b(o.PadW);
		s.value8b(o.Cycles);
		s.value8b(o.Epochs);
		s.value8b(o.EpochMultiplier);
		s.value4b(o.MaximumRate);
		s.value4b(o.MinimumRate);
		s.value4b(o.FinalRate);
		s.value4b(o.Gamma);
		s.value8b(o.DecayAfterEpochs);
		s.value4b(o.DecayFactor);
		s.boolValue(o.HorizontalFlip);
		s.boolValue(o.VerticalFlip);
		s.value4b(o.InputDropout);
		s.value4b(o.Cutout);
		s.boolValue(o.CutMix);
		s.value4b(o.AutoAugment);
		s.value4b(o.ColorCast);
		s.value8b(o.ColorAngle);
		s.value4b(o.Distortion);
		s.value4b(o.Interpolation);
		s.value4b(o.Scaling);
		s.value4b(o.Rotation);
	}

	template<typename S>
	void serialize(S& s, Model& o)
	{
		//s.container1b(o.TrainingSamplesHFlip, 1000000);
		//s.container1b(o.TrainingSamplesVFlip, 1000000);
		//s.container1b(o.TestingSamplesHFlip, 500000);
		//s.container1b(o.TestingSamplesVFlip, 500000);
		s.container8b(o.RandomTrainingSamples, 1000000);
		//s.text1b(o.Name, 128);
		//s.text1b(o.Definition, 250000);
		s.value4b(o.Format);
		s.value4b(o.Dataset);
		//s.value4b<States>(o.State);
		//s.value4b<TaskStates>(o.TaskState);
		s.value4b(o.CostFuction);
		s.value4b(o.Optimizer);
		s.value8b(o.CostIndex);
		s.value8b(o.LabelIndex);
		s.value8b(o.GroupIndex);
		s.value8b(o.TotalCycles);
		s.value8b(o.TotalEpochs);
		s.value8b(o.CurrentCycle);
		s.value8b(o.CurrentEpoch);
		s.value8b(o.SampleIndex);
		s.value8b(o.BatchSize);
		s.value8b(o.GoToEpoch);
		s.value8b(o.AdjustedTrainingSamplesCount);
		s.value8b(o.AdjustedTestingSamplesCount);
		s.value8b(o.TrainSkipCount);
		s.value8b(o.TestSkipCount);
		s.value8b(o.TrainOverflowCount);
		s.value8b(o.TestOverflowCount);
		s.value8b(o.C);
		s.value8b(o.D);
		s.value8b(o.H);
		s.value8b(o.W);
		s.value8b(o.PadC);
		s.value8b(o.PadD);
		s.value8b(o.PadH);
		s.value8b(o.PadW);
		s.boolValue(o.MirrorPad);
		s.boolValue(o.RandomCrop);
		s.boolValue(o.MeanStdNormalization);
		s.boolValue(o.FixedDepthDrop);
		s.value4b(o.DepthDrop);
		s.value4b(o.WeightsFiller);
		s.value4b(o.WeightsFillerMode);
		s.value4b(o.WeightsGain);
		s.value4b(o.WeightsScale);
		s.value4b(o.WeightsLRM);
		s.value4b(o.WeightsWDM);
		s.value4b(o.BiasesFiller);
		s.value4b(o.BiasesFillerMode);
		s.value4b(o.BiasesGain);
		s.value4b(o.BiasesScale);
		s.value4b(o.BiasesLRM);
		s.value4b(o.BiasesWDM);
		s.value4b(o.AlphaFiller);
		s.value4b(o.BetaFiller);
		s.value4b(o.BatchNormMomentum);
		s.value4b(o.BatchNormEps);
		s.value4b(o.Dropout);
		s.value8b(o.TrainErrors);
		s.value8b(o.TestErrors);
		s.value4b(o.TrainLoss);
		s.value4b(o.TestLoss);
		s.value4b(o.AvgTrainLoss);
		s.value4b(o.AvgTestLoss);
		s.value4b(o.TrainErrorPercentage);
		s.value4b(o.TestErrorPercentage);
		s.value4b(o.Accuracy);
		s.value4b(o.SampleSpeed);
		s.value4b(o.Rate);
		s.boolValue(o.BatchNormScaling);
		s.boolValue(o.HasBias);
		s.boolValue(o.PersistOptimizer);
		s.boolValue(o.DisableLocking);
		s.object(o.CurrentTrainingRate);
		s.container<std::vector<TrainingRate>>(o.TrainingRates, 1024);
		s.container<std::vector<TrainingStrategy>>(o.TrainingStrategies, 1024);
		s.boolValue(o.UseTrainingStrategy);
		//s.value8b<UInt>(o.FirstUnlockedLayer);
	};
}