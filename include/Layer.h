#pragma once
#include "Dataprovider.h"

namespace dnn
{
	class Model;
	
	struct TrainingRate;

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
		BatchNormRelu = 8,
		BatchNormReluDropout = 9,
		BatchNormSwish = 10,
		ChannelMultiply = 11,
		ChannelShuffle = 12,
		ChannelSplit = 13,
		ChannelZeroPad = 14,
		Concat = 15,
		Convolution = 16,
		ConvolutionTranspose = 17,
		Cost = 18,
		Dense = 19,
		DepthwiseConvolution = 20,
		Divide = 21,
		Dropout = 22,
		GlobalAvgPooling = 23,
		GlobalMaxPooling = 24,
		Input = 25,
		LocalResponseNormalization = 26,
		Max = 27,
		MaxPooling = 38,
		Min = 29,
		Multiply = 30,
		PartialDepthwiseConvolution = 31,
		Resampling = 32,
		Substract = 33
	};

	enum class Optimizers
	{
		AdaDelta = 0,
		AdaGrad = 1,
		Adam = 2,
		Adamax = 3,
		NAG = 4,
		RMSProp = 5,
		SGD = 6,
		SGDMomentum = 7,
		RAdam = 8
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
	
	typedef std::pair<const dnnl::engine&, dnnl::stream> Device;

	class Layer
	{
	protected:
		dnn::Device Device;
		std::mt19937 RandomEngine;
		
	public:
		const std::string Name;
		const LayerTypes LayerType;
		const size_t WeightCount;
		const size_t BiasCount;
		const size_t C;
		const size_t D;
		const size_t H;
		const size_t W;
		const size_t HW;
		const size_t CDHW;
		const size_t PaddedC;
		const size_t PaddedCDHW;
		const size_t PadD;
		const size_t PadH;
		const size_t PadW;
		const bool HasPadding;
		const std::vector<Layer*> Inputs;
		Layer* InputLayer;
		std::vector<Layer*> Outputs;
		bool LayerBeforeCost;
		bool SharesInput;
		dnnl::memory::format_tag Format;
		const bool HasBias;
		const bool HasWeights;
		bool UseDefaultParameters;
		Fillers WeightsFiller;
		Float WeightsScale;
		Float WeightsLRM;
		Float WeightsWDM;
		Fillers BiasesFiller;
		Float BiasesScale;
		Float BiasesLRM;
		Float BiasesWDM;
		FloatVector Neurons;
		FloatVector NeuronsD1;
		FloatVector Weights;
		FloatVector WeightsD1;
		FloatVector Biases;
		FloatVector BiasesD1;
		FloatVector WeightsPar1;
		FloatVector WeightsPar2;
		FloatVector BiasesPar1;
		FloatVector BiasesPar2;
		size_t Moments;
		Float B1;
		Float B2;
		Float AdaDeltaEps;
		Float AdaGradEps;
		Float AdamEps;
		Float AdamBeta2;
		Float AdamaxEps;
		Float AdamaxBeta2;
		Float RMSPropEps;
		Float RAdamEps;
		Float RAdamBeta1;
		Float RAdamBeta2;
		Float NeuronsMin;
		Float NeuronsMax;
		Float NeuronsStdDev;
		Float NeuronsMean;
		Float WeightsMin;
		Float WeightsMax;
		Float WeightsStdDev;
		Float WeightsMean;
		Float BiasesMin;
		Float BiasesMax;
		Float BiasesStdDev;
		Float BiasesMean;
		std::atomic<bool> LockUpdate;
		std::atomic<bool> RefreshingStats;
		std::chrono::duration<Float> fpropTime;
		std::chrono::duration<Float> bpropTime;
		std::chrono::duration<Float> updateTime;
		std::unique_ptr<dnnl::memory::desc> DstMemDesc;
		std::unique_ptr<dnnl::memory::desc> DiffDstMemDesc;
		std::unique_ptr<dnnl::memory::desc> WeightsMemDesc;
		std::unique_ptr<dnnl::memory::desc> PersistWeightsMemDesc;

		Layer(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const LayerTypes layerType, const size_t weightCount, const size_t biasCount, const size_t c, const size_t d, const size_t h, const size_t w, const size_t padD, const size_t padH, const size_t padW, const std::vector<Layer*>& inputs, const bool hasBias = false);
		virtual ~Layer() = default;

		auto GetDescriptionHeader() const
		{
			auto description = std::string("");

			description.append(" Type:" + dtab + std::string(magic_enum::enum_name<LayerTypes>(LayerType)));

			if (LayerType != LayerTypes::Input)
			{
				description.append(nwl + " Inputs:" + tab);
				for (auto i = 0ull; i < Inputs.size(); i++)
					description.append((i == 0 ? "" : ",") + Inputs[i]->Name);
			}

			description.append(nwl + " Features:" + tab + std::to_string(C) + "x" + std::to_string(H) + "x" + std::to_string(W));
			description.append(nwl + " Neurons:" + tab + std::to_string(CDHW));
			
			return description;
		}

		auto GetWeightsDescription(const bool visible = true) const
		{
			auto description = std::string("");
			
			if (visible)
			{
				description.append(nwl + " Weights:" + tab + std::to_string(WeightCount));
				description.append(nwl + "  lr mult:" + tab + FloatToString(WeightsLRM));
				description.append(nwl + "  wd mult:" + tab + FloatToString(WeightsWDM));

				if (HasBias)
				{
					description.append(nwl + " Biases:" + tab + std::to_string(BiasCount));
					description.append(nwl + "  lr mult:" + tab + FloatToString(BiasesLRM));
					description.append(nwl + "  wd mult:" + tab + FloatToString(BiasesWDM));
				}
			}

			return description;
		}

		void SetParameters(const bool useDefaults, const Fillers weightsFiller, const Float weightsScale, const Float weighsLRM, const Float weightsWDM, const Fillers biasesFiller, const Float biasesScale, const Float biasesLRM, const Float biasesWDM);
		bool RefreshStatistics(const size_t batchSize);
		void ResetGradients();
		void CheckOptimizer(const Optimizers optimizer);
		void ResetOptimizer(const Optimizers optimizer);
		void SetOptimizer(const Optimizers optimizer);
		
		void UpdateWeights(const TrainingRate& rate, const Optimizers optimizer, const bool disableLocking);

		inline void AdaDelta(const TrainingRate& rate);
		inline void AdaGrad(const TrainingRate& rate);
		inline void Adam(const TrainingRate& rate);
		inline void Adamax(const TrainingRate& rate);
		inline void NAG(const TrainingRate& rate);
		inline void RMSProp(const TrainingRate& rate);
		inline void SGD(const TrainingRate& rate);
		inline void SGDMomentum(const TrainingRate& rate);
		inline void RAdam(const TrainingRate& rate);

		virtual void ResetWeights(const Fillers weightFiller, const Float weightFillerScale, const Fillers biasFiller, const Float biasFillerScale);
		
		/*size_t OffsetPaddedMem(const size_t n, const size_t c, const size_t h, const size_t w) const
		{
			return n * PaddedCDHW + (c / VectorSize) * HW * VectorSize + h * W * VectorSize + w * VectorSize + (c % VectorSize);
		}*/
		
		virtual std::string GetDescription() const = 0;

		virtual size_t FanIn() const = 0;
		
		virtual size_t FanOut() const = 0;
		
		virtual void InitializeDescriptors(const size_t) = 0;
			
		virtual bool Lockable() const
		{
			return WeightCount > 0;
		}

#ifdef DNN_LEAN
		inline void ZeroGradient(const size_t batchSize)
		{
			ZeroFloatVectorAllocate(InputLayer->NeuronsD1, batchSize * InputLayer->PaddedCDHW);
		}

		inline void ZeroGradientMulti(const size_t batchSize)
		{
			for (auto i = 0ull; i < Inputs.size(); i++)
				ZeroFloatVectorAllocate(Inputs[i]->NeuronsD1, batchSize * Inputs[i]->PaddedCDHW);
		}

		inline void ReleaseGradient()
		{
			NeuronsD1.~vector();
		}
#endif // DNN_LEAN

		virtual void SetBatchSize(const size_t batchSize)
		{
			while (RefreshingStats.load())
				std::this_thread::sleep_for(std::chrono::milliseconds(250));
			
			ZeroFloatVectorAllocate(Neurons, batchSize * PaddedCDHW);
#ifndef DNN_LEAN
			ZeroFloatVectorAllocate(NeuronsD1, batchSize * PaddedCDHW);
#else
			ReleaseGradient();
#endif // DNN_LEAN

			InitializeDescriptors(batchSize);
		}

		virtual void ForwardProp(const size_t batchSize, const bool training) = 0;

		virtual void BackwardProp(const size_t batchSize) = 0;
	
		virtual void Save(std::ostream& os, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD)
		{
			if (HasWeights)
			{
				os.write(reinterpret_cast<const char*>(&LockUpdate), sizeof(std::atomic<bool>));
				
				if (*WeightsMemDesc != *PersistWeightsMemDesc)
				{
					auto weights = FloatVector(WeightCount);
					auto memWeights = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());
					auto weightsMem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weights.data());
					dnnl::reorder(memWeights, weightsMem).execute(Device.second, { {DNNL_ARG_FROM, memWeights}, {DNNL_ARG_TO, weightsMem} });
					Device.second.wait();
					
					os.write(reinterpret_cast<const char*>(weights.data()), std::streamsize(WeightCount * sizeof(Float)));
					if (HasBias)
						os.write(reinterpret_cast<const char*>(Biases.data()), std::streamsize(BiasCount * sizeof(Float)));
					
					if (persistOptimizer)
						switch (optimizer)
						{
						case Optimizers::AdaDelta:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::Adam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							auto memWeightsPar1 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							auto memWeightsPar2 = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							os.write(reinterpret_cast<const char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
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
						case Optimizers::Adam:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							os.write(reinterpret_cast<const char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								os.write(reinterpret_cast<const char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B1), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&B2), std::streamsize(sizeof(Float)));
							os.write(reinterpret_cast<const char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
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
					auto memWeights = dnnl::memory(*PersistWeightsMemDesc, Device.first, weights.data());
					auto weightsMem = dnnl::memory(*WeightsMemDesc, Device.first, Weights.data());
					dnnl::reorder(memWeights, weightsMem).execute(Device.second, { {DNNL_ARG_FROM, memWeights}, {DNNL_ARG_TO, weightsMem} });
					Device.second.wait();
					if (HasBias)
						is.read(reinterpret_cast<char*>(Biases.data()), std::streamsize(BiasCount * sizeof(Float)));
					
					if (persistOptimizer)
						switch (optimizer)
						{
						case Optimizers::AdaDelta:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::Adam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							auto weightsPar1 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar1 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar1.data());
							auto weightsPar1Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar1.data());
							dnnl::reorder(memWeightsPar1, weightsPar1Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar1}, {DNNL_ARG_TO, weightsPar1Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));

							auto weightsPar2 = FloatVector(WeightCount);
							is.read(reinterpret_cast<char*>(weightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							auto memWeightsPar2 = dnnl::memory(*PersistWeightsMemDesc, Device.first, weightsPar2.data());
							auto weightsPar2Mem = dnnl::memory(*WeightsMemDesc, Device.first, WeightsPar2.data());
							dnnl::reorder(memWeightsPar2, weightsPar2Mem).execute(Device.second, { {DNNL_ARG_FROM, memWeightsPar2}, {DNNL_ARG_TO, weightsPar2Mem} });
							Device.second.wait();
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));

							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
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
						case Optimizers::Adam:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::Adamax:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
						}
						break;

						case Optimizers::AdaGrad:
						case Optimizers::NAG:
						case Optimizers::RMSProp:
						case Optimizers::SGDMomentum:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
						}
						break;

						case Optimizers::RAdam:
						{
							is.read(reinterpret_cast<char*>(WeightsPar1.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar1.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(WeightsPar2.data()), std::streamsize(WeightCount * sizeof(Float)));
							if (HasBias)
								is.read(reinterpret_cast<char*>(BiasesPar2.data()), std::streamsize(BiasCount * sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B1), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&B2), std::streamsize(sizeof(Float)));
							is.read(reinterpret_cast<char*>(&Moments), std::streamsize(sizeof(size_t)));
						}
						break;
						}
				}
			}
		}

		virtual size_t GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const
		{
			size_t weightsSize = 0;

			if (HasWeights)
			{
				weightsSize += sizeof(std::atomic<bool>);

				if (persistOptimizer)
				{
					switch (optimizer)
					{
					case Optimizers::AdaDelta:
					{
						weightsSize += 3 * WeightCount * sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::Adam:
					{
						weightsSize += 3 * WeightCount * sizeof(Float) + 2 * sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::Adamax:
					{
						weightsSize += 3 * WeightCount * sizeof(Float) + sizeof(Float);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
					}
					break;

					case Optimizers::AdaGrad:
					case Optimizers::NAG:
					case Optimizers::RMSProp:
					case Optimizers::SGDMomentum:
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

					case Optimizers::RAdam:
					{
						weightsSize += 3 * WeightCount * sizeof(Float) + 2 * sizeof(Float) + sizeof(size_t);
						if (HasBias)
							weightsSize += 3 * BiasCount * sizeof(Float);
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
					
		virtual size_t GetNeuronsSize(const size_t batchSize) const
		{
			auto neuronsSize = 0ull;

#ifndef DNN_LEAN
			neuronsSize += PaddedCDHW * batchSize * sizeof(Float) * 2;
#else
			neuronsSize += PaddedCDHW * batchSize * sizeof(Float);
#endif // DNN_LEAN

			return neuronsSize;
		}

		virtual ByteVector GetImage(const Byte fillColor) { return ByteVector(); }
	};
}