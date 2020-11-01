#pragma once
#include "Layer.h"

namespace dnn
{
	class BatchNormRelu final : public Layer
	{
	public:
		BatchNormRelu(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const std::vector<Layer*>& inputs, const bool scaling = true, const Float momentum = Float(0.99), const Float eps = Float(1e-04), const bool hasBias = true);
				
		const bool Scaling;
		const Float Eps;
		const Float Momentum;
		const Float OneMinusMomentum;

		FloatVector Mean;
		FloatVector RunningMean;
		FloatVector Variance;
		FloatVector RunningVariance;
		FloatVector InvStdDev;
		
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		bool Lockable() const final override
		{
			return WeightCount > 0 && Scaling;
		}

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

		ByteVector GetImage(const Byte fillColor) final override;

		void ResetWeights(const Fillers weightFiller, const Float weightFillerScale, const Fillers biasFiller, const Float biasFillerScale) override
		{
			Weights = FloatVector(PaddedC, Float(1));
			Biases = FloatVector(PaddedC, Float(0));

			RunningMean = FloatVector(PaddedC, Float(0));
			RunningVariance = FloatVector(PaddedC, Float(1));
		}

		void Save(std::ostream& os, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			os.write(reinterpret_cast<const char*>(RunningMean.data()), std::streamsize(C * sizeof(Float)));
			os.write(reinterpret_cast<const char*>(RunningVariance.data()), std::streamsize(C * sizeof(Float)));

			if (Scaling)
			{
				for (size_t c = 0; c < C; c++)
				{
					Weights[c] = ScaleShift[c];
					Biases[c] = ScaleShift[C + c];
				}
			}
			
			Layer::Save(os, persistOptimizer, optimizer);
		}

		void Load(std::istream& is, const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) override
		{
			is.read(reinterpret_cast<char*>(RunningMean.data()), std::streamsize(C * sizeof(Float)));
			is.read(reinterpret_cast<char*>(RunningVariance.data()), std::streamsize(C * sizeof(Float)));

			Layer::Load(is, persistOptimizer, optimizer);

			if (Scaling)
			{
				for (size_t c = 0; c < C; c++)
				{
					ScaleShift[c] = Weights[c];
					ScaleShift[C + c] = Biases[c];
				}
			}
		}

		size_t GetWeightsSize(const bool persistOptimizer = false, const Optimizers optimizer = Optimizers::SGD) const override
		{
			return (2 * C * sizeof(Float)) + Layer::GetWeightsSize(persistOptimizer, optimizer);
		}

		private:
			dnnl::normalization_flags Flags;
			std::unique_ptr<dnnl::batch_normalization_forward::primitive_desc> fwdDesc;
			std::unique_ptr<dnnl::batch_normalization_backward::primitive_desc> bwdDesc;
			std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
			std::unique_ptr<dnnl::memory> WorkspaceMemory;

			std::unique_ptr<dnnl::batch_normalization_forward> fwd;
			std::unique_ptr<dnnl::batch_normalization_backward> bwd;
			std::unique_ptr<dnnl::binary> bwdAdd;

			FloatVector ScaleShift;
			FloatVector DiffScaleShift;

			bool inference;
			bool reorderFwdSrc;
			bool reorderBwdSrc;
			bool reorderBwdDiffSrc;
			bool plainFormat;
	};
}
