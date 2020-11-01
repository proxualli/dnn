#pragma once
#include "Layer.h"

namespace dnn
{
	enum class Costs
	{
		BinaryCrossEntropy = 0,
		CategoricalCrossEntropy = 1,
		MeanAbsoluteEpsError = 2,
		MeanAbsoluteError = 3,
		MeanSquaredError = 4,
		SmoothHinge = 5
	};

	class Cost : public Layer
	{
	public:
		Cost(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Costs cost, const size_t groupIndex, const size_t labelIndex, const size_t c, const std::vector<Layer*>& inputs, const Float labelTrue = Float(0.9), const Float labelFalse = Float(0.1), const Float weight = Float(1), const Float eps = Float(0.001));

		const Costs CostFunction;
		const size_t GroupIndex;
		const size_t LabelIndex;
		const Float LabelTrue;
		const Float LabelFalse;
		const Float Weight;
		const Float Eps;
		
		size_t TrainErrors;
		Float TrainLoss;
		Float AvgTrainLoss;
		Float TrainErrorPercentage;

		size_t TestErrors;
		Float TestLoss;
		Float AvgTestLoss;
		Float TestErrorPercentage;

		std::vector<std::vector<size_t>> ConfusionMatrix;
		
		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void SetSampleLabel(const std::vector<size_t>& SampleLabel);
		void SetSampleLabels(const std::vector<std::vector<size_t>>& SampleLabels);

		void Reset();

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::vector<size_t> SampleLabel;
		std::vector<std::vector<size_t>> SampleLabels;
		bool isLogSoftmax;
	};
}