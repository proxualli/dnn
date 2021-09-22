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
	private:
		std::vector<LabelInfo> SampleLabel;
		std::vector<std::vector<LabelInfo>> SampleLabels;
		bool isLogSoftmax;

	public:
		const Costs CostFunction;
		const UInt GroupIndex;
		const UInt LabelIndex;
		const Float LabelTrue;
		const Float LabelFalse;
		const Float Weight;
		const Float Eps;

		UInt TrainErrors;
		Float TrainLoss;
		Float AvgTrainLoss;
		Float TrainErrorPercentage;
		UInt TestErrors;
		Float TestLoss;
		Float AvgTestLoss;
		Float TestErrorPercentage;

		std::vector<std::vector<UInt>> ConfusionMatrix;

		Cost(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Costs cost, const UInt groupIndex, const UInt labelIndex, const UInt c, const std::vector<Layer*>& inputs, const Float labelTrue, const Float labelFalse, const Float weight, const Float eps) :
			Layer(device, format, name, LayerTypes::Cost, 0, 0, c, 1, 1, 1, 0, 0, 0, inputs),
			CostFunction(cost),
			GroupIndex(groupIndex),
			LabelIndex(labelIndex),
			LabelTrue(labelTrue),
			LabelFalse(labelFalse),
			Weight(weight),
			Eps(eps),
			isLogSoftmax(false)
		{
			assert(Inputs.size() == 1);

			InputLayer->LayerBeforeCost = true;

			TrainErrors = 0;
			TrainErrorPercentage = Float(0);
			TrainLoss = Float(0);
			AvgTrainLoss = Float(0);

			TestErrors = 0;
			TestErrorPercentage = Float(0);
			TestLoss = Float(0);
			AvgTestLoss = Float(0);

			ConfusionMatrix = std::vector<std::vector<UInt>>(C, std::vector<UInt>(C, 0));
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Cost:") + dtab + std::string(magic_enum::enum_name<Costs>(CostFunction)));
			description.append(nwl + std::string(" Channels:") + tab + std::to_string(C));
			description.append(nwl + std::string(" LabelTrue:") + tab + FloatToStringFixed(LabelTrue));
			description.append(nwl + std::string(" LabelFalse:") + tab + FloatToStringFixed(LabelFalse));
			description.append(nwl + std::string(" Weight:") + tab + FloatToStringFixed(Weight));
			if (CostFunction == Costs::MeanAbsoluteEpsError || CostFunction == Costs::CategoricalCrossEntropy)
				description.append(nwl + std::string(" Epsilon:") + tab + FloatToStringFixed(Eps, 6));

			return description;
		}

		UInt FanIn() const final override
		{
			return 1;
		}

		UInt FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const UInt batchSize) final override
		{
			ChosenFormat = dnnl::memory::format_tag::nc;
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ int(batchSize), int(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			isLogSoftmax = static_cast<Activation*>(InputLayer)->ActivationFunction == Activations::LogSoftmax;
		}

		void SetSampleLabel(const std::vector<LabelInfo>& sampleLabel)
		{
			SampleLabel = sampleLabel;
		}

		void SetSampleLabels(const std::vector<std::vector<LabelInfo>>& sampleLabels)
		{
			SampleLabels = sampleLabels;
		}

		void Reset()
		{
			TrainErrors = 0;
			TrainErrorPercentage = Float(0);
			TrainLoss = Float(0);
			AvgTrainLoss = Float(0);

			TestErrors = 0;
			TestErrorPercentage = Float(0);
			TestLoss = Float(0);
			AvgTestLoss = Float(0);

			ConfusionMatrix = std::vector<std::vector<UInt>>(C, std::vector<UInt>(C, 0));
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			DNN_UNREF_PAR(training);

			switch (CostFunction)
			{
			case Costs::BinaryCrossEntropy:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						Neurons[c] = -LabelFalse * std::log(InputLayer->Neurons[c]) - (Float(1) - LabelFalse) * std::log(Float(1) - InputLayer->Neurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif
					}
					const auto label = SampleLabel[LabelIndex].LabelA;
					Neurons[label] = -LabelTrue * std::log(InputLayer->Neurons[label]) - (Float(1) - LabelTrue) * std::log(Float(1) - InputLayer->Neurons[label]);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						Neurons[nc] = -LabelFalse * std::log(InputLayer->Neurons[nc]) - (Float(1) - LabelFalse) * std::log(Float(1) - InputLayer->Neurons[nc]);
#ifndef DNN_LEAN
						NeuronsD1[nc] = Float(0);
#endif
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = -LabelTrue * std::log(InputLayer->Neurons[label]) - (Float(1) - LabelTrue) * std::log(Float(1) - InputLayer->Neurons[label]);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::CategoricalCrossEntropy:
			{
				if (isLogSoftmax)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						for (auto c = 0ull; c < C; c++)
						{
							Neurons[c] = Float(0);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif
						}
						const auto label = SampleLabel[LabelIndex].LabelA;
						Neurons[label] = -InputLayer->Neurons[label];
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
						{
							Neurons[nc] = Float(0);
#ifndef DNN_LEAN
							NeuronsD1[nc] = Float(0);
#endif
						}
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
							Neurons[label] = -InputLayer->Neurons[label];
						}
#ifdef DNN_STOCHASTIC
					}
#endif
				}
				else
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						for (auto c = 0ull; c < C; c++)
						{
							Neurons[c] = Float(0);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif
						}
						const auto label = SampleLabel[LabelIndex].LabelA;
						Neurons[label] = -std::log(InputLayer->Neurons[label]);
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
						{
							Neurons[nc] = Float(0);
#ifndef DNN_LEAN
							NeuronsD1[nc] = Float(0);
#endif
						}
						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
							Neurons[label] = -std::log(InputLayer->Neurons[label]);
						}
#ifdef DNN_STOCHASTIC
					}
#endif
				}
			}
			break;

			case Costs::MeanAbsoluteError:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						Neurons[c] = std::abs(InputLayer->Neurons[c] - LabelFalse);

					const auto label = SampleLabel[LabelIndex].LabelA;
					Neurons[label] = std::abs(InputLayer->Neurons[label] - LabelTrue);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						Neurons[nc] = std::abs(InputLayer->Neurons[nc] - LabelFalse);

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = std::abs(InputLayer->Neurons[label] - LabelTrue);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::MeanAbsoluteEpsError:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto diff = std::abs(InputLayer->Neurons[c] - LabelFalse);
						Neurons[c] = diff > Eps ? diff : Float(0);
					}
					const auto label = SampleLabel[LabelIndex].LabelA;
					const auto diff = std::abs(InputLayer->Neurons[label] - LabelTrue);
					Neurons[label] = diff > Eps ? diff : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto diff = std::abs(InputLayer->Neurons[nc] - LabelFalse);
						Neurons[nc] = diff > Eps ? diff : Float(0);
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto diff = std::abs(InputLayer->Neurons[label] - LabelTrue);
						Neurons[label] = diff > Eps ? diff : Float(0);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::MeanSquaredError:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						Neurons[c] = FloatSquare(InputLayer->Neurons[c] - LabelFalse);

					const auto label = SampleLabel[LabelIndex].LabelA;
					Neurons[label] = FloatSquare(InputLayer->Neurons[label] - LabelTrue);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						Neurons[nc] = FloatSquare(InputLayer->Neurons[nc] - LabelFalse);

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = FloatSquare(InputLayer->Neurons[label] - LabelTrue);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::SmoothHinge:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto ty = LabelFalse * InputLayer->Neurons[c];
						Neurons[c] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? FloatSquare(1 - ty) * Float(0.5) : Float(0);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif
					}
					const auto label = SampleLabel[LabelIndex].LabelA;
					const auto ty = LabelTrue * InputLayer->Neurons[label];
					Neurons[label] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? FloatSquare(1 - ty) * Float(0.5) : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto ty = LabelFalse * InputLayer->Neurons[nc];
						Neurons[nc] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? FloatSquare(1 - ty) * Float(0.5) : Float(0);
#ifndef DNN_LEAN
						NeuronsD1[nc] = Float(0);
#endif
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto ty = LabelTrue * InputLayer->Neurons[label];
						Neurons[label] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? FloatSquare(1 - ty) * Float(0.5) : Float(0);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			switch (CostFunction)
			{
			case Costs::BinaryCrossEntropy:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] = (InputLayerOriginal->Neurons[c] - LabelFalse) / (InputLayerOriginal->Neurons[c] * (Float(1) - InputLayerOriginal->Neurons[c]));

					const auto label = SampleLabel[LabelIndex].LabelA;
					InputLayer->NeuronsD1[label] = (InputLayerOriginal->Neurons[label] - LabelTrue) / (InputLayerOriginal->Neurons[label] * (Float(1) - InputLayerOriginal->Neurons[label]));
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						InputLayer->NeuronsD1[nc] = (InputLayerOriginal->Neurons[nc] - LabelFalse) / (InputLayerOriginal->Neurons[nc] * (Float(1) - InputLayerOriginal->Neurons[nc]));

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						InputLayer->NeuronsD1[label] = (InputLayerOriginal->Neurons[label] - LabelTrue) / (InputLayerOriginal->Neurons[label] * (Float(1) - InputLayerOriginal->Neurons[label]));
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::CategoricalCrossEntropy:
			{
				if (isLogSoftmax)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						for (auto c = 0ull; c < C; c++)
							InputLayer->NeuronsD1[c] = std::exp(InputLayerOriginal->Neurons[c]) - (Eps / C);

						const auto labelA = SampleLabel[LabelIndex].LabelA;
						const auto labelB = SampleLabel[LabelIndex].LabelB;
						const auto weightA = SampleLabel[LabelIndex].Lambda;
						const auto weightB = ((weightA != Float(1)) && (SampleLabel[LabelIndex].LabelA != SampleLabel[LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
						InputLayer->NeuronsD1[labelA] = std::exp(InputLayerOriginal->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
						InputLayer->NeuronsD1[labelB] = std::exp(InputLayerOriginal->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
							InputLayer->NeuronsD1[nc] = std::exp(InputLayerOriginal->Neurons[nc]) - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = SampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = SampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = SampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (SampleLabels[n][LabelIndex].LabelA != SampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							InputLayer->NeuronsD1[labelA] = std::exp(InputLayerOriginal->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							InputLayer->NeuronsD1[labelB] = std::exp(InputLayerOriginal->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
						}
#ifdef DNN_STOCHASTIC
					}
#endif
				}
				else
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						for (auto c = 0ull; c < C; c++)
							InputLayer->NeuronsD1[c] = InputLayerOriginal->Neurons[c] - (Eps / C);

						const auto labelA = SampleLabel[LabelIndex].LabelA;
						const auto labelB = SampleLabel[LabelIndex].LabelB;
						const auto weightA = SampleLabel[LabelIndex].Lambda;
						const auto weightB = ((weightA != Float(1)) && (SampleLabel[LabelIndex].LabelA != SampleLabel[LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
						InputLayer->NeuronsD1[labelA] = (InputLayerOriginal->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
						InputLayer->NeuronsD1[labelB] = (InputLayerOriginal->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
							InputLayer->NeuronsD1[nc] = InputLayerOriginal->Neurons[nc] - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = SampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = SampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = SampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (SampleLabels[n][LabelIndex].LabelA != SampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							InputLayer->NeuronsD1[labelA] = (InputLayerOriginal->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							InputLayer->NeuronsD1[labelB] = (InputLayerOriginal->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
						}
#ifdef DNN_STOCHASTIC
					}
#endif
				}
			}
			break;

			case Costs::MeanAbsoluteError:
			{
				const auto factor = Float(1) / static_cast<Float>(C);

#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto sign = InputLayer->InputLayerOriginal->Neurons[c] - LabelFalse;
						InputLayer->NeuronsD1[c] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}
					const auto label = SampleLabel[LabelIndex].LabelA;
					const auto sign = InputLayer->InputLayerOriginal->Neurons[label] - LabelTrue;
					InputLayer->NeuronsD1[label] = sign < 0 ? -factor : sign > 0 ? factor : 0;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto sign = InputLayer->InputLayerOriginal->Neurons[nc] - LabelFalse;
						InputLayer->NeuronsD1[nc] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto sign = InputLayerOriginal->Neurons[label] - LabelTrue;
						InputLayer->NeuronsD1[label] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::MeanAbsoluteEpsError:
			{
				const auto factor = Float(1) / static_cast<Float>(C);

#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto sign = InputLayerOriginal->Neurons[c] - LabelFalse;
						InputLayer->NeuronsD1[c] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}
					const auto label = SampleLabel[LabelIndex].LabelA;
					const auto sign = InputLayerOriginal->Neurons[label] - LabelTrue;
					InputLayer->NeuronsD1[label] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto sign = InputLayerOriginal->Neurons[nc] - LabelFalse;
						InputLayer->NeuronsD1[nc] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto sign = InputLayerOriginal->Neurons[label] - LabelTrue;
						InputLayer->NeuronsD1[label] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::MeanSquaredError:
			{
				const auto factor = Float(2) / static_cast<Float>(C);

#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] = (InputLayerOriginal->Neurons[c] - LabelFalse) * factor;

					const auto label = SampleLabel[LabelIndex].LabelA;
					InputLayer->NeuronsD1[label] = (InputLayerOriginal->Neurons[label] - LabelTrue) * factor;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						InputLayer->NeuronsD1[nc] = (InputLayerOriginal->Neurons[nc] - LabelFalse) * factor;

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						InputLayer->NeuronsD1[label] = (InputLayerOriginal->Neurons[label] - LabelTrue) * factor;
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::SmoothHinge:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						const auto ty = LabelFalse * InputLayerOriginal->Neurons[c];
						InputLayer->NeuronsD1[c] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
					const auto label = SampleLabel[LabelIndex].LabelA;
					const auto ty = LabelTrue * InputLayerOriginal->Neurons[label];
					InputLayer->NeuronsD1[label] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto ty = LabelFalse * InputLayerOriginal->Neurons[nc];
						InputLayer->NeuronsD1[nc] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = SampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto ty = LabelTrue * InputLayerOriginal->Neurons[label];
						InputLayer->NeuronsD1[label] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}