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
		std::vector<LabelInfo> sampleLabel;
		std::vector<std::vector<LabelInfo>> sampleLabels;
		bool reorderFwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const Costs CostFunction;
		const UInt GroupIndex;
		const UInt LabelIndex;
		const Float LabelTrue;
		const Float LabelFalse;
		const Float Weight;
		const Float Eps;
		const bool IsLogSoftmax;
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
			reorderFwdSrc(false),
			reorderBwdDiffSrc(false),
			CostFunction(cost),
			GroupIndex(groupIndex),
			LabelIndex(labelIndex),
			LabelTrue(labelTrue),
			LabelFalse(labelFalse),
			Weight(weight),
			Eps(eps),
			IsLogSoftmax(inputs.size() == 1 && inputs[0]->LayerType == LayerTypes::LogSoftmax)
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

			description.append(nwl + std::string(" Cost:       ") + tab + std::string(magic_enum::enum_name<Costs>(CostFunction)));
			description.append(nwl + std::string(" Channels:   ") + tab + std::to_string(C));
			description.append(nwl + std::string(" LabelTrue:  ") + tab + FloatToStringFixed(LabelTrue));
			description.append(nwl + std::string(" LabelFalse: ") + tab + FloatToStringFixed(LabelFalse));
			description.append(nwl + std::string(" Weight:     ") + tab + FloatToStringFixed(Weight));
			if (CostFunction == Costs::MeanAbsoluteEpsError || CostFunction == Costs::CategoricalCrossEntropy)
				description.append(nwl + std::string(" Epsilon:    ") + tab + FloatToStringFixed(Eps, 6));

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
			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				ChosenFormat = PlainFmt;
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(1) , dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, PlainFmt));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) , dnnl::memory::dim(1) , dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, PlainFmt));
			}

			reorderFwdSrc = *DstMemDesc != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = *DiffDstMemDesc != *InputLayerBwd->DiffDstMemDesc;
		}

		void SetSampleLabel(const std::vector<LabelInfo>& label)
		{
			sampleLabel = label;
		}

		void SetSampleLabels(const std::vector<std::vector<LabelInfo>>& labels)
		{
			sampleLabels = labels;
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

			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(*DstMemDesc, Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			Float* inputNeurons = (Float*)srcMem.get_data_handle();

			switch (CostFunction)
			{
			case Costs::BinaryCrossEntropy:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						Neurons[c] = -LabelFalse * std::log(inputNeurons[c]) - (Float(1) - LabelFalse) * std::log(Float(1) - inputNeurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					Neurons[label] = -LabelTrue * std::log(inputNeurons[label]) - (Float(1) - LabelTrue) * std::log(Float(1) - inputNeurons[label]);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						Neurons[nc] = -LabelFalse * std::log(inputNeurons[nc]) - (Float(1) - LabelFalse) * std::log(Float(1) - inputNeurons[nc]);
#ifndef DNN_LEAN
						NeuronsD1[nc] = Float(0);
#endif
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = -LabelTrue * std::log(inputNeurons[label]) - (Float(1) - LabelTrue) * std::log(Float(1) - inputNeurons[label]);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::CategoricalCrossEntropy:
			{
				if (IsLogSoftmax)
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
						const auto label = sampleLabel[LabelIndex].LabelA;
						Neurons[label] = -inputNeurons[label];
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
							const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
							Neurons[label] = -inputNeurons[label];
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
						const auto label = sampleLabel[LabelIndex].LabelA;
						Neurons[label] = -std::log(inputNeurons[label]);
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
							const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
							Neurons[label] = -std::log(inputNeurons[label]);
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
						Neurons[c] = std::abs(inputNeurons[c] - LabelFalse);

					const auto label = sampleLabel[LabelIndex].LabelA;
					Neurons[label] = std::abs(inputNeurons[label] - LabelTrue);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						Neurons[nc] = std::abs(inputNeurons[nc] - LabelFalse);

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = std::abs(inputNeurons[label] - LabelTrue);
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
						const auto diff = std::abs(inputNeurons[c] - LabelFalse);
						Neurons[c] = diff > Eps ? diff : Float(0);
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto diff = std::abs(inputNeurons[label] - LabelTrue);
					Neurons[label] = diff > Eps ? diff : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto diff = std::abs(inputNeurons[nc] - LabelFalse);
						Neurons[nc] = diff > Eps ? diff : Float(0);
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto diff = std::abs(inputNeurons[label] - LabelTrue);
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
						Neurons[c] = Square<Float>(inputNeurons[c] - LabelFalse);

					const auto label = sampleLabel[LabelIndex].LabelA;
					Neurons[label] = Square<Float>(inputNeurons[label] - LabelTrue);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						Neurons[nc] = Square<Float>(inputNeurons[nc] - LabelFalse);

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = Square<Float>(inputNeurons[label] - LabelTrue);
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
						const auto ty = LabelFalse * inputNeurons[c];
						Neurons[c] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto ty = LabelTrue * inputNeurons[label];
					Neurons[label] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto ty = LabelFalse * inputNeurons[nc];
						Neurons[nc] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
#ifndef DNN_LEAN
						NeuronsD1[nc] = Float(0);
#endif
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto ty = LabelTrue * inputNeurons[label];
						Neurons[label] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
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

			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(*DstMemDesc, Device.engine) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
				Device.stream.wait();
			}

			auto memDiffSrc = dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(*DiffDstMemDesc, Device.engine) : memDiffSrc;

			Float* inputNeurons = (Float*)srcMem.get_data_handle();
			Float* inputDiffNeurons = (Float*)diffSrcMem.get_data_handle();

			switch (CostFunction)
			{
			case Costs::BinaryCrossEntropy:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						inputDiffNeurons[c] = (inputNeurons[c] - LabelFalse) / (inputNeurons[c] * (Float(1) - inputNeurons[c]));

					const auto label = sampleLabel[LabelIndex].LabelA;
					inputDiffNeurons[label] = (inputNeurons[label] - LabelTrue) / (inputNeurons[label] * (Float(1) - inputNeurons[label]));
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						inputDiffNeurons[nc] = (inputNeurons[nc] - LabelFalse) / (inputNeurons[nc] * (Float(1) - inputNeurons[nc]));

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						inputDiffNeurons[label] = (inputNeurons[label] - LabelTrue) / (inputNeurons[label] * (Float(1) - inputNeurons[label]));
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;

			case Costs::CategoricalCrossEntropy:
			{
				if (IsLogSoftmax)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						for (auto c = 0ull; c < C; c++)
							inputDiffNeurons[c] = std::exp(inputNeurons[c]) - (Eps / C);

						const auto labelA = sampleLabel[LabelIndex].LabelA;
						const auto labelB = sampleLabel[LabelIndex].LabelB;
						const auto weightA = sampleLabel[LabelIndex].Lambda;
						const auto weightB = ((weightA != Float(1)) && (sampleLabel[LabelIndex].LabelA != sampleLabel[LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
						inputDiffNeurons[labelA] = std::exp(inputNeurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
						inputDiffNeurons[labelB] = std::exp(inputNeurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
							inputDiffNeurons[nc] = std::exp(inputNeurons[nc]) - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = sampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = sampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = sampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (sampleLabels[n][LabelIndex].LabelA != sampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							inputDiffNeurons[labelA] = std::exp(inputNeurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							inputDiffNeurons[labelB] = std::exp(inputNeurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
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
							inputDiffNeurons[c] = inputNeurons[c] - (Eps / C);

						const auto labelA = sampleLabel[LabelIndex].LabelA;
						const auto labelB = sampleLabel[LabelIndex].LabelB;
						const auto weightA = sampleLabel[LabelIndex].Lambda;
						const auto weightB = ((weightA != Float(1)) && (sampleLabel[LabelIndex].LabelA != sampleLabel[LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
						inputDiffNeurons[labelA] = (inputNeurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
						inputDiffNeurons[labelB] = (inputNeurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
							inputDiffNeurons[nc] = inputNeurons[nc] - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = sampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = sampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = sampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (sampleLabels[n][LabelIndex].LabelA != sampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							inputDiffNeurons[labelA] = (inputNeurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							inputDiffNeurons[labelB] = (inputNeurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
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
						const auto sign = inputNeurons[c] - LabelFalse;
						inputDiffNeurons[c] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto sign = inputNeurons[label] - LabelTrue;
					inputDiffNeurons[label] = sign < 0 ? -factor : sign > 0 ? factor : 0;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto sign = inputNeurons[nc] - LabelFalse;
						inputDiffNeurons[nc] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto sign = inputNeurons[label] - LabelTrue;
						inputDiffNeurons[label] = sign < 0 ? -factor : sign > 0 ? factor : 0;
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
						const auto sign = inputNeurons[c] - LabelFalse;
						inputDiffNeurons[c] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto sign = inputNeurons[label] - LabelTrue;
					inputDiffNeurons[label] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto sign = inputNeurons[nc] - LabelFalse;
						inputDiffNeurons[nc] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto sign = inputNeurons[label] - LabelTrue;
						inputDiffNeurons[label] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
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
						inputDiffNeurons[c] = (inputNeurons[c] - LabelFalse) * factor;

					const auto label = sampleLabel[LabelIndex].LabelA;
					inputDiffNeurons[label] = (inputNeurons[label] - LabelTrue) * factor;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						inputDiffNeurons[nc] = (inputNeurons[nc] - LabelFalse) * factor;

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						inputDiffNeurons[label] = (inputNeurons[label] - LabelTrue) * factor;
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
						const auto ty = LabelFalse * inputNeurons[c];
						inputDiffNeurons[c] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto ty = LabelTrue * inputNeurons[label];
					inputDiffNeurons[label] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto ty = LabelFalse * inputNeurons[nc];
						inputDiffNeurons[nc] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto ty = LabelTrue * inputNeurons[label];
						inputDiffNeurons[label] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			break;
			}

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
				Device.stream.wait();
			}

#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}