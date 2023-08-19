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
			ChosenFormat = PlainFmt;
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(1) , dnnl::memory::dim(1) } ), dnnl::memory::data_type::f32, PlainFmt));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) , dnnl::memory::dim(1) , dnnl::memory::dim(1) }), dnnl::memory::data_type::f32, PlainFmt));
			reorderFwdSrc = *DstMemDesc != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = *DiffDstMemDesc != *InputLayer->DiffDstMemDesc;
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
					const auto label = sampleLabel[LabelIndex].LabelA;
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
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = -LabelTrue * std::log(InputLayer->Neurons[label]) - (Float(1) - LabelTrue) * std::log(Float(1) - InputLayer->Neurons[label]);
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
							const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
							//Neurons[label] = -InputLayer->Neurons[label];
							Neurons[label] = -((Float*)srcMem.get_data_handle())[label];
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
							const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
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

					const auto label = sampleLabel[LabelIndex].LabelA;
					Neurons[label] = std::abs(InputLayer->Neurons[label] - LabelTrue);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						Neurons[nc] = std::abs(InputLayer->Neurons[nc] - LabelFalse);

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
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
					const auto label = sampleLabel[LabelIndex].LabelA;
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
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
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
						Neurons[c] = Square<Float>(InputLayer->Neurons[c] - LabelFalse);

					const auto label = sampleLabel[LabelIndex].LabelA;
					Neurons[label] = Square<Float>(InputLayer->Neurons[label] - LabelTrue);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						Neurons[nc] = Square<Float>(InputLayer->Neurons[nc] - LabelFalse);

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						Neurons[label] = Square<Float>(InputLayer->Neurons[label] - LabelTrue);
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
						Neurons[c] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto ty = LabelTrue * InputLayer->Neurons[label];
					Neurons[label] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto ty = LabelFalse * InputLayer->Neurons[nc];
						Neurons[nc] = ty <= 0 ? Float(0.5) - ty : ty < Float(1) ? Square<Float>(1 - ty) * Float(0.5) : Float(0);
#ifndef DNN_LEAN
						NeuronsD1[nc] = Float(0);
#endif
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto ty = LabelTrue * InputLayer->Neurons[label];
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

			auto memDiffSrc = dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(*DiffDstMemDesc, Device.engine) : memDiffSrc;

			switch (CostFunction)
			{
			case Costs::BinaryCrossEntropy:
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] = (InputLayerFwd->Neurons[c] - LabelFalse) / (InputLayerFwd->Neurons[c] * (Float(1) - InputLayerFwd->Neurons[c]));

					const auto label = sampleLabel[LabelIndex].LabelA;
					InputLayer->NeuronsD1[label] = (InputLayerFwd->Neurons[label] - LabelTrue) / (InputLayerFwd->Neurons[label] * (Float(1) - InputLayerFwd->Neurons[label]));
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						InputLayer->NeuronsD1[nc] = (InputLayerFwd->Neurons[nc] - LabelFalse) / (InputLayerFwd->Neurons[nc] * (Float(1) - InputLayerFwd->Neurons[nc]));

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						InputLayer->NeuronsD1[label] = (InputLayerFwd->Neurons[label] - LabelTrue) / (InputLayerFwd->Neurons[label] * (Float(1) - InputLayerFwd->Neurons[label]));
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
							InputLayer->NeuronsD1[c] = std::exp(InputLayerFwd->Neurons[c]) - (Eps / C);

						const auto labelA = sampleLabel[LabelIndex].LabelA;
						const auto labelB = sampleLabel[LabelIndex].LabelB;
						const auto weightA = sampleLabel[LabelIndex].Lambda;
						const auto weightB = ((weightA != Float(1)) && (sampleLabel[LabelIndex].LabelA != sampleLabel[LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
						InputLayer->NeuronsD1[labelA] = std::exp(InputLayerFwd->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
						InputLayer->NeuronsD1[labelB] = std::exp(InputLayerFwd->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
					}
					else
					{
#endif
						/*for (auto nc = 0ull; nc < C * batchSize; nc++)
							InputLayer->NeuronsD1[nc] = std::exp(InputLayerFwd->Neurons[nc]) - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = sampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = sampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = sampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (sampleLabels[n][LabelIndex].LabelA != sampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							InputLayer->NeuronsD1[labelA] = std::exp(InputLayerFwd->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							InputLayer->NeuronsD1[labelB] = std::exp(InputLayerFwd->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
						}*/

						for (auto nc = 0ull; nc < C * batchSize; nc++)
							((Float*)diffSrcMem.get_data_handle())[nc] = std::exp(((Float*)srcMem.get_data_handle())[nc]) - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = sampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = sampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = sampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (sampleLabels[n][LabelIndex].LabelA != sampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							((Float*)diffSrcMem.get_data_handle())[labelA] = std::exp(((Float*)srcMem.get_data_handle())[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							((Float*)diffSrcMem.get_data_handle())[labelB] = std::exp(((Float*)srcMem.get_data_handle())[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
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
							InputLayer->NeuronsD1[c] = InputLayerFwd->Neurons[c] - (Eps / C);

						const auto labelA = sampleLabel[LabelIndex].LabelA;
						const auto labelB = sampleLabel[LabelIndex].LabelB;
						const auto weightA = sampleLabel[LabelIndex].Lambda;
						const auto weightB = ((weightA != Float(1)) && (sampleLabel[LabelIndex].LabelA != sampleLabel[LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
						InputLayer->NeuronsD1[labelA] = (InputLayerFwd->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
						InputLayer->NeuronsD1[labelB] = (InputLayerFwd->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
					}
					else
					{
#endif
						for (auto nc = 0ull; nc < C * batchSize; nc++)
							InputLayer->NeuronsD1[nc] = InputLayerFwd->Neurons[nc] - (Eps / C);

						for (auto n = 0ull; n < batchSize; n++)
						{
							const auto labelA = sampleLabels[n][LabelIndex].LabelA + (n * C);
							const auto labelB = sampleLabels[n][LabelIndex].LabelB + (n * C);
							const auto weightA = sampleLabels[n][LabelIndex].Lambda;
							const auto weightB = ((weightA != Float(1)) && (sampleLabels[n][LabelIndex].LabelA != sampleLabels[n][LabelIndex].LabelB)) ? Float(1) - weightA : Float(1);
							InputLayer->NeuronsD1[labelA] = (InputLayerFwd->Neurons[labelA] * weightA) - ((Float(1) - Eps) + (Eps / C));
							InputLayer->NeuronsD1[labelB] = (InputLayerFwd->Neurons[labelB] * weightB) - ((Float(1) - Eps) + (Eps / C));
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
						const auto sign = InputLayer->InputLayerFwd->Neurons[c] - LabelFalse;
						InputLayer->NeuronsD1[c] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto sign = InputLayer->InputLayerFwd->Neurons[label] - LabelTrue;
					InputLayer->NeuronsD1[label] = sign < 0 ? -factor : sign > 0 ? factor : 0;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto sign = InputLayer->InputLayerFwd->Neurons[nc] - LabelFalse;
						InputLayer->NeuronsD1[nc] = sign < 0 ? -factor : sign > 0 ? factor : 0;
					}

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto sign = InputLayerFwd->Neurons[label] - LabelTrue;
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
						const auto sign = InputLayerFwd->Neurons[c] - LabelFalse;
						InputLayer->NeuronsD1[c] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto sign = InputLayerFwd->Neurons[label] - LabelTrue;
					InputLayer->NeuronsD1[label] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto sign = InputLayerFwd->Neurons[nc] - LabelFalse;
						InputLayer->NeuronsD1[nc] = sign < -Eps ? -factor : sign > Eps ? factor : 0;
					}

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto sign = InputLayerFwd->Neurons[label] - LabelTrue;
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
						InputLayer->NeuronsD1[c] = (InputLayerFwd->Neurons[c] - LabelFalse) * factor;

					const auto label = sampleLabel[LabelIndex].LabelA;
					InputLayer->NeuronsD1[label] = (InputLayerFwd->Neurons[label] - LabelTrue) * factor;
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
						InputLayer->NeuronsD1[nc] = (InputLayerFwd->Neurons[nc] - LabelFalse) * factor;

					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						InputLayer->NeuronsD1[label] = (InputLayerFwd->Neurons[label] - LabelTrue) * factor;
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
						const auto ty = LabelFalse * InputLayerFwd->Neurons[c];
						InputLayer->NeuronsD1[c] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
					const auto label = sampleLabel[LabelIndex].LabelA;
					const auto ty = LabelTrue * InputLayerFwd->Neurons[label];
					InputLayer->NeuronsD1[label] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
				}
				else
				{
#endif
					for (auto nc = 0ull; nc < C * batchSize; nc++)
					{
						const auto ty = LabelFalse * InputLayerFwd->Neurons[nc];
						InputLayer->NeuronsD1[nc] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
					}
					for (auto n = 0ull; n < batchSize; n++)
					{
						const auto label = sampleLabels[n][LabelIndex].LabelA + (n * C);
						const auto ty = LabelTrue * InputLayerFwd->Neurons[label];
						InputLayer->NeuronsD1[label] = ty <= 0 ? -Float(1) : ty < Float(1) ? ty - Float(1) : Float(0);
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