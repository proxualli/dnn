#include "Model.h"

namespace dnn
{
	Activation::Activation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector<Layer*>& inputs, const Float alpha, const Float beta) :
		Layer(device, format, name, LayerTypes::Activation, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, false),
		ActivationFunction(activation),
		Alpha(alpha),
		Beta(beta),
		algorithm(dnnl::algorithm::eltwise_linear),
		reorderFwdSrc(false),
		reorderBwdSrc(false),
		reorderBwdDiffSrc(false)
	{
		assert(Inputs.size() == 1);
	}
	
	std::string Activation::GetDescription() const
	{
		std::string description = GetDescriptionHeader();

		description.append(nwl + " Activation:" + tab + std::string(magic_enum::enum_name<Activations>(ActivationFunction)));
		description.append(nwl + " Alpha:" + dtab + FloatToString(Alpha));
		description.append(nwl + " Beta:" + dtab + FloatToString(Beta));
				
		return description;
	}

	size_t Activation::FanIn() const
	{
		return 1;
	}

	size_t Activation::FanOut() const
	{
		return 1;
	}

	void Activation::InitializeDescriptors(const size_t batchSize)
	{
		if (InputLayer->DstMemDesc->data.ndims == 2)
		{
			Format = dnnl::memory::format_tag::nc;

			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, Format));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, Format));
		}
		else
		{
			if (Format == dnnl::memory::format_tag::any)
				Format = LayerBeforeCost || IsPlainDataFmt(*InputLayer->DstMemDesc) ? PlainFmt : BlockedFmt;
			
			DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
			DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, Format));
		}
		
		bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.first));
		bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));

		switch (ActivationFunction)
		{
		case Activations::LogSoftmax:
		{
			const auto axis = (H == 1 && W == 1) ? 1 : 3;
			fwdDescLogSoftmax = std::make_unique<dnnl::logsoftmax_forward::primitive_desc>(dnnl::logsoftmax_forward::primitive_desc(dnnl::logsoftmax_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.first));
			bwdDescLogSoftmax = std::make_unique<dnnl::logsoftmax_backward::primitive_desc>(dnnl::logsoftmax_backward::primitive_desc(dnnl::logsoftmax_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.first, *fwdDescLogSoftmax));

			fwdLogSoftmax = std::make_unique<dnnl::logsoftmax_forward>(dnnl::logsoftmax_forward(*fwdDescLogSoftmax));
			bwdLogSoftmax = std::make_unique<dnnl::logsoftmax_backward>(dnnl::logsoftmax_backward(*bwdDescLogSoftmax));

			reorderFwdSrc = fwdDescLogSoftmax->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDescLogSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
		}
		break;

		case Activations::Softmax:
		{
			const auto axis = (H == 1 && W == 1) ? 1 : 3;
			fwdDescSoftmax = std::make_unique<dnnl::softmax_forward::primitive_desc>(dnnl::softmax_forward::primitive_desc(dnnl::softmax_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.first));
			bwdDescSoftmax = std::make_unique<dnnl::softmax_backward::primitive_desc>(dnnl::softmax_backward::primitive_desc(dnnl::softmax_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.first, *fwdDescSoftmax));

			fwdSoftmax = std::make_unique<dnnl::softmax_forward>(dnnl::softmax_forward(*fwdDescSoftmax));
			bwdSoftmax = std::make_unique<dnnl::softmax_backward>(dnnl::softmax_backward(*bwdDescSoftmax));

			reorderFwdSrc = fwdDescSoftmax->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDescSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
		}
		break;

		case Activations::HardLogistic:
		case Activations::HardSwish:
		case Activations::Mish:
		case Activations::LogLogistic:
			if (!IsPlainDataFmt(*InputLayer->DstMemDesc) && !IsBlockedDataFmt(*InputLayer->DstMemDesc))
				throw std::invalid_argument("Input memory format not supported for this activation function");
			break;

		default:
		{
			switch (ActivationFunction)
			{
			case Activations::Abs:
				algorithm = dnnl::algorithm::eltwise_abs;
				break;

			case Activations::Clip:
				algorithm = dnnl::algorithm::eltwise_clip;
				break;

			case Activations::BoundedRelu:
				algorithm = dnnl::algorithm::eltwise_bounded_relu;
				break;

			case Activations::Elu:
				algorithm = dnnl::algorithm::eltwise_elu;
				break;

			case Activations::Exp:
				algorithm = dnnl::algorithm::eltwise_exp;
				break;

			case Activations::Gelu:
				algorithm = dnnl::algorithm::eltwise_gelu;
				break;

			case Activations::GeluErf:
				algorithm = dnnl::algorithm::eltwise_gelu_erf;
				break;

			case Activations::Linear:
				algorithm = dnnl::algorithm::eltwise_linear;
				break;

			case Activations::Log:
				algorithm = dnnl::algorithm::eltwise_log;
				break;

			case Activations::Logistic:
				algorithm = dnnl::algorithm::eltwise_logistic;
				break;

			case Activations::Pow:
				algorithm = dnnl::algorithm::eltwise_pow;
				break;

			case Activations::Relu:
				algorithm = dnnl::algorithm::eltwise_relu;
				break;

			case Activations::Round:
				algorithm = dnnl::algorithm::eltwise_round;
				break;

			case Activations::SoftRelu:
				algorithm = dnnl::algorithm::eltwise_soft_relu;
				break;

			case Activations::Sqrt:
				algorithm = dnnl::algorithm::eltwise_sqrt;
				break;

			case Activations::Square:
				algorithm = dnnl::algorithm::eltwise_square;
				break;

			case Activations::Swish:
				algorithm = dnnl::algorithm::eltwise_swish;
				break;

			case Activations::Tanh:
				algorithm = dnnl::algorithm::eltwise_tanh;
				break;
			}

			fwdDesc = std::make_unique<dnnl::eltwise_forward::primitive_desc>(dnnl::eltwise_forward::primitive_desc(dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, algorithm, *InputLayer->DstMemDesc, Alpha, Beta), Device.first));
			bwdDesc = std::make_unique<dnnl::eltwise_backward::primitive_desc>(dnnl::eltwise_backward::primitive_desc(dnnl::eltwise_backward::desc(algorithm, *DiffDstMemDesc, *DstMemDesc, Alpha, Beta), Device.first, *fwdDesc));

			fwd = std::make_unique<dnnl::eltwise_forward>(dnnl::eltwise_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::eltwise_backward>(dnnl::eltwise_backward(*bwdDesc));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;
		}
		}
	}

	void Activation::ForwardProp(const size_t batchSize, const bool training)
	{
		switch (ActivationFunction)
		{
		case Activations::LogSoftmax:
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDescLogSoftmax->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
				Device.second.wait();
			}

			auto dstMem = dnnl::memory(fwdDescLogSoftmax->dst_desc(), Device.first, Neurons.data());

			fwdLogSoftmax->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem} , {DNNL_ARG_DST,  dstMem} });
			Device.second.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
		}
		break;

		case Activations::Softmax:
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDescSoftmax->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
				Device.second.wait();
			}

			auto dstMem = dnnl::memory(fwdDescSoftmax->dst_desc(), Device.first, Neurons.data());

			fwdSoftmax->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem} , {DNNL_ARG_DST,  dstMem} });
			Device.second.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
		}
		break;

		case Activations::HardLogistic:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						Neurons[c] = HardLogistic::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
					}
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;

						for (auto c = offsetN; c < offsetN + C; c++)
						{
							Neurons[c] = HardLogistic::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto vecZero = VecFloat(0);
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;

					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;

						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;

							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;

						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;

							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
								}
							}
						}
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}
		break;

		case Activations::HardSwish:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						Neurons[c] = HardSwish::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
					}
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;

						for (auto c = offsetN; c < offsetN + C; c++)
						{
							Neurons[c] = HardSwish::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto vecZero = VecFloat(0);
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;

					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;

						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;

							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								HardSwish::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;

						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;

							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									HardSwish::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
								}
							}
						}
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}
		break;

		case Activations::Mish:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						Neurons[c] = Mish::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
					}
				}
				else
				{
#endif
				for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
				{
					const auto offsetN = n * CDHW;

					for (auto c = offsetN; c < offsetN + C; c++)
					{
						Neurons[c] = Mish::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
					}
				});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto vecZero = VecFloat(0);
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;

					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;

						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;

							for (size_t w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								Mish::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_a(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;

						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;

							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									Mish::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
									vecZero.store_a(&NeuronsD1[w]);
#endif // DNN_LEAN
								}
							}
						}
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}
		break;

		case Activations::LogLogistic:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
					{
						Neurons[c] = LogLogistic::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
						NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
					}
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;

						for (auto c = offsetN; c < offsetN + C; c++)
						{
							Neurons[c] = LogLogistic::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
							NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto vecZero = VecFloat(0);
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;

					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;

						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;

							for (size_t w = offsetH; w < offsetH + strideH; w += VectorSize)
							{
								LogLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
								vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
							}
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;

						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;

							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;

								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								{
									LogLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[w])).store_a(&Neurons[w]);
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[w]);
#endif // DNN_LEAN
								}
							}
						}
					});
				}
#ifdef DNN_STOCHASTIC
			}
#endif
		}
		break;

		default:
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.first) : memSrc;
			if (reorderFwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
				Device.second.wait();
			}

			auto dstMem = dnnl::memory(fwdDesc->dst_desc(), Device.first, Neurons.data());

			fwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem} , {DNNL_ARG_DST, dstMem} });
			Device.second.wait();

#ifndef DNN_LEAN
			if (training)
				ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
		}
		}
	}

	void Activation::BackwardProp(const size_t batchSize)
	{
#ifdef DNN_LEAN
		ZeroGradient(batchSize);
#else
		DNN_UNREF_PAR(batchSize);
#endif // DNN_LEAN


		switch (ActivationFunction)
		{
		case Activations::LogSoftmax:
		{
			auto dstMem = dnnl::memory(bwdDescLogSoftmax->dst_desc(), Device.first, Neurons.data());
			auto diffDstMem = dnnl::memory(bwdDescLogSoftmax->diff_dst_desc(), Device.first, NeuronsD1.data());

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescLogSoftmax->diff_src_desc(), Device.first) : memDiffSrc;

			bwdLogSoftmax->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem},  {DNNL_ARG_DIFF_DST, diffDstMem}, {DNNL_ARG_DIFF_SRC, diffSrcMem} });
			Device.second.wait();

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, {DNNL_ARG_TO, memDiffSrc} });
				Device.second.wait();
			}

			if (SharesInput)
			{
				bwdAdd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) } });
				Device.second.wait();
			}
		}
		break;

		case Activations::Softmax:
		{
			auto dstMem = dnnl::memory(bwdDescSoftmax->dst_desc(), Device.first, Neurons.data());
			auto diffDstMem = dnnl::memory(bwdDescSoftmax->diff_dst_desc(), Device.first, NeuronsD1.data());

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescSoftmax->diff_src_desc(), Device.first) : memDiffSrc;

			bwdSoftmax->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem},  {DNNL_ARG_DIFF_DST, diffDstMem}, {DNNL_ARG_DIFF_SRC, diffSrcMem} });
			Device.second.wait();

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, {DNNL_ARG_TO, memDiffSrc} });
				Device.second.wait();
			}

			if (SharesInput)
			{
				bwdAdd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) } });
				Device.second.wait();
			}
		}
		break;

		case Activations::HardLogistic:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] += HardLogistic::df(Neurons[c]) * NeuronsD1[c];
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;
						for (auto c = offsetN; c < offsetN + C; c++)
							InputLayer->NeuronsD1[c] += HardLogistic::df(Neurons[c]) * NeuronsD1[c];
					});
#ifdef DNN_STOCHASTIC
			}
#endif
		}
			else
			{
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;
						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;
							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								mul_add(HardLogistic::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;
							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									mul_add(HardLogistic::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
							}
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
		}
		break;

		case Activations::HardSwish:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] += HardSwish::df(Neurons[c]) * NeuronsD1[c];
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;
						for (auto c = offsetN; c < offsetN + C; c++)
							InputLayer->NeuronsD1[c] += HardSwish::df(Neurons[c]) * NeuronsD1[c];
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;
						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;
							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								mul_add(HardSwish::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;
							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									mul_add(HardSwish::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
							}
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
		}
		break;

		case Activations::Mish:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] += Mish::df(Neurons[c]) * NeuronsD1[c];
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;
						for (auto c = offsetN; c < offsetN + C; c++)
							InputLayer->NeuronsD1[c] += Mish::df(Neurons[c]) * NeuronsD1[c];
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;
						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;
							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								mul_add(Mish::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;
							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									mul_add(Mish::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
							}
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
		}
		break;

		case Activations::LogLogistic:
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					for (auto c = 0ull; c < C; c++)
						InputLayer->NeuronsD1[c] += LogLogistic::df(Neurons[c]) * NeuronsD1[c];
				}
				else
				{
#endif
					for_i(batchSize, LIGHT_COMPUTE, [=](size_t n)
					{
						const auto offsetN = n * CDHW;
						for (auto c = offsetN; c < offsetN + C; c++)
							InputLayer->NeuronsD1[c] += LogLogistic::df(Neurons[c]) * NeuronsD1[c];
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
			else
			{
				const auto strideH = W * VectorSize;
#ifdef DNN_STOCHASTIC
				if (batchSize == 1)
				{
					size_t offsetC, offsetH;
					for (auto c = 0ull; c < PaddedC; c += VectorSize)
					{
						offsetC = c * HW;
						for (auto h = 0ull; h < H; h++)
						{
							offsetH = offsetC + h * strideH;
							for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
								mul_add(LogLogistic::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
						}
					}
				}
				else
				{
#endif
					for_i(batchSize, MEDIUM_COMPUTE, [=](size_t n)
					{
						size_t offsetC, offsetH;
						for (auto c = 0ull; c < PaddedC; c += VectorSize)
						{
							offsetC = n * PaddedCDHW + c * HW;
							for (auto h = 0ull; h < H; h++)
							{
								offsetH = offsetC + h * strideH;
								for (auto w = offsetH; w < offsetH + strideH; w += VectorSize)
									mul_add(LogLogistic::dfVec(VecFloat().load_a(&Neurons[w])), VecFloat().load_a(&NeuronsD1[w]), VecFloat().load_a(&InputLayer->NeuronsD1[w])).store_a(&InputLayer->NeuronsD1[w]);
							}
						}
					});
#ifdef DNN_STOCHASTIC
				}
#endif
			}
		}
		break;

		default:
		{
			auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.first, InputLayer->Neurons.data());
			auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.first) : memSrc;
			if (reorderBwdSrc)
			{
				dnnl::reorder(memSrc, srcMem).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, {DNNL_ARG_TO, srcMem} });
				Device.second.wait();
			}

			auto diffDstMem = dnnl::memory(bwdDesc->diff_dst_desc(), Device.first, NeuronsD1.data());

			auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data());
			auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.first) : memDiffSrc;
			
			bwd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem},  {DNNL_ARG_DIFF_DST, diffDstMem}, {DNNL_ARG_DIFF_SRC, diffSrcMem} });
			Device.second.wait();

			if (reorderBwdDiffSrc)
			{
				dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.second, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, {DNNL_ARG_TO, memDiffSrc} });
				Device.second.wait();
			}

			if (SharesInput)
			{
				bwdAdd->execute(Device.second, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.first, InputLayer->NeuronsD1.data()) } });
				Device.second.wait();
			}
		}
		}
#ifdef DNN_LEAN
		ReleaseGradient();
#endif // DNN_LEAN
	}
}