#pragma once
#include "Layer.h"

namespace dnn
{
	struct Abs
	{
		inline static Float f(const Float& x) noexcept { return std::abs(x); }
		inline static Float df(const Float& x) noexcept { return x / std::abs(x); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return abs(x); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return x / abs(x); }
	};

	struct ASinh
	{
		inline static Float f(const Float& x) noexcept { return std::asinh(x); }
		inline static Float df(const Float& x) noexcept { return Float(1) / std::cosh(x); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return asinh(x); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return VecFloat(1) / cosh(x); }
	};

	struct Elu
	{
		inline static Float f(const Float& x) noexcept { return x > Float(0) ? x : std::exp(x) - Float(1); }
		inline static Float df(const Float& x) noexcept { return x > Float(0) ? Float(1) : x + Float(1); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), x, exp(x) - VecFloat(1)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), VecFloat(1), x + VecFloat(1)); }
	};

	struct HardLogistic
	{
		inline static Float f(const Float& x) noexcept { return std::min(Float(1), std::max(Float(0), x * Float(0.2) + Float(0.5))); }
		inline static Float df(const Float& x) noexcept { return x < Float(-2.5) || x > Float(2.5) ? Float(0) : Float(0.2); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return min(VecFloat(1), max(VecFloat(0), x * VecFloat(Float(0.2)) + VecFloat(Float(0.5)))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x < VecFloat(Float(-2.5)) | x > VecFloat(Float(2.5)), VecFloat(0), VecFloat(Float(0.2))); }
	};

	struct Relu6
	{
		inline static Float f(const Float& x) noexcept { return std::min(std::max(x, Float(0)), Float(6)); }
		inline static Float df(const Float& x) noexcept { return x < Float(0) || x > Float(6) ? Float(0) : Float(1); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return min(max(x, VecFloat(0)), VecFloat(6)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x < VecFloat(0) | x > VecFloat(6), VecFloat(0), VecFloat(1)); }
	};

	struct HardSwish
	{
		inline static Float f(const Float& x) noexcept { return x * Relu6::f(x + Float(3)) * Float(1.0 / 6.0); }
		inline static Float df(const Float& x) noexcept { return x < Float(-3) ? Float(0) : x > Float(3) ? Float(1) : (Float(1.0 / 3.0) * x + Float(0.5)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x * Relu6::fVec(x + VecFloat(3)) * VecFloat(Float(1.0 / 6.0)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x < VecFloat(-3), VecFloat(0), select(x > VecFloat(3), VecFloat(1), (VecFloat(Float(1.0 / 3.0))* x + VecFloat(Float(0.5))))); }
	};

	struct Identity
	{
		inline static Float f(const Float& x) noexcept { return x; }
		inline static Float df(const Float& x) noexcept { return Float(1); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x; }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return VecFloat(1); }
	};

	struct Log
	{
		inline static Float f(const Float& x) noexcept { return std::log(x); }
		inline static Float df(const Float& x) noexcept { return Float(1) / x; }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return log(x); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return VecFloat(1) / x; }
	};

	struct Logistic
	{
		inline static Float f(const Float& x) noexcept { return (Float(1) / (Float(1) + std::exp(-x))); }
		inline static Float df(const Float& x) noexcept { return (x * (Float(1) - x)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return (VecFloat(1) / (VecFloat(1) + exp(-x))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return x * (VecFloat(1) - x); }
	};

	struct SoftRelu
	{
		inline static Float f(const Float& x) noexcept { return std::log(Float(1) + std::exp(x)); }
		inline static Float df(const Float& x) noexcept { return Float(1) / (Float(1) + std::exp(-x)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return log(VecFloat(1) + exp(x)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return VecFloat(1) / (VecFloat(1) + exp(-x)); }
	};

	struct LogLogistic
	{
		inline static Float f(const Float& x) noexcept { return -SoftRelu::f(-x); }
		inline static Float df(const Float& x) noexcept { return Float(1) / (std::exp(x) + Float(1)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return -SoftRelu::fVec(-x); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return VecFloat(1) / (exp(x) + VecFloat(1)); }
	};

	struct LRelu
	{
		inline static Float f(const Float& x, const Float& alpha) noexcept { return x > Float(0) ? x : x * alpha; }
		inline static Float df(const Float& x, const Float& alpha) noexcept { return x > Float(0) ? Float(1) : alpha; }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha) noexcept { return select(x > VecFloat(0), x, x * alpha); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha) noexcept { return select(x > VecFloat(0), VecFloat(1), alpha); }
	};

	struct Mish
	{
		inline static Float f(const Float& x) noexcept { return x * std::tanh(std::log1p(std::exp(x))); }
		inline static Float df(const Float& x) noexcept { const Float tmpExp = std::exp(x); const Float tmpSoftplus = std::log1p(tmpExp); const Float tmpSech = Float(1) / std::cosh(tmpSoftplus); return std::tanh(tmpSoftplus) + x * tmpExp * FloatSquare(tmpSech) / (tmpExp + Float(1)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x * tanh(log1p(exp(x))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { const VecFloat tmpExp = exp(x); const VecFloat tmpSoftplus = log1p(tmpExp); const VecFloat tmpSech = VecFloat(1) / cosh(tmpSoftplus); return tanh(tmpSoftplus) + x * tmpExp * square(tmpSech) / (tmpExp + VecFloat(1)); }
	};

	struct Relu
	{
		inline static Float f(const Float& x) noexcept { return std::max(x, Float(0)); }
		inline static Float df(const Float& x) noexcept { return x > Float(0) ? Float(1) : Float(0); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return max(x, VecFloat(0)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), VecFloat(1), VecFloat(0)); }
	};

	struct Selu
	{
		inline static Float f(const Float& x) noexcept { return Float(1.0507009873554804934193349852946) * (x > Float(0) ? x : Float(1.6732632423543772848170429916717) * (std::exp(x) - Float(1))); }
		inline static Float df(const Float& x) noexcept { return x > Float(0) ? Float(1.0507009873554804934193349852946) : Float(1.7580993408473768599402175208123) * std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return VecFloat(Float(1.0507009873554804934193349852946)) * select(x > VecFloat(0), x, VecFloat(Float(1.6732632423543772848170429916717)) * (exp(x) - VecFloat(1))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), VecFloat(Float(1.0507009873554804934193349852946)), VecFloat(Float(1.7580993408473768599402175208123)) * exp(x)); }
	};

	struct SoftPlus
	{
		inline static Float f(const Float& x, const Float& beta = Float(1), const Float& treshold = Float(20)) noexcept { const Float y = beta * x; return y > treshold ? x : std::log1p(std::exp(y)) / beta; }
		inline static Float df(const Float& x, const Float& beta = Float(1), const Float& treshold = Float(20)) noexcept { const Float y = beta * x;  const Float tmpExp = std::exp(y); return y > treshold ? x : x * (tmpExp - Float(1)) / tmpExp; }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& beta = VecFloat(1), const VecFloat& treshold = VecFloat(20)) noexcept { const VecFloat y = beta * x; return select(y > treshold, x, log1p(exp(y)) / beta); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& beta = VecFloat(1), const VecFloat& treshold = VecFloat(20)) noexcept { const VecFloat y = beta * x; const VecFloat tmpExp = exp(y); return select(y > treshold, x, x * (tmpExp - VecFloat(1)) / tmpExp); }
	};
	
	struct SoftSign
	{
		inline static Float f(const Float& x) noexcept { return x / (Float(1) + std::abs(x)); }
		inline static Float df(const Float& x) noexcept { return Float(1) / FloatSquare(Float(1) + std::abs(x)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x / (VecFloat(1) + abs(x)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return VecFloat(1) / square(VecFloat(1) + abs(x)); }
	};

	struct Swish
	{
		inline static Float f(const Float& x) noexcept { return x / (std::exp(-x) + Float(1)); }
		inline static Float df(const Float& x) noexcept { return x + ((Float(1) - x) / (std::exp(-x) + Float(1))); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x / (exp(-x) + VecFloat(1)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return x + ((VecFloat(1) - x) / (exp(-x) + VecFloat(1))); }
	};

	struct Tanh
	{
		inline static Float f(const Float& x) noexcept { const Float tmpExp2 = std::exp(Float(2) * x);  return (tmpExp2 - Float(1)) / (tmpExp2 + Float(1)); }
		inline static Float df(const Float& x) noexcept { return Float(1) - FloatSquare(x); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { const VecFloat tmpExp2 = exp(VecFloat(2) * x);  return (tmpExp2 - VecFloat(1)) / (tmpExp2 + VecFloat(1)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return (VecFloat(1) - square(x)); }
	};
	

	enum class Activations
	{
		Abs = 0,
		BoundedRelu = 1,
		Clip = 2,
		Elu = 3,
		Exp = 4,
		Gelu = 5,
		GeluErf = 6,
		HardLogistic = 7,
		HardSwish = 8,
		Linear = 9,
		Log = 10,
		Logistic = 11,
		LogLogistic = 12,
		LogSoftmax = 13,
		Mish = 14,
		Pow = 15,
		PRelu = 16,
		Relu = 17,
		Round = 18,
		Softmax = 19,
		SoftRelu = 20,
		Sqrt = 21,
		Square = 22,
		Swish = 23,
		Tanh = 24
	};

	class Activation final : public Layer
	{
	private:
		std::unique_ptr<dnnl::prelu_forward::primitive_desc> fwdDescPRelu;
		std::unique_ptr<dnnl::prelu_backward::primitive_desc> bwdDescPRelu;
		std::unique_ptr<dnnl::prelu_forward> fwdPRelu;
		std::unique_ptr<dnnl::prelu_backward> bwdPRelu;

		std::unique_ptr<dnnl::logsoftmax_forward::primitive_desc> fwdDescLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_backward::primitive_desc> bwdDescLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_forward> fwdLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_backward> bwdLogSoftmax;

		std::unique_ptr<dnnl::softmax_forward::primitive_desc> fwdDescSoftmax;
		std::unique_ptr<dnnl::softmax_backward::primitive_desc> bwdDescSoftmax;
		std::unique_ptr<dnnl::softmax_forward> fwdSoftmax;
		std::unique_ptr<dnnl::softmax_backward> bwdSoftmax;

		std::unique_ptr<dnnl::eltwise_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::eltwise_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::eltwise_forward> fwd;
		std::unique_ptr<dnnl::eltwise_backward> bwd;

		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
		std::unique_ptr<dnnl::binary> bwdAdd;

		dnnl::algorithm algorithm;

		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const Activations ActivationFunction;
		const Float Alpha;
		const Float Beta;

		Activation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector< std::shared_ptr<Layer>>& inputs, const Float alpha = Float(0), const Float beta = Float(0)) :
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
				
		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Activation:") + tab + std::string(magic_enum::enum_name<Activations>(ActivationFunction)));
			description.append(nwl + std::string(" Alpha:") + dtab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:") + dtab + FloatToString(Beta));

			return description;
		}

		size_t FanIn() const final override
		{
			return 1;
		}

		size_t FanOut() const final override
		{
			return 1;
		}

		void InitializeDescriptors(const size_t batchSize) final override
		{
			if (InputLayer->DstMemDesc->data.ndims == 2)
			{
				if (Format == dnnl::memory::format_tag::any)
					chosenFormat = dnnl::memory::format_tag::nc;
				
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
					chosenFormat = LayerBeforeCost || IsPlainDataFmt(*InputLayer->DstMemDesc) ? PlainFmt : GetDataFmt(*InputLayer->DstMemDesc);

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.engine));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));

			switch (ActivationFunction)
			{
			case Activations::PRelu:
			{
				/*
				fwdDescPRelu = std::make_unique<dnnl::prelu_forward::primitive_desc>(dnnl::prelu_forward::primitive_desc(dnnl::prelu_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.engine));
				bwdDescPRelu = std::make_unique<dnnl::prelu_backward::primitive_desc>(dnnl::prelu_backward::primitive_desc(dnnl::prelu_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.engine, *fwdDescSoftmax));

				fwdPRelu = std::make_unique<dnnl::prelu_forward>(dnnl::prelu_forward(*fwdDescPRelu));
				bwdPRelu = std::make_unique<dnnl::prelu_backward>(dnnl::prelu_backward(*bwdDescPRelu));

				reorderFwdSrc = fwdDescPRelu->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDescPRelu->diff_src_desc() != *InputLayer->DiffDstMemDesc;
				*/
			}
			break;

			case Activations::LogSoftmax:
			{
				const auto axis = (H == 1 && W == 1) ? 1 : 3;
				fwdDescLogSoftmax = std::make_unique<dnnl::logsoftmax_forward::primitive_desc>(dnnl::logsoftmax_forward::primitive_desc(dnnl::logsoftmax_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.engine));
				bwdDescLogSoftmax = std::make_unique<dnnl::logsoftmax_backward::primitive_desc>(dnnl::logsoftmax_backward::primitive_desc(dnnl::logsoftmax_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.engine, *fwdDescLogSoftmax));

				fwdLogSoftmax = std::make_unique<dnnl::logsoftmax_forward>(dnnl::logsoftmax_forward(*fwdDescLogSoftmax));
				bwdLogSoftmax = std::make_unique<dnnl::logsoftmax_backward>(dnnl::logsoftmax_backward(*bwdDescLogSoftmax));

				reorderFwdSrc = fwdDescLogSoftmax->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDescLogSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
			}
			break;

			case Activations::Softmax:
			{
				const auto axis = (H == 1 && W == 1) ? 1 : 3;
				fwdDescSoftmax = std::make_unique<dnnl::softmax_forward::primitive_desc>(dnnl::softmax_forward::primitive_desc(dnnl::softmax_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.engine));
				bwdDescSoftmax = std::make_unique<dnnl::softmax_backward::primitive_desc>(dnnl::softmax_backward::primitive_desc(dnnl::softmax_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.engine, *fwdDescSoftmax));

				fwdSoftmax = std::make_unique<dnnl::softmax_forward>(dnnl::softmax_forward(*fwdDescSoftmax));
				bwdSoftmax = std::make_unique<dnnl::softmax_backward>(dnnl::softmax_backward(*bwdDescSoftmax));

				reorderFwdSrc = fwdDescSoftmax->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDescSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
			}
			break;

			case Activations::HardLogistic:
			case Activations::HardSwish:
			case Activations::Mish:
				if (!IsPlainDataFmt(*InputLayer->DstMemDesc) && !IsBlockedDataFmt(*InputLayer->DstMemDesc))
					throw std::invalid_argument("Input memory format not supported for this activation function");
				break;

			default:
			{
				switch (ActivationFunction)
				{
				case Activations::HardLogistic:
			    case Activations::HardSwish:
			    case Activations::Mish:
			    case Activations::LogSoftmax:
				case Activations::PRelu:
			    case Activations::Softmax:
				    break;

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

				case Activations::LogLogistic:
					algorithm = dnnl::algorithm::eltwise_logsigmoid;
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

				fwdDesc = std::make_unique<dnnl::eltwise_forward::primitive_desc>(dnnl::eltwise_forward::primitive_desc(dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, algorithm, *InputLayer->DstMemDesc, Alpha, Beta), Device.engine));
				bwdDesc = std::make_unique<dnnl::eltwise_backward::primitive_desc>(dnnl::eltwise_backward::primitive_desc(dnnl::eltwise_backward::desc(algorithm, *DiffDstMemDesc, *DstMemDesc, Alpha, Beta), Device.engine, *fwdDesc));

				fwd = std::make_unique<dnnl::eltwise_forward>(dnnl::eltwise_forward(*fwdDesc));
				bwd = std::make_unique<dnnl::eltwise_backward>(dnnl::eltwise_backward(*bwdDesc));

				reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;
			}
			}
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			switch (ActivationFunction)
			{
			case Activations::LogSoftmax:
			{
				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDescLogSoftmax->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto dstMem = dnnl::memory(fwdDescLogSoftmax->dst_desc(), Device.engine, Neurons.data());

				fwdLogSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
				Device.stream.wait();

#ifndef DNN_LEAN
				if (training)
					ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
			}
			break;

			case Activations::Softmax:
			{
				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDescSoftmax->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto dstMem = dnnl::memory(fwdDescSoftmax->dst_desc(), Device.engine, Neurons.data());

				fwdSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
				Device.stream.wait();

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

			default:
			{
				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto dstMem = dnnl::memory(fwdDesc->dst_desc(), Device.engine, Neurons.data());

				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
				Device.stream.wait();

#ifndef DNN_LEAN
				if (training)
					ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
			}
			}
		}

		void BackwardProp(const size_t batchSize) final override
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
				auto dstMem = dnnl::memory(bwdDescLogSoftmax->dst_desc(), Device.engine, Neurons.data());
				auto diffDstMem = dnnl::memory(bwdDescLogSoftmax->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescLogSoftmax->diff_src_desc(), Device.engine) : memDiffSrc;

				bwdLogSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
					Device.stream.wait();
				}
			}
			break;

			case Activations::Softmax:
			{
				auto dstMem = dnnl::memory(bwdDescSoftmax->dst_desc(), Device.engine, Neurons.data());
				auto diffDstMem = dnnl::memory(bwdDescSoftmax->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescSoftmax->diff_src_desc(), Device.engine) : memDiffSrc;

				bwdSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
					Device.stream.wait();
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

			default:
			{
				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderBwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto diffDstMem = dnnl::memory(bwdDesc->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
					Device.stream.wait();
				}
			}
			}
#ifdef DNN_LEAN
			ReleaseGradient();
#endif // DNN_LEAN
		}
	};
}