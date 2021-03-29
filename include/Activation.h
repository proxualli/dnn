#pragma once
#include "Layer.h"

namespace dnn
{
	struct Abs
	{
		inline static Float f(const Float& x) noexcept { return std::abs(x); }
		inline static Float df(const Float& x) noexcept { return x > Float(0) ? Float(1) : x < Float(0) ? Float(-1) : Float(0); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return abs(x); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), VecFloat(1), select(x < VecFloat(0), VecFloat(-1), VecFloat(0))); }
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
		inline static Float df(const Float& x) noexcept { return x > Float(0) ? Float(1) : std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), x, exp(x) - VecFloat(1)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), VecFloat(1), exp(x)); }
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
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x < VecFloat(-3), VecFloat(0), select(x > VecFloat(3), VecFloat(1), (VecFloat(Float(1.0 / 3.0)) * x + VecFloat(Float(0.5))))); }
	};

	struct Identity
	{
		inline static Float f(const Float& x) noexcept { return x; }
		inline static Float df(const Float&) noexcept { return Float(1); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x; }
		inline static VecFloat dfVec(const VecFloat&) noexcept { return VecFloat(1); }
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
		inline static Float df(const Float& x) noexcept { const auto y = Logistic::f(x); return (  y * (Float(1) - y)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return (VecFloat(1) / (VecFloat(1) + exp(-x))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { const auto y = Logistic::fVec(x); return y * (VecFloat(1) - y); }
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

	struct LRelu // alpha >= 0
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
		inline static Float df(const Float& x) noexcept { const auto y = Tanh::f(x); return Float(1) - FloatSquare(y); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { const VecFloat tmpExp2 = exp(VecFloat(2) * x);  return (tmpExp2 - VecFloat(1)) / (tmpExp2 + VecFloat(1)); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { const auto y = Tanh::fVec(x); return (VecFloat(1) - square(y)); }
	};
	 
	struct FTS
	{
		inline static Float f(const Float& x, const Float& alpha = Float(-0.2)) noexcept { return x >= Float(0) ? x * Logistic::f(x) + alpha : alpha; }
		inline static Float df(const Float& x, const Float& alpha = Float(-0.2)) noexcept { return x >= Float(0) ? Logistic::f(x) * (Float(1) - x) + x : Float(0); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(Float(-0.2))) noexcept { return select(x >= VecFloat(0), x * Logistic::fVec(x) + alpha, alpha); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(Float(-0.2))) noexcept { return select(x >= VecFloat(0), Logistic::fVec(x) * (VecFloat(1) - x) + x, VecFloat(0)); }
	};

	enum class Activations
	{
		Abs = 0,
		BoundedRelu = 1,
		Clip = 2,
		Elu = 3,
		Exp = 4,
		FTS = 5,
		Gelu = 6,
		GeluErf = 7,
		HardLogistic = 8,
		HardSwish = 9,
		Linear = 10,
		Log = 11,
		Logistic = 12,
		LogLogistic = 13,
		LogSoftmax = 14,
		Mish = 15,
		Pow = 16,
		PRelu = 17,
		Relu = 18,
		Round = 19,
		Softmax = 20,
		SoftRelu = 21,
		Sqrt = 22,
		Square = 23,
		Swish = 24,
		Tanh = 25
	};

	class Activation final : public Layer
	{
	private:
		std::unique_ptr<dnnl::prelu_forward::primitive_desc> fwdDescPRelu;
		std::unique_ptr<dnnl::prelu_backward::primitive_desc> bwdDescPRelu;
		std::unique_ptr<dnnl::logsoftmax_forward::primitive_desc> fwdDescLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_backward::primitive_desc> bwdDescLogSoftmax;
		std::unique_ptr<dnnl::softmax_forward::primitive_desc> fwdDescSoftmax;
		std::unique_ptr<dnnl::softmax_backward::primitive_desc> bwdDescSoftmax;
		std::unique_ptr<dnnl::eltwise_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::eltwise_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::prelu_forward> fwdPRelu;
		std::unique_ptr<dnnl::prelu_backward> bwdPRelu;
		std::unique_ptr<dnnl::logsoftmax_forward> fwdLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_backward> bwdLogSoftmax;
		std::unique_ptr<dnnl::softmax_forward> fwdSoftmax;
		std::unique_ptr<dnnl::softmax_backward> bwdSoftmax;
		std::unique_ptr<dnnl::eltwise_forward> fwd;
		std::unique_ptr<dnnl::eltwise_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		dnnl::algorithm algorithm;

		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
		bool reorderBwdDiffWeights;

	public:
		const Activations ActivationFunction;
		const Float Alpha;
		const Float Beta;

		Activation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector<Layer*>& inputs, const Float alpha = Float(0), const Float beta = Float(0)) :
			Layer(device, format, name, LayerTypes::Activation, activation == dnn::Activations::PRelu ? inputs[0]->C : 0, activation == dnn::Activations::PRelu ? inputs[0]->C : 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, activation == dnn::Activations::PRelu),
			ActivationFunction(activation),
			Alpha(alpha),
			Beta(beta),
			algorithm(dnnl::algorithm::eltwise_linear),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false),
			reorderBwdDiffWeights(false)
		{
			assert(Inputs.size() == 1);

			PersistWeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 1, dnnl::memory::dim(C), 1, 1}), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw));
			WeightsMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ 1, dnnl::memory::dim(C), 1, 1 }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw));
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
				chosenFormat = dnnl::memory::format_tag::nc;
				
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, chosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
					chosenFormat = LayerBeforeCost || IsPlainDataFmt(*InputLayer->DstMemDesc) ? PlainFmt : GetDataFmt(*InputLayer->DstMemDesc);
				else
					chosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, chosenFormat));
			}

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(dnnl::binary::desc(dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc), Device.engine));
#ifdef DNN_CACHE_PRIMITIVES
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
			switch (ActivationFunction)
			{
			case Activations::PRelu:
			{
				auto memDesc = dnnl::memory::desc(dnnl::memory::dims({ 1, dnnl::memory::dim(C), 1, 1 }), dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
							
				fwdDescPRelu = std::make_unique<dnnl::prelu_forward::primitive_desc>(dnnl::prelu_forward::primitive_desc(dnnl::prelu_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, memDesc), Device.engine));
				bwdDescPRelu = std::make_unique<dnnl::prelu_backward::primitive_desc>(dnnl::prelu_backward::primitive_desc(dnnl::prelu_backward::desc(*DstMemDesc, memDesc, *DiffDstMemDesc, memDesc), Device.engine, *fwdDescPRelu));

				if (*WeightsMemDesc != fwdDescPRelu->weights_desc())
				{
					auto memWeights = dnnl::memory(*WeightsMemDesc, Device.engine, Biases.data());

					auto weights = FloatVector(fwdDescPRelu->weights_desc().get_size());
					auto weightsMem = dnnl::memory(fwdDescPRelu->weights_desc(), Device.engine, weights.data());

					dnnl::reorder(memWeights, weightsMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memWeights}, { DNNL_ARG_TO, weightsMem } });
					Device.stream.wait();

					Biases = weights;
					WeightsMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescPRelu->weights_desc());
				}

				DstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescPRelu->dst_desc());
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(fwdDescPRelu->dst_desc());

				if (Format == dnnl::memory::format_tag::any)
					chosenFormat = GetDataFmt(*DstMemDesc);

				reorderFwdSrc = fwdDescPRelu->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdSrc = bwdDescPRelu->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDescPRelu->diff_src_desc() != *InputLayer->DiffDstMemDesc;
				reorderBwdDiffWeights = bwdDescPRelu->diff_weights_desc() != fwdDescPRelu->weights_desc();
#ifdef DNN_CACHE_PRIMITIVES
				fwdPRelu = std::make_unique<dnnl::prelu_forward>(dnnl::prelu_forward(*fwdDescPRelu));
				bwdPRelu = std::make_unique<dnnl::prelu_backward>(dnnl::prelu_backward(*bwdDescPRelu));
#endif
			}
			break;

			case Activations::LogSoftmax:
			{
				const auto axis = (H == 1 && W == 1) ? 1 : 3;
				fwdDescLogSoftmax = std::make_unique<dnnl::logsoftmax_forward::primitive_desc>(dnnl::logsoftmax_forward::primitive_desc(dnnl::logsoftmax_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.engine));
				bwdDescLogSoftmax = std::make_unique<dnnl::logsoftmax_backward::primitive_desc>(dnnl::logsoftmax_backward::primitive_desc(dnnl::logsoftmax_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.engine, *fwdDescLogSoftmax));
				reorderFwdSrc = fwdDescLogSoftmax->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDescLogSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
#ifdef DNN_CACHE_PRIMITIVES
				fwdLogSoftmax = std::make_unique<dnnl::logsoftmax_forward>(dnnl::logsoftmax_forward(*fwdDescLogSoftmax));
				bwdLogSoftmax = std::make_unique<dnnl::logsoftmax_backward>(dnnl::logsoftmax_backward(*bwdDescLogSoftmax));
#endif
			}
			break;

			case Activations::Softmax:
			{
				const auto axis = (H == 1 && W == 1) ? 1 : 3;
				fwdDescSoftmax = std::make_unique<dnnl::softmax_forward::primitive_desc>(dnnl::softmax_forward::primitive_desc(dnnl::softmax_forward::desc(dnnl::prop_kind::forward, *DstMemDesc, axis), Device.engine));
				bwdDescSoftmax = std::make_unique<dnnl::softmax_backward::primitive_desc>(dnnl::softmax_backward::primitive_desc(dnnl::softmax_backward::desc(*DiffDstMemDesc, *DstMemDesc, axis), Device.engine, *fwdDescSoftmax));
				reorderFwdSrc = fwdDescSoftmax->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDescSoftmax->diff_src_desc() != *InputLayer->DiffDstMemDesc;
#ifdef DNN_CACHE_PRIMITIVES
				fwdSoftmax = std::make_unique<dnnl::softmax_forward>(dnnl::softmax_forward(*fwdDescSoftmax));
				bwdSoftmax = std::make_unique<dnnl::softmax_backward>(dnnl::softmax_backward(*bwdDescSoftmax));
#endif
			}
			break;

            default:
			{
				switch (ActivationFunction)
				{
				case Activations::FTS:
				case Activations::HardLogistic:
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
				case Activations::HardSwish:
					algorithm = dnnl::algorithm::eltwise_hardswish;
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
				case Activations::Mish:
					algorithm = dnnl::algorithm::eltwise_mish;
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

				reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
				reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
				fwd = std::make_unique<dnnl::eltwise_forward>(dnnl::eltwise_forward(*fwdDesc));
				bwd = std::make_unique<dnnl::eltwise_backward>(dnnl::eltwise_backward(*bwdDesc));
#endif
			}
			}
		}

		ByteVector GetImage(const Byte fillColor) final override
		{
			if (HasWeights)
			{
				const auto rangeWeights = GetColorRange(BiasesStats.Min, BiasesStats.Max);

				const auto width = BiasCount;
				const auto height = 1;
				const auto totalSize = width * (height + 3);

				auto image = ByteVector(totalSize, fillColor);

				for (auto y = 0ull; y < height; y++)
				{
					const auto start = y * width;
					const auto end = start + width;
					for (auto x = start; x < end; x++)
						image[x] = GetColorFromRange(rangeWeights, BiasesStats.Min, Biases[x]);
				}
				
				return image;
			}
			else
				return ByteVector();
		}

		void ResetWeights(const Fillers weightFiller, const Float weightFillerScale, const Fillers biasFiller, const Float biasFillerScale) override
		{
			if (HasWeights)
				Biases = FloatVector(PaddedC, Float(Alpha));
			
			DNN_UNREF_PAR(weightFiller);
			DNN_UNREF_PAR(weightFillerScale);
			DNN_UNREF_PAR(biasFiller);
			DNN_UNREF_PAR(biasFillerScale);
		}

		void ForwardProp(const size_t batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto strideHW = HW * VectorSize;
			const auto vecZero = VecFloat(0);

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
#ifdef DNN_CACHE_PRIMITIVES
				fwdLogSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
#else
				dnnl::logsoftmax_forward(*fwdDescLogSoftmax).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
#endif
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
#ifdef DNN_CACHE_PRIMITIVES
				fwdSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
#else
				dnnl::softmax_forward(*fwdDescSoftmax).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST,  dstMem } });
#endif
				Device.stream.wait();

#ifndef DNN_LEAN
				if (training)
					ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
			}
			break;

			case Activations::PRelu:
			{
				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderFwdSrc ? dnnl::memory(fwdDescPRelu->src_desc(), Device.engine) : memSrc;
				if (reorderFwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto weightsMem = dnnl::memory(fwdDescPRelu->weights_desc(), Device.engine, Biases.data());

				auto dstMem = dnnl::memory(*DstMemDesc, Device.engine, Neurons.data());
#ifdef DNN_CACHE_PRIMITIVES
				fwdPRelu->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#else
				dnnl::prelu_forward(*fwdDescPRelu).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DST, dstMem } });
#endif
				Device.stream.wait();

#ifndef DNN_LEAN
				if (training)
					ZeroFloatVector(NeuronsD1.data(), batchSize * PaddedCDHW);
#endif
			}
			break;

			case Activations::FTS:
			{
				if (InputLayer->DstMemDesc->data.ndims == 2)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (training)
						{
							if (!plain)
							{
								const auto vecZero = VecFloat(0);
								for (auto c = 0ull; c < PaddedC; c+=VectorSize)
								{
									FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);			
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
								}
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									Neurons[c] = FTS::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
									NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
								}
							}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c+=VectorSize)
									FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);			
							else
								for (auto c = 0ull; c < C; c++)
									Neurons[c] = FTS::f(InputLayer->Neurons[c]);
						}
					}
					else
					{
#endif
						if (training)
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto vecZero = VecFloat(0);
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c+=VectorSize)
									{
										FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
									{
										Neurons[c] = FTS::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
										NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
									}
								});
							}
						}
						else
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto vecZero = VecFloat(0);
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c+=VectorSize)
										FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
										Neurons[c] = FTS::f(InputLayer->Neurons[c]);
								});
							}
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
						if (training)
						{
							if (!plain)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
									}
								}
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < HW; hw++)
									{
										Neurons[hw + offset] = FTS::f(InputLayer->Neurons[hw + offset]);
#ifndef DNN_LEAN
										NeuronsD1[hw + offset] = Float(0);
#endif // DNN_LEAN
									}
								}								
							}
							
						}
						else
						{
							if (!plain)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
								}
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < HW; hw++)
										Neurons[hw + offset] = FTS::f(InputLayer->Neurons[hw + offset]);
								}			
							}
						}
					}
					else
					{
#endif
						if (training)
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto vecZero = VecFloat(0);
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW + c * HW;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
											vecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
										}
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW + c * HW;
										for (auto hw = 0ull; hw < HW; hw++)
										{
											Neurons[hw + offset] = FTS::f(InputLayer->Neurons[hw + offset]);
#ifndef DNN_LEAN
											NeuronsD1[hw + offset] = Float(0);
#endif // DNN_LEAN
										}
									}
								});
							}
						}
						else
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW + c * HW;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											FTS::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW + c * HW;
										for (auto hw = 0ull; hw < HW; hw++)
											Neurons[hw + offset] = FTS::f(InputLayer->Neurons[hw + offset]);
									}
								});
							}
						}
					}
#ifdef DNN_STOCHASTIC
				}
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
						if (training)
						{
							if (!plain)
							{
								const auto vecZero = VecFloat(0);
								for (auto c = 0ull; c < PaddedC; c+=VectorSize)
								{
									HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);			
#ifndef DNN_LEAN
									vecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
								}
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									Neurons[c] = HardLogistic::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
									NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
								}
							}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c+=VectorSize)
									HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);			
							else
								for (auto c = 0ull; c < C; c++)
									Neurons[c] = HardLogistic::f(InputLayer->Neurons[c]);
						}
					}
					else
					{
#endif
						if (training)
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto vecZero = VecFloat(0);
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c+=VectorSize)
									{
										HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
									{
										Neurons[c] = HardLogistic::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
										NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
									}
								});
							}
						}
						else
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto vecZero = VecFloat(0);
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c+=VectorSize)
										HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
										Neurons[c] = HardLogistic::f(InputLayer->Neurons[c]);
								});
							}
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
						if (training)
						{
							if (!plain)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
										vecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
									}
								}
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < HW; hw++)
									{
										Neurons[hw + offset] = HardLogistic::f(InputLayer->Neurons[hw + offset]);
#ifndef DNN_LEAN
										NeuronsD1[hw + offset] = Float(0);
#endif // DNN_LEAN
									}
								}								
							}
							
						}
						else
						{
							if (!plain)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
								}
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW;
									for (auto hw = 0ull; hw < HW; hw++)
										Neurons[hw + offset] = HardLogistic::f(InputLayer->Neurons[hw + offset]);
								}			
							}
						}
					}
					else
					{
#endif
						if (training)
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									const auto vecZero = VecFloat(0);
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW + c * HW;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
											vecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
										}
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW + c * HW;
										for (auto hw = 0ull; hw < HW; hw++)
										{
											Neurons[hw + offset] = HardLogistic::f(InputLayer->Neurons[hw + offset]);
#ifndef DNN_LEAN
											NeuronsD1[hw + offset] = Float(0);
#endif // DNN_LEAN
										}
									}
								});
							}
						}
						else
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW + c * HW;
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											HardLogistic::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](size_t n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW + c * HW;
										for (auto hw = 0ull; hw < HW; hw++)
											Neurons[hw + offset] = HardLogistic::f(InputLayer->Neurons[hw + offset]);
									}
								});
							}
						}
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
#ifdef DNN_CACHE_PRIMITIVES
				fwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
#else
				dnnl::eltwise_forward(*fwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
#endif
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
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto elements = plain ? batchSize * CDHW : batchSize * PaddedCDHW;
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto strideHW = HW * VectorSize;

			switch (ActivationFunction)
			{
			case Activations::LogSoftmax:
			{
				auto dstMem = dnnl::memory(bwdDescLogSoftmax->dst_desc(), Device.engine, Neurons.data());
				auto diffDstMem = dnnl::memory(bwdDescLogSoftmax->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescLogSoftmax->diff_src_desc(), Device.engine) : memDiffSrc;
#ifdef DNN_CACHE_PRIMITIVES
				bwdLogSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::logsoftmax_backward(*bwdDescLogSoftmax).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
#ifdef DNN_CACHE_PRIMITIVES
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });		
#else
					dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#endif
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
#ifdef DNN_CACHE_PRIMITIVES
				bwdSoftmax->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::softmax_backward(*bwdDescSoftmax).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_DST, dstMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
#ifdef DNN_CACHE_PRIMITIVES
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#else
					dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#endif
					Device.stream.wait();
				}
			}
			break;

			case Activations::PRelu:
			{
				auto dstMem = dnnl::memory(bwdDescPRelu->dst_desc(), Device.engine, Neurons.data());
				auto diffDstMem = dnnl::memory(bwdDescPRelu ->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDescPRelu->diff_src_desc(), Device.engine) : memDiffSrc;
				
				auto memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDescPRelu->src_desc(), Device.engine) : memSrc;
				if (reorderBwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}
				
				auto memDiffWeights = dnnl::memory(*WeightsMemDesc, Device.engine, BiasesD1.data());
				auto diffWeightsMem = reorderBwdDiffWeights ? dnnl::memory(bwdDescPRelu->diff_weights_desc(), Device.engine) : memDiffWeights;
	
				auto weightsMem = dnnl::memory(bwdDescPRelu->weights_desc(), Device.engine, Biases.data());
#ifdef DNN_CACHE_PRIMITIVES
				bwdPRelu->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::prelu_backward(*bwdDescPRelu).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_WEIGHTS, weightsMem }, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_WEIGHTS, diffWeightsMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
				Device.stream.wait();

				if (reorderBwdDiffWeights)
				{
					dnnl::reorder(diffWeightsMem, memDiffWeights).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffWeightsMem}, { DNNL_ARG_TO, memDiffWeights } });
					Device.stream.wait();
				}
				
				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
#ifdef DNN_CACHE_PRIMITIVES
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#else
					dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#endif
					Device.stream.wait();
				}
			}
			break;

			case Activations::FTS:
			{
				if (InputLayer->DstMemDesc->data.ndims == 2)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (!plain)
							for (auto c = 0ull; c < PaddedC; c+=VectorSize)
								mul_add(FTS::dfVec(VecFloat().load_a(&InputLayer->Neurons[c])), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
						else
							for (auto c = 0ull; c < C; c++)
								InputLayer->NeuronsD1[c] += FTS::df(InputLayer->Neurons[c]) * NeuronsD1[c];
					}
					else
					{
#endif
						if (!plain)
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto offset = n * PaddedC;
								for (auto c = offset; c < offset + PaddedC; c+=VectorSize)
									mul_add(FTS::dfVec(VecFloat().load_a(&InputLayer->Neurons[c])), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
							});
						else
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto offset = n * C;
								for (auto c = offset; c < offset + C; c++)
									InputLayer->NeuronsD1[c] += FTS::df(InputLayer->Neurons[c]) * NeuronsD1[c];
							});
#ifdef DNN_STOCHASTIC
					}
#endif
				}
				else
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (!plain)
						{
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = c * HW;
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
									mul_add(FTS::dfVec(VecFloat().load_a(&InputLayer->Neurons[hw])), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
						}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = c * HW;
								for (auto hw = offset; hw < offset + HW; hw++)
									InputLayer->NeuronsD1[hw] += FTS::df(InputLayer->Neurons[hw]) * NeuronsD1[hw];
							}
						}
					}
					else
					{
#endif
						if (!plain)
						{
							for_i(batchSize, threads, [=](size_t n)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = n * PaddedCDHW + c * HW;
									for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
										mul_add(FTS::dfVec(VecFloat().load_a(&InputLayer->Neurons[hw])), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
							});
						}
						else
						{
							for_i(batchSize, threads, [=](size_t n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = n * CDHW + c * HW;
									for (auto hw = offset; hw < offset + HW; hw++)
										InputLayer->NeuronsD1[hw] += FTS::df(InputLayer->Neurons[hw]) * NeuronsD1[hw];
								}
							});
						}
#ifdef DNN_STOCHASTIC
					}
#endif
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
						if (!plain)
							for (auto c = 0ull; c < PaddedC; c+=VectorSize)
								mul_add(HardLogistic::dfVec(VecFloat().load_a(&InputLayer->Neurons[c])), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
						else
							for (auto c = 0ull; c < C; c++)
								InputLayer->NeuronsD1[c] += HardLogistic::df(InputLayer->Neurons[c]) * NeuronsD1[c];
					}
					else
					{
#endif
						if (!plain)
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto offset = n * PaddedC;
								for (auto c = offset; c < offset + PaddedC; c+=VectorSize)
									mul_add(HardLogistic::dfVec(VecFloat().load_a(&InputLayer->Neurons[c])), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
							});
						else
							for_i(batchSize, threads, [=](size_t n)
							{
								const auto offset = n * C;
								for (auto c = offset; c < offset + C; c++)
									InputLayer->NeuronsD1[c] += HardLogistic::df(InputLayer->Neurons[c]) * NeuronsD1[c];
							});
#ifdef DNN_STOCHASTIC
					}
#endif
				}
				else
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (!plain)
						{
							for (auto c = 0ull; c < PaddedC; c += VectorSize)
							{
								const auto offset = c * HW;
								for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
									mul_add(HardLogistic::dfVec(VecFloat().load_a(&InputLayer->Neurons[hw])), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
							}
						}
						else
						{
							for (auto c = 0ull; c < C; c++)
							{
								const auto offset = c * HW;
								for (auto hw = offset; hw < offset + HW; hw++)
									InputLayer->NeuronsD1[hw] += HardLogistic::df(InputLayer->Neurons[hw]) * NeuronsD1[hw];
							}
						}
					}
					else
					{
#endif
						if (!plain)
						{
							for_i(batchSize, threads, [=](size_t n)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = n * PaddedCDHW + c * HW;
									for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
										mul_add(HardLogistic::dfVec(VecFloat().load_a(&InputLayer->Neurons[hw])), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
							});
						}
						else
						{
							for_i(batchSize, threads, [=](size_t n)
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = n * CDHW + c * HW;
									for (auto hw = offset; hw < offset + HW; hw++)
										InputLayer->NeuronsD1[hw] += HardLogistic::df(InputLayer->Neurons[hw]) * NeuronsD1[hw];
								}
							});
						}
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
#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::eltwise_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput)
				{
#ifdef DNN_CACHE_PRIMITIVES
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#else
					dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data()) } });
#endif
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