#pragma once
#include "Layer.h"

namespace dnn
{
	struct Abs
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return std::abs(x); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1) : x < Float(0) ? Float(-1) : Float(0); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return abs(x); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x > VecFloat(0), VecFloat(1), select(x < VecFloat(0), VecFloat(-1), VecFloat(0))); }
	};

	struct ASinh
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return std::asinh(x); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1) / std::cosh(x); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return asinh(x); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(1) / cosh(x); }
	};

	struct Elu 
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? x : alpha * (std::exp(x) - Float(1)); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1) : alpha * std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x > VecFloat(0), x, alpha * (exp(x) - VecFloat(1))); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x > VecFloat(0), VecFloat(1), alpha * exp(x)); }
	};

	struct HardLogistic
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0.2), const Float& beta = Float(0.5)) NOEXCEPT { return std::min(Float(1), std::max(Float(0), x * alpha + beta)); }
		inline static Float df(const Float& x, const Float& alpha = Float(0.2), const Float& beta = Float(0.5)) NOEXCEPT { return std::abs(x) > (beta / alpha) ? Float(0) : alpha; }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(Float(0.2)), const VecFloat& beta = VecFloat(0.5)) NOEXCEPT { return min(VecFloat(1), max(VecFloat(0), x * alpha + beta)); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(Float(0.2)), const VecFloat& beta = VecFloat(0.5)) NOEXCEPT { return select(abs(x) > (beta / alpha), VecFloat(0), alpha); }
	};

	struct Relu6
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return std::min(std::max(x, Float(0)), Float(6)); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x < Float(0) || x > Float(6) ? Float(0) : Float(1); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return min(max(x, VecFloat(0)), VecFloat(6)); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x < VecFloat(0) | x > VecFloat(6), VecFloat(0), VecFloat(1)); }
	};

	struct HardSwish
	{
		inline static Float f(const Float& x, const Float& alpha = Float(3), const Float& beta = Float(1) / Float(6)) NOEXCEPT{return x * Relu6::f(x + alpha) * beta;}
		inline static Float df(const Float& x, const Float& alpha = Float(3), const Float& beta = Float(1) / Float(6)) NOEXCEPT { return x < -alpha ? Float(0) : x > alpha ? Float(1) : ((Float(2) * beta * x) + (alpha * beta)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(Float(3)), const VecFloat& beta = VecFloat(Float(1) / Float(6))) NOEXCEPT { return x * Relu6::fVec(x + alpha) * beta; }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(Float(3)), const VecFloat& beta = VecFloat(Float(1) / Float(6))) NOEXCEPT { return select(x < -alpha, VecFloat(Float(0)), select(x > alpha, VecFloat(Float(1)), ((VecFloat(Float(2)) * beta * x) + (alpha * beta)))); }
	};

	struct Identity
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x; }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return x; }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(1); }
	};

	struct Log
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return std::log(x); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1) / x; }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return log(x); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(1) / x; }
	};

	struct Logistic
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return (Float(1) / (Float(1) + std::exp(-x))); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { const auto y = Logistic::f(x); return ( y * (Float(1) - y)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return (VecFloat(1) / (VecFloat(1) + exp(-x))); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { const auto y = Logistic::fVec(x); return y * (VecFloat(1) - y); }
	};

	struct SoftRelu
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return std::log(Float(1) + std::exp(x)); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1) / (Float(1) + std::exp(-x)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return log(VecFloat(1) + exp(x)); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(1) / (VecFloat(1) + exp(-x)); }
	};

	struct LogLogistic
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return -SoftRelu::f(-x); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1) / (std::exp(x) + Float(1)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return -SoftRelu::fVec(-x); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(1) / (exp(x) + VecFloat(1)); }
	};

	struct Relu // alpha >= 0
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? x : x * alpha; }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1) : alpha; }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x > VecFloat(0), x, x * alpha); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x > VecFloat(0), VecFloat(1), alpha); }
	};

	struct Selu
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1.0507009873554804934193349852946) * (x > Float(0) ? x : Float(1.6732632423543772848170429916717) * (std::exp(x) - Float(1))); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1.0507009873554804934193349852946) : Float(1.7580993408473768599402175208123) * std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(Float(1.0507009873554804934193349852946)) * select(x > VecFloat(0), x, VecFloat(Float(1.6732632423543772848170429916717)) * (exp(x) - VecFloat(1))); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return select(x > VecFloat(0), VecFloat(Float(1.0507009873554804934193349852946)), VecFloat(Float(1.7580993408473768599402175208123)) * exp(x)); }
	};

	struct SoftPlus
	{
		inline static Float f(const Float& x, const Float& alpha = Float(20), const Float& beta = Float(1)) NOEXCEPT { const auto y = beta * x; return y > alpha ? x : std::log1p(std::exp(y)) / beta; }
		inline static Float df(const Float& x, const Float& alpha = Float(20), const Float& beta = Float(1)) NOEXCEPT { const auto y = beta * x;  const auto tmpExp = std::exp(y); return y > alpha ? x : x * (tmpExp - Float(1)) / tmpExp; }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(20), const VecFloat& beta = VecFloat(1)) NOEXCEPT { const auto y = beta * x; return select(y > alpha, x, log1p(exp(y)) / beta); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(20), const VecFloat& beta = VecFloat(1)) NOEXCEPT { const auto y = beta * x; const auto tmpExp = exp(y); return select(y > alpha, x, x * (tmpExp - VecFloat(1)) / tmpExp); }
	};
	
	struct SoftSign
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x / (Float(1) + std::abs(x)); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1) / Square<Float>(Float(1) + std::abs(x)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return x / (VecFloat(1) + abs(x)); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return VecFloat(1) / square(VecFloat(1) + abs(x)); }
	};

	struct Swish
	{
		inline static Float f(const Float& x, const Float& alpha = Float(1), const Float& beta = Float(0)) NOEXCEPT { return x / (std::exp(-alpha * x) + Float(1)); }
		inline static Float df(const Float& x, const Float& alpha = Float(1), const Float& beta = Float(0)) NOEXCEPT { return (Float(1) / (std::exp(-alpha * x) + Float(1))) * (Float(1) + alpha * x * (Float(1) - (Float(1) / (std::exp(-alpha * x) + Float(1))))); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(1), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return x / (exp(-alpha * x) + VecFloat(1)); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(1), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return (VecFloat(1) / (exp(-alpha * x) + VecFloat(1))) * (VecFloat(1) + alpha * x * (VecFloat(1) - (VecFloat(1) / (exp(-alpha * x) + VecFloat(1))))); }
	};

	struct Tanh
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return std::tanh(x); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return Float(1) - Square<Float>(std::tanh(x)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return tanh(x); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return (VecFloat(1) - square(tanh(x))); }
	};
	 
	struct TanhExp
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x * std::tanh(std::exp(x)); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { const auto y = std::exp(x);  const auto z = std::tanh(y); return z - (x * y * (Square<Float>(z) - Float(1))); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return x * tanh(exp(x)); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { const auto y = exp(x); const auto z = tanh(y); return z - (x * y * (square(z) - VecFloat(1))); }
	};

	struct Mish
	{
		inline static Float f(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { return x * std::tanh(std::log1p(std::exp(x))); }
		inline static Float df(const Float& x, const Float& alpha = Float(0), const Float& beta = Float(0)) NOEXCEPT { const auto tmpExp = std::exp(x); const auto tmpSoftplus = std::log1p(tmpExp); const auto tmpSech = Float(1) / std::cosh(tmpSoftplus); return std::tanh(tmpSoftplus) + x * tmpExp * Square<Float>(tmpSech) / (tmpExp + Float(1)); }
		inline static VecFloat fVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { return x * tanh(log1p(exp(x))); }
		inline static VecFloat dfVec(const VecFloat& x, const VecFloat& alpha = VecFloat(0), const VecFloat& beta = VecFloat(0)) NOEXCEPT { const auto tmpExp = exp(x); const auto tmpSoftplus = log1p(tmpExp); const auto tmpSech = VecFloat(1) / cosh(tmpSoftplus); return tanh(tmpSoftplus) + x * tmpExp * square(tmpSech) / (tmpExp + VecFloat(1)); }
	};
	
	enum class Activations
	{
		Abs = 0,
		BoundedRelu = 1,
		Clip = 2,
		ClipV2 = 3,			//
		Elu = 4,			//
		Exp = 5,			//
		Gelu = 6,
		GeluErf = 7,
		HardLogistic = 8,
		HardSwish = 9,
		Linear = 10,
		Log = 11,
		Logistic = 12,		//
		LogLogistic = 13,
		Mish = 14,
		Pow = 15,
		Relu = 16,			//
		Round = 17,
		SoftRelu = 18,
		Sqrt = 19,			//
		Square = 20,
		Swish = 21,
		Tanh = 22,			//
		TanhExp = 23
	};

	class Activation final : public Layer
	{
	private:
		std::unique_ptr<dnnl::eltwise_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::eltwise_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;
#ifdef DNN_CACHE_PRIMITIVES
		std::unique_ptr<dnnl::eltwise_forward> fwd;
		std::unique_ptr<dnnl::eltwise_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;
#endif
		dnnl::algorithm algorithm;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;

	public:
		const Activations ActivationFunction;
		const Float Alpha;
		const Float Beta;

		Activation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector<Layer*>& inputs, const Float alpha = Float(0), const Float beta = Float(0)) :
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
			
		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Activation:") + tab + std::string(magic_enum::enum_name<Activations>(ActivationFunction)));
			description.append(nwl + std::string(" Alpha:") + dtab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:") + dtab + FloatToString(Beta));

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
			if (InputLayer->DstMemDesc->get_ndims() == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;
				
				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (Format == dnnl::memory::format_tag::any)
					ChosenFormat = LayerBeforeCost || IsPlainDataFmt(*InputLayer->DstMemDesc) ? PlainFmt : GetDataFmt(*InputLayer->DstMemDesc);
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));
#ifdef DNN_CACHE_PRIMITIVES
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
			switch (ActivationFunction)
			{
				case Activations::TanhExp:
				    break;

				case Activations::Abs:
					algorithm = dnnl::algorithm::eltwise_abs;
					break;
				case Activations::Clip:
					algorithm = dnnl::algorithm::eltwise_clip;
					break;
				case Activations::ClipV2:
					algorithm = dnnl::algorithm::eltwise_clip_v2;
					break;
				case Activations::BoundedRelu:
					algorithm = dnnl::algorithm::eltwise_clip; // alpha = 0 , Beta = former alpha
					break;
				case Activations::Elu:
					algorithm = dnnl::algorithm::eltwise_elu;
					break;
				case Activations::Exp:
					algorithm = dnnl::algorithm::eltwise_exp;
					break;
				case Activations::Gelu:
					algorithm = dnnl::algorithm::eltwise_gelu_tanh;
					break;
				case Activations::GeluErf:
					algorithm = dnnl::algorithm::eltwise_gelu_erf;
					break;
				case Activations::HardLogistic:
					algorithm = dnnl::algorithm::eltwise_hardsigmoid;
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
					algorithm = dnnl::algorithm::eltwise_soft_relu;  // alpha = -1
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

			fwdDesc = std::make_unique<dnnl::eltwise_forward::primitive_desc>(dnnl::eltwise_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, algorithm, *InputLayer->DstMemDesc, *DstMemDesc, Alpha, Beta));
			bwdDesc = std::make_unique<dnnl::eltwise_backward::primitive_desc>(dnnl::eltwise_backward::primitive_desc(Device.engine, algorithm, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, *DstMemDesc, Alpha, Beta, *fwdDesc));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::eltwise_forward>(dnnl::eltwise_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::eltwise_backward>(dnnl::eltwise_backward(*bwdDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			const auto plain = IsPlainFormat();
			const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * (plain ? CDHW() : PaddedCDHW()), Float(0.1));

			const auto strideHW = HW() * VectorSize;
			const auto vecZero = VecFloat(0);

			switch (ActivationFunction)
			{
			case Activations::TanhExp:
			{
				if (InputLayer->DstMemDesc->get_ndims () == 2)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (training)
						{
							if (!plain)
							{
								const auto vecZero = VecFloat(0);
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
#ifndef DNN_LEAN
									if (!InplaceBwd)
										vecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
								}
							}
							else
								for (auto c = 0ull; c < C; c++)
								{
									Neurons[c] = TanhExp::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
									if (!InplaceBwd)
										NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
								}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
									TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
							else
								for (auto c = 0ull; c < C; c++)
									Neurons[c] = TanhExp::f(InputLayer->Neurons[c]);
						}
					}
					else
					{
#endif
						if (training)
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto vecZero = VecFloat(0);
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c += VectorSize)
									{
										TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											vecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
									{
										Neurons[c] = TanhExp::f(InputLayer->Neurons[c]);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
									}
								});
						}
						else
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto vecZero = VecFloat(0);
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c += VectorSize)
										TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[c])).store_a(&Neurons[c]);
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
										Neurons[c] = TanhExp::f(InputLayer->Neurons[c]);
								});
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
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW();
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
									{
										TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											vecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
									}
								}
							else
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW();
									for (auto hw = 0ull; hw < HW(); hw++)
									{
										Neurons[hw + offset] = TanhExp::f(InputLayer->Neurons[hw + offset]);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											NeuronsD1[hw + offset] = Float(0);
#endif // DNN_LEAN
									}
								}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW();
									for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
								}
							else
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW();
									for (auto hw = 0ull; hw < HW(); hw++)
										Neurons[hw + offset] = TanhExp::f(InputLayer->Neurons[hw + offset]);
								}
						}
					}
					else
					{
#endif
						if (training)
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto vecZero = VecFloat(0);
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW() + c * HW();
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
										{
											TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
#ifndef DNN_LEAN
											if (!InplaceBwd)
												vecZero.store_nt(&NeuronsD1[hw + offset]);
#endif // DNN_LEAN
										}
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW() + c * HW();
										for (auto hw = 0ull; hw < HW(); hw++)
										{
											Neurons[hw + offset] = TanhExp::f(InputLayer->Neurons[hw + offset]);
#ifndef DNN_LEAN
											if (!InplaceBwd)
												NeuronsD1[hw + offset] = Float(0);
#endif // DNN_LEAN
										}
									}
								});
						}
						else
						{
							if (!plain)
							{
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW() + c * HW();
										for (auto hw = 0ull; hw < strideHW; hw += VectorSize)
											TanhExp::fVec(VecFloat().load_a(&InputLayer->Neurons[hw + offset])).store_a(&Neurons[hw + offset]);
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW() + c * HW();
										for (auto hw = 0ull; hw < HW(); hw++)
											Neurons[hw + offset] = TanhExp::f(InputLayer->Neurons[hw + offset]);
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
				if (training && !InplaceBwd)
					InitArray<Float>(NeuronsD1.data(), batchSize * PaddedCDHW());
#endif
			}
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			const auto plain = IsPlainFormat();
			const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * (plain ? CDHW() : PaddedCDHW()), Float(0.1));
			const auto strideHW = HW() * VectorSize;

			switch (ActivationFunction)
			{
			case Activations::TanhExp:
			{
				if (InputLayer->DstMemDesc->get_ndims() == 2)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (InplaceBwd)
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
									(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[c])), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
							else
							{
								for (auto c = 0ull; c < C; c++)
									InputLayer->NeuronsD1[c] = TanhExp::df(InputLayerFwd->Neurons[c]) * InputLayer->NeuronsD1[c];
							}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
									mul_add(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[c])), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
							else
							{
								for (auto c = 0ull; c < C; c++)
									InputLayer->NeuronsD1[c] += TanhExp::df(InputLayerFwd->Neurons[c]) * NeuronsD1[c];
							}
					}
					else
					{
#endif
						if (InplaceBwd)
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c += VectorSize)
										(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[c])), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
										InputLayer->NeuronsD1[c] = TanhExp::df(InputLayerFwd->Neurons[c]) * InputLayer->NeuronsD1[c];
								});
						}
						else
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto offset = n * PaddedC;
									for (auto c = offset; c < offset + PaddedC; c += VectorSize)
										mul_add(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[c])), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayer->NeuronsD1[c])).store_a(&InputLayer->NeuronsD1[c]);
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto offset = n * C;
									for (auto c = offset; c < offset + C; c++)
										InputLayer->NeuronsD1[c] += TanhExp::df(InputLayerFwd->Neurons[c]) * NeuronsD1[c];
								});
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
						if (InplaceBwd)
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW();
									for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
										(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[hw])), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW();
									for (auto hw = offset; hw < offset + HW(); hw++)
										InputLayer->NeuronsD1[hw] = TanhExp::df(InputLayerFwd->Neurons[hw]) * InputLayer->NeuronsD1[hw];
								}
							}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto offset = c * HW();
									for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
										mul_add(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[hw])), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
								}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto offset = c * HW();
									for (auto hw = offset; hw < offset + HW(); hw++)
										InputLayer->NeuronsD1[hw] += TanhExp::df(InputLayerFwd->Neurons[hw]) * NeuronsD1[hw];
								}
							}
						}
					}
					else
					{
#endif
						if (InplaceBwd)
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW() + c * HW();
										for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
											(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[hw])), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW() + c * HW();
										for (auto hw = offset; hw < offset + HW(); hw++)
											InputLayer->NeuronsD1[hw] = TanhExp::df(InputLayerFwd->Neurons[hw]) * InputLayer->NeuronsD1[hw];
									}
								});
						}
						else
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto offset = n * PaddedCDHW() + c * HW();
										for (auto hw = offset; hw < offset + strideHW; hw += VectorSize)
											mul_add(TanhExp::dfVec(VecFloat().load_a(&InputLayerFwd->Neurons[hw])), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayer->NeuronsD1[hw])).store_a(&InputLayer->NeuronsD1[hw]);
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto offset = n * CDHW() + c * HW();
										for (auto hw = offset; hw < offset + HW(); hw++)
											InputLayer->NeuronsD1[hw] += TanhExp::df(InputLayerFwd->Neurons[hw]) * NeuronsD1[hw];
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
				auto memSrc = dnnl::memory(*InputLayerFwd->DstMemDesc, Device.engine, InputLayerFwd->Neurons.data());
				auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderBwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				auto diffDstMem = dnnl::memory(bwdDesc->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput && !InplaceBwd ? dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayer->DiffDstMemDesc, Device.engine, InputLayer->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::eltwise_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : dnnl::memory(*DiffDstMemDesc, Device.engine, NeuronsD1.data()) }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
				Device.stream.wait();

				if (reorderBwdDiffSrc)
				{
					dnnl::reorder(diffSrcMem, memDiffSrc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, diffSrcMem}, { DNNL_ARG_TO, memDiffSrc } });
					Device.stream.wait();
				}

				if (SharesInput && !InplaceBwd)
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