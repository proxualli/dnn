#pragma once
#include "Layer.h"

namespace dnn
{
	enum class Activations
	{
		Abs = 0,
		ASinh = 1,
		BoundedRelu = 2,
		Clip = 3,
		ClipV2 = 4,			//
		Elu = 5,			//
		Exp = 6,			//
		GeluErf = 7,
		GeluTanh = 8,
		HardSigmoid = 9,
		HardSwish = 10,
		Linear = 11,
		Log = 12,
		LogSigmoid = 13,
		Mish = 14,
		Pow = 15,
		Relu = 16,			//
		Round = 17,
		Selu = 18,
		Sigmoid = 19,		//
		SoftPlus = 20,
		SoftRelu = 21,
		SoftSign = 22,
		Sqrt = 23,			//
		Square = 24,
		Swish = 25,
		Tanh = 26,			//
		TanhExp = 27
	};

	struct Abs
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::abs(x); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1) : x < Float(0) ? Float(-1) : Float(0); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return abs(x); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), Float(1), select(x < Float(0), VecFloat(Float(-1)), VecFloat(Float(0)))); }
		inline static Activations Enum() NOEXCEPT { return Activations::Abs; }
	};

	struct ASinh
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::asinh(x); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / std::cosh(x); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return asinh(x); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / cosh(x); }
		inline static Activations Enum() NOEXCEPT { return Activations::ASinh; }
	};

	struct BoundedRelu // alpha >= 0
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(6), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::max(Float(0), std::min(alpha, x)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(6), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) && x <= alpha ? Float(1) : Float(0); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(6), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return max(Float(0), min(alpha, x)); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(6), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), select(x <= alpha, VecFloat(Float(1)), VecFloat(Float(0))), VecFloat(Float(0))); }
		inline static Activations Enum() NOEXCEPT { return Activations::BoundedRelu; }
	};

	struct Elu 
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? x : alpha * (std::exp(x) - Float(1)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1) : alpha * std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), x, alpha * (exp(x) - Float(1))); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), VecFloat(Float(1)), alpha * exp(x)); }
		inline static Activations Enum() NOEXCEPT { return Activations::Elu; }
	};

	struct Exp
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::exp(x); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return exp(x); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return exp(x); }
		inline static Activations Enum() NOEXCEPT { return Activations::Exp; }
	};

	struct HardSigmoid
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0.2), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return std::max(Float(0), std::min(Float(1), x * alpha + beta)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0.2), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return ((x > (-beta / alpha)) && (x < ((Float(1) - beta) / alpha))) ? alpha : Float(0); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0.2), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return max(Float(0), min(Float(1), x * alpha + beta)); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0.2), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return select((x > (-beta / alpha)) & (x < ((VecFloat(1) - beta) / alpha)), VecFloat(alpha), VecFloat(Float(0))); }
		inline static Activations Enum() NOEXCEPT { return Activations::HardSigmoid; }
	};

	struct HardSwish
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = (Float(1) / Float(6)), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return x * std::max(Float(0), std::min(Float(1), x * alpha + beta)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = (Float(1) / Float(6)), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return ((x >= (Float(1) - beta) / alpha) ? Float(1) : (((x > -beta / alpha) && x < ((Float(1) - beta) / alpha))) ? (Float(2) * x * alpha + beta) : Float(0)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = (Float(1) / Float(6)), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return x * max(Float(0), min(Float(1), x * alpha + beta)); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = (Float(1) / Float(6)), [[maybe_unused]] const Float& beta = Float(0.5)) NOEXCEPT { return select(x >= (Float(1) - beta) / alpha, VecFloat(Float(1)), select(x > - beta / alpha & x < (Float(1) - beta) / alpha, Float(2) * x * alpha + beta, VecFloat(Float(0)))); }
		inline static Activations Enum() NOEXCEPT { return Activations::HardSwish; }
	};

	struct Linear
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x * alpha + beta; }
		inline static Float df([[maybe_unused]] const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return alpha; }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x * alpha + beta; }
		inline static VecFloat dfVec([[maybe_unused]] const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return VecFloat(alpha); }
		inline static Activations Enum() NOEXCEPT { return Activations::Linear; }
	};

	struct Log
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::log(x); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / x; }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return log(x); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return VecFloat(Float(1)) / x; }
		inline static Activations Enum() NOEXCEPT { return Activations::Log; }
	};

	struct Pow
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return alpha * std::pow(x, beta); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return alpha * beta * std::pow(x, beta - Float(1)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return alpha * pow(x, beta); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return alpha * beta * pow(x, beta - Float(1)); }
		inline static Activations Enum() NOEXCEPT { return Activations::Pow; }
	};

	struct Sigmoid
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return (Float(1) / (Float(1) + std::exp(-x))); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { const auto y = Sigmoid::f(x); return ( y * (Float(1) - y)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return (Float(1) / (Float(1) + exp(-x))); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { const auto y = Sigmoid::fVec(x); return y * (Float(1) - y); }
		inline static Activations Enum() NOEXCEPT { return Activations::Sigmoid; }
	};
	
	struct SoftRelu
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::log(Float(1) + std::exp(alpha * x)) / alpha; }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / (Float(1) + std::exp(-alpha * x)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return log(Float(1) + exp(alpha * x)) / alpha; }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return VecFloat(Float(1)) / (VecFloat(Float(1)) + exp(-alpha * x)); }
		inline static Activations Enum() NOEXCEPT { return Activations::SoftRelu; }
	};

	struct LogSigmoid
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return -SoftRelu::f(-x); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / (std::exp(x) + Float(1)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return -SoftRelu::fVec(-x); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return VecFloat(Float(1)) / (exp(x) + VecFloat(Float(1))); }
		inline static Activations Enum() NOEXCEPT { return Activations::LogSigmoid; }
	};

	struct Mish
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x * std::tanh(std::log1p(std::exp(x))); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { const auto tmpExp = std::exp(x); const auto tmpSoftplus = std::log1p(tmpExp); const auto tmpSech = Float(1) / std::cosh(tmpSoftplus); return std::tanh(tmpSoftplus) + x * tmpExp * Square<Float>(tmpSech) / (tmpExp + Float(1)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x * tanh(log1p(exp(x))); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { const auto tmpExp = exp(x); const auto tmpSoftplus = log1p(tmpExp); const auto tmpSech = Float(1) / cosh(tmpSoftplus); return tanh(tmpSoftplus) + x * tmpExp * square(tmpSech) / (tmpExp + Float(1)); }
		inline static Activations Enum() NOEXCEPT { return Activations::Mish; }
	};

	struct Relu // alpha >= 0
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? x : x * alpha; }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1) : alpha; }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), x, x * alpha); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), VecFloat(Float(1)), VecFloat(alpha)); }
		inline static Activations Enum() NOEXCEPT { return Activations::Relu; }
	};
	
	struct Selu
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1.0507009873554804934193349852946) * (x > Float(0) ? x : Float(1.6732632423543772848170429916717) * (std::exp(x) - Float(1))); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x > Float(0) ? Float(1.0507009873554804934193349852946) : Float(1.7580993408473768599402175208123) * std::exp(x); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1.0507009873554804934193349852946) * select(x > Float(0), x, Float(1.6732632423543772848170429916717) * (exp(x) - Float(1))); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return select(x > Float(0), VecFloat(Float(1.0507009873554804934193349852946)), Float(1.7580993408473768599402175208123) * exp(x)); }
		inline static Activations Enum() NOEXCEPT { return Activations::Selu; }
	};

	struct SoftPlus
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(20), [[maybe_unused]] const Float& beta = Float(1)) NOEXCEPT { const auto y = beta * x; return y > alpha ? x : std::log1p(std::exp(y)) / beta; }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(20), [[maybe_unused]] const Float& beta = Float(1)) NOEXCEPT { const auto y = beta * x;  const auto tmpExp = std::exp(y); return y > alpha ? x : x * (tmpExp - Float(1)) / tmpExp; }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(20), [[maybe_unused]] const Float& beta = Float(1)) NOEXCEPT { const auto y = beta * x; return select(y > alpha, x, log1p(exp(y)) / beta); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(20), [[maybe_unused]] const Float& beta = Float(1)) NOEXCEPT { const auto y = beta * x; const auto tmpExp = exp(y); return select(y > alpha, x, x * (tmpExp - Float(1)) / tmpExp); }
		inline static Activations Enum() NOEXCEPT { return Activations::SoftPlus; }
	};
	
	struct SoftSign
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x / (Float(1) + std::abs(x)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / Square<Float>(Float(1) + std::abs(x)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x / (Float(1) + abs(x)); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) / square(Float(1) + abs(x)); }
		inline static Activations Enum() NOEXCEPT { return Activations::SoftSign; }
	};

	struct Swish
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x / (std::exp(-alpha * x) + Float(1)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return (Float(1) / (std::exp(-alpha * x) + Float(1))) * (Float(1) + alpha * x * (Float(1) - (Float(1) / (std::exp(-alpha * x) + Float(1))))); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x / (exp(-alpha * x) + Float(1)); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(1), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return (Float(1) / (exp(-alpha * x) + Float(1))) * (Float(1) + alpha * x * (Float(1) - (Float(1) / (exp(-alpha * x) + Float(1))))); }
		inline static Activations Enum() NOEXCEPT { return Activations::Swish; }
	};

	struct Tanh
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return std::tanh(x); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return Float(1) - Square<Float>(std::tanh(x)); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return tanh(x); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return (Float(1) - square(tanh(x))); }
		inline static Activations Enum() NOEXCEPT { return Activations::Tanh; }
	};
	 
	struct TanhExp
	{
		inline static Float f(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x * std::tanh(std::exp(x)); }
		inline static Float df(const Float& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { const auto y = std::exp(x);  const auto z = std::tanh(y); return z - (x * y * (Square<Float>(z) - Float(1))); }
		inline static VecFloat fVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { return x * tanh(exp(x)); }
		inline static VecFloat dfVec(const VecFloat& x, [[maybe_unused]] const Float& alpha = Float(0), [[maybe_unused]] const Float& beta = Float(0)) NOEXCEPT { const auto y = exp(x); const auto z = tanh(y); return z - (x * y * (square(z) - Float(1))); }
		inline static Activations Enum() NOEXCEPT { return Activations::TanhExp; }
	};
	
	// Dummy struct for an activation function
	struct Act
	{
		typedef Float(*FloatFuncPtrType)(const Float&, const Float&, const Float&);
		typedef VecFloat(*VecFloatFuncPtrType)(const VecFloat&, const Float&, const Float&);
		
		FloatFuncPtrType f, df;
		VecFloatFuncPtrType fVec, dfVec;
		Float alpha, beta;
		Activations Enum;
		dnnl::algorithm algorithm;
		bool test;
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
		const Act Func;

		static auto GetAlpha(const Activations activation, [[maybe_unused]] const Float alpha, [[maybe_unused]] const Float beta)
		{
			switch (activation)
			{
			case Activations::Abs:
			case Activations::ASinh:
			case Activations::Clip:
			case Activations::ClipV2:
			case Activations::Exp:
			case Activations::GeluErf:
			case Activations::GeluTanh:
			case Activations::Log:
			case Activations::Mish:
			case Activations::Round:
			case Activations::Selu:
			case Activations::Sigmoid:
			case Activations::SoftSign:
			case Activations::Sqrt:
			case Activations::Square:
			case Activations::Tanh:
			case Activations::TanhExp:
				break;
			case Activations::BoundedRelu:
				return alpha == Float(0) ? Float(6) : alpha;
			case Activations::Elu:
			case Activations::Linear:
			case Activations::Pow:
			case Activations::Relu:
			case Activations::SoftRelu:
			case Activations::Swish:
				return alpha == Float(0) ? Float(1) : alpha;
			case Activations::SoftPlus:
				return alpha == Float(0) ? Float(20) : alpha;
			case Activations::HardSigmoid:
				return alpha == Float(0) ? Float(0.2) : alpha;
			case Activations::HardSwish:
				return alpha == Float(0) ? (Float(1) / Float(6)) : alpha;
			case Activations::LogSigmoid:
				return Float(-1);
			}

			return alpha;
		}

		static auto GetBeta(const Activations activation, [[maybe_unused]] const Float alpha, [[maybe_unused]] const Float beta)
		{
			switch (activation)
			{
			case Activations::Abs:
			case Activations::ASinh:
			case Activations::Clip:
			case Activations::ClipV2:
			case Activations::Elu:
			case Activations::Exp:
			case Activations::GeluErf:
			case Activations::GeluTanh:
			case Activations::Linear:
			case Activations::Log:
			case Activations::LogSigmoid:
			case Activations::Mish:
			case Activations::Pow:
			case Activations::Relu:
			case Activations::Round:
			case Activations::Selu:
			case Activations::Sigmoid:
			case Activations::SoftRelu:
			case Activations::SoftSign:
			case Activations::Sqrt:
			case Activations::Square:
			case Activations::Swish:
			case Activations::Tanh:
			case Activations::TanhExp:
				break;
			case Activations::BoundedRelu:
				return Float(0);
			case Activations::HardSigmoid:
			case Activations::HardSwish:
				return beta == Float(0) ? Float(0.5) : beta;
			case Activations::SoftPlus:
				return beta == Float(0) ? Float(1) : beta;
			}

			return beta;
		}

		static auto GetActivation(Activations activation)
		{
			Act act = {};

			switch (activation)
			{
			case Activations::Abs:
				act.f = &Abs::f;
				act.df = &Abs::df;
				act.fVec = &Abs::fVec;
				act.dfVec = &Abs::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = Abs::Enum();
				act.algorithm = dnnl::algorithm::eltwise_abs;
				act.test = true;
				break;

			case Activations::ASinh:
				act.f = &ASinh::f;
				act.df = &ASinh::df;
				act.fVec = &ASinh::fVec;
				act.dfVec = &ASinh::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = ASinh::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = false;
				break;

			case Activations::BoundedRelu:
				act.f = &BoundedRelu::f;
				act.df = &BoundedRelu::df;
				act.fVec = &BoundedRelu::fVec;
				act.dfVec = &BoundedRelu::dfVec;
				act.alpha = Float(6);
				act.beta = Float(0);
				act.Enum = BoundedRelu::Enum();
				act.algorithm = dnnl::algorithm::eltwise_clip;
				act.test = true;
				break;

			case Activations::Clip:
				act.alpha = Float(0);
				act.beta = Float(0);
				act.algorithm = dnnl::algorithm::eltwise_clip;
				act.test = false;
				break;

			case Activations::ClipV2:
				act.alpha = Float(0);
				act.beta = Float(0);
				act.algorithm = dnnl::algorithm::eltwise_clip_v2;
				act.test = false;
				break;

			case Activations::Elu:
				act.f = &Elu::f;
				act.df = &Elu::df;
				act.fVec = &Elu::fVec;
				act.dfVec = &Elu::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = Elu::Enum();
				act.algorithm = dnnl::algorithm::eltwise_elu;
				act.test = true;
				break;

			case Activations::Exp:
				act.f = &Exp::f;
				act.df = &Exp::df;
				act.fVec = &Exp::fVec;
				act.dfVec = &Exp::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = Exp::Enum();
				act.algorithm = dnnl::algorithm::eltwise_exp;
				act.test = true;
				break;

			case Activations::GeluErf:
				act.alpha = Float(0);
				act.beta = Float(0);
				act.algorithm = dnnl::algorithm::eltwise_gelu_erf;
				act.test = false;
				break;

			case Activations::GeluTanh:
				act.alpha = Float(0);
				act.beta = Float(0);
				act.algorithm = dnnl::algorithm::eltwise_gelu_tanh;
				act.test = false;
				break;			

			case Activations::HardSigmoid:
				act.f = &HardSigmoid::f;
				act.df = &HardSigmoid::df;
				act.fVec = &HardSigmoid::fVec;
				act.dfVec = &HardSigmoid::dfVec;
				act.alpha = Float(0.2);
				act.beta = Float(0.5);
				act.Enum = HardSigmoid::Enum();
				act.algorithm = dnnl::algorithm::eltwise_hardsigmoid;
				act.test = true;
				break;

			case Activations::HardSwish:
				act.f = &HardSwish::f;
				act.df = &HardSwish::df;
				act.fVec = &HardSwish::fVec;
				act.dfVec = &HardSwish::dfVec;
				act.alpha = Float(1) / Float(6);
				act.beta = Float(0.5);
				act.Enum = HardSwish::Enum();
				act.algorithm = dnnl::algorithm::eltwise_hardswish;
				act.test = true;
				break;

			case Activations::Linear:
				act.f = &Linear::f;
				act.df = &Linear::df;
				act.fVec = &Linear::fVec;
				act.dfVec = &Linear::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = Linear::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = true;
				break;

			case Activations::Log:
				act.f = &Log::f;
				act.df = &Log::df;
				act.fVec = &Log::fVec;
				act.dfVec = &Log::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = Log::Enum();
				act.algorithm = dnnl::algorithm::eltwise_log;
				act.test = true;
				break;

			case Activations::LogSigmoid:
				act.f = &LogSigmoid::f;
				act.df = &LogSigmoid::df;
				act.fVec = &LogSigmoid::fVec;
				act.dfVec = &LogSigmoid::dfVec;
				act.alpha = Float(-1);
				act.beta = Float(0);
				act.Enum = LogSigmoid::Enum();
				act.algorithm = dnnl::algorithm::eltwise_soft_relu;
				act.test = true;
				break;

			case Activations::Mish:
				act.f = &Mish::f;
				act.df = &Mish::df;
				act.fVec = &Mish::fVec;
				act.dfVec = &Mish::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = Mish::Enum();
				act.algorithm = dnnl::algorithm::eltwise_mish;
				act.test = true;
				break;

			case Activations::Pow:
				act.f = &Pow::f;
				act.df = &Pow::df;
				act.fVec = &Pow::fVec;
				act.dfVec = &Pow::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = Pow::Enum();
				act.algorithm = dnnl::algorithm::eltwise_pow;
				act.test = false;
				break;

			case Activations::Relu:
				act.f = &Relu::f;
				act.df = &Relu::df;
				act.fVec = &Relu::fVec;
				act.dfVec = &Relu::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = Relu::Enum();
				act.algorithm = dnnl::algorithm::eltwise_relu;
				act.test = true;
				break;

			case Activations::Selu:
				act.f = &Selu::f;
				act.df = &Selu::df;
				act.fVec = &Selu::fVec;
				act.dfVec = &Selu::dfVec;
				act.alpha = Float(20);
				act.beta = Float(1);
				act.Enum = Selu::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = false;
				break;

			case Activations::Sigmoid:
				act.f = &Sigmoid::f;
				act.df = &Sigmoid::df;
				act.fVec = &Sigmoid::fVec;
				act.dfVec = &Sigmoid::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = Sigmoid::Enum();
				act.algorithm = dnnl::algorithm::eltwise_logistic;
				act.test = true;
				break;

			case Activations::SoftPlus:
				act.f = &SoftPlus::f;
				act.df = &SoftPlus::df;
				act.fVec = &SoftPlus::fVec;
				act.dfVec = &SoftPlus::dfVec;
				act.alpha = Float(20);
				act.beta = Float(1);
				act.Enum = SoftPlus::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = false;
				break;

			case Activations::SoftRelu:
				act.f = &SoftRelu::f;
				act.df = &SoftRelu::df;
				act.fVec = &SoftRelu::fVec;
				act.dfVec = &SoftRelu::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = SoftRelu::Enum();
				act.algorithm = dnnl::algorithm::eltwise_soft_relu;
				act.test = true;
				break;

			case Activations::SoftSign:
				act.f = &SoftSign::f;
				act.df = &SoftSign::df;
				act.fVec = &SoftSign::fVec;
				act.dfVec = &SoftSign::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = SoftSign::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = false;
				break;

			case Activations::Swish:
				act.f = &Swish::f;
				act.df = &Swish::df;
				act.fVec = &Swish::fVec;
				act.dfVec = &Swish::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = Swish::Enum();
				act.algorithm = dnnl::algorithm::eltwise_swish;
				act.test = true;
				break;

			case Activations::Tanh:
				act.f = &Tanh::f;
				act.df = &Tanh::df;
				act.fVec = &Tanh::fVec;
				act.dfVec = &Tanh::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = Tanh::Enum();
				act.algorithm = dnnl::algorithm::eltwise_tanh;
				act.test = true;
				break;

			case Activations::TanhExp:
				act.f = &TanhExp::f;
				act.df = &TanhExp::df;
				act.fVec = &TanhExp::fVec;
				act.dfVec = &TanhExp::dfVec;
				act.alpha = Float(0);
				act.beta = Float(0);
				act.Enum = TanhExp::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = false;
				break;

			default:
				act.f = &Linear::f;
				act.df = &Linear::df;
				act.fVec = &Linear::fVec;
				act.dfVec = &Linear::dfVec;
				act.alpha = Float(1);
				act.beta = Float(0);
				act.Enum = Linear::Enum();
				act.algorithm = dnnl::algorithm::eltwise_linear;
				act.test = false;
				break;
			}

			return act;
		}

		static bool CheckActivations(std::string& msg, const Float errorLimit = Float(0.00001))
		{
			msg = std::string("");
			std::atomic<bool> ret(true);
			
			if constexpr (TestActivations)
			{
				std::mutex lock;
				const auto eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
				auto stream = dnnl::stream(eng);
				auto activations = magic_enum::enum_names<Activations>();

				std::for_each(std::execution::par, activations.begin(), activations.end(), [&msg, &errorLimit, &ret, &lock, &eng, &stream](const std::string_view& activation)
				{
					if (magic_enum::enum_cast<Activations>(activation).has_value())
					{
						auto act = Activation::GetActivation(magic_enum::enum_cast<Activations>(activation).value());

						if (act.test)
						{
							auto lmsg = std::string();

							const auto N = dnnl::memory::dim(64);
							const auto C = dnnl::memory::dim(64);
							const auto H = dnnl::memory::dim(64);
							const auto W = dnnl::memory::dim(64);

							auto memDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ N, C, H, W }), dnnl::memory::data_type::f32, PlainFmt));

							auto fwdDesc = std::make_unique<dnnl::eltwise_forward::primitive_desc>(dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward, act.algorithm, *memDesc, *memDesc, (act.Enum == Activations::BoundedRelu) ? act.beta : act.alpha, (act.Enum == Activations::BoundedRelu) ? act.alpha : act.beta));
							auto bwdDesc = std::make_unique<dnnl::eltwise_backward::primitive_desc>(dnnl::eltwise_backward::primitive_desc(eng, act.algorithm, *memDesc, *memDesc, *memDesc, (act.Enum == Activations::BoundedRelu) ? act.beta : act.alpha, (act.Enum == Activations::BoundedRelu) ? act.alpha : act.beta, *fwdDesc));
#ifdef DNN_CACHE_PRIMITIVES
							auto fwd = std::make_unique<dnnl::eltwise_forward>(dnnl::eltwise_forward(*fwdDesc));
							auto bwd = std::make_unique<dnnl::eltwise_backward>(dnnl::eltwise_backward(*bwdDesc));
#endif							
							const auto size = UInt(N * C * H * W);
							const auto part = (size / 2ull) + (size / 4ull);

							const auto margin = Float(1.5);
							const auto minLimit = ((act.Enum == Activations::Exp) || (act.Enum == Activations::Elu) || (act.Enum == Activations::Log) || (act.Enum == Activations::Mish)) ? margin + Float(0.1): ((act.alpha != Float(0)) ? (-act.beta / act.alpha) : -margin);
							const auto maxLimit = (act.alpha != Float(0)) ? ((Float(1) - act.beta) / act.alpha) : margin;
														
							auto input = FloatVector(size);
							for (auto i = 0ull; i < size; i += VectorSize)
								UniformVecFloat(minLimit - margin, maxLimit + margin).store_a(&input[i]);

							auto outputFwd = FloatVector(size);
							auto outputBwd = FloatVector(size);

							try
							{
								std::feclearexcept(FE_ALL_EXCEPT);

								for (auto i = 0ull; i < part; i += VectorSize)
								{
									act.fVec(VecFloat().load_a(&input[i]), act.alpha, act.beta).store_a(&outputFwd[i]);
									act.dfVec(VecFloat().load_a(&input[i]), act.alpha, act.beta).store_a(&outputBwd[i]);
								}
								for (auto i = part; i < size; i++)
								{
									outputFwd[i] = act.f(input[i], act.alpha, act.beta);
									outputBwd[i] = act.df(input[i], act.alpha, act.beta);
								}

								int fe = std::fetestexcept(FE_ALL_EXCEPT & ~(FE_INEXACT | FE_UNDERFLOW));

								if (fe & FE_DIVBYZERO)
								{
									if (ret.load())
										ret.store(false);
									lmsg.append(std::string(activation) + std::string(" FE_DIVBYZERO") + nwl);
								}
								if (fe & FE_INEXACT)
								{
									if (ret.load())
										ret.store(false);
									lmsg.append(std::string(activation) + std::string(" FE_INEXACT") + nwl);
								}
								if (fe & FE_INVALID)
								{
									if (ret.load())
										ret.store(false);
									lmsg.append(std::string(activation) + std::string(" FE_INVALID") + nwl);
								}
								if (fe & FE_OVERFLOW)
								{
									if (ret.load())
										ret.store(false);
									lmsg.append(std::string(activation) + std::string(" FE_OVERFLOW") + nwl);
								}
								if (fe & FE_UNDERFLOW)
								{
									if (ret.load())
										ret.store(false);
									lmsg.append(std::string(activation) + std::string(" FE_UNDERFLOW") + nwl);
								}
							}
							catch (const std::exception& e)
							{
								if (ret.load())
									ret.store(false);
								lmsg.append(std::string(activation) + nwl + std::string(e.what()) + nwl);
							}

							if (ret.load())
							{
								auto outputFwdRef = FloatVector(size);
								auto outputBwdRef = FloatVector(size, Float(1));

								auto srcMem = dnnl::memory(*memDesc, eng, input.data());
								auto dstMem = dnnl::memory(*memDesc, eng, outputFwdRef.data());
#ifdef DNN_CACHE_PRIMITIVES
								fwd->execute(stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
#else
								dnnl::eltwise_forward(*fwdDesc).execute(stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DST, dstMem } });
#endif
								stream.wait();


								auto diffSrcMem = dnnl::memory(*memDesc, eng, outputBwdRef.data());
#ifdef DNN_CACHE_PRIMITIVES
								bwd->execute(stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffSrcMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
								dnnl::eltwise_backward(*bwdDesc).execute(stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, diffSrcMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#endif
								stream.wait();

								for (auto i = 0ull; i < size; i += VectorSize)
								{
									const auto fwdRef = VecFloat().load_a(&outputFwdRef[i]);
									const auto fwdVal = VecFloat().load_a(&outputFwd[i]);
									const auto fwdRet = ((fwdRef - errorLimit) > fwdVal) | ((fwdRef + errorLimit) < fwdVal);
									const bool fwdErr = horizontal_or(fwdRet);

									if (fwdErr)
									{
										const auto index = i + horizontal_find_first(fwdRet);
										const auto in = input[index];
										const auto ref = outputFwdRef[index];
										const auto out = outputFwd[index];

										lmsg.append(
											std::string(activation) + std::string(" forward pass not passed") + nwl +
											std::string("In:") + tab + std::to_string(in) + nwl +
											std::string("Ref:") + tab + std::to_string(ref) + nwl +
											std::string("Out:") + tab + std::to_string(out) + nwl);
									}

									const auto bwdRef = VecFloat().load_a(&outputBwdRef[i]);
									const auto bwdVal = VecFloat().load_a(&outputBwd[i]);
									const auto bwdRet = ((bwdRef - errorLimit) > bwdVal) | ((bwdRef + errorLimit) < bwdVal);
									const bool bwdErr = horizontal_or(fwdRet);

									if (bwdErr)
									{
										const auto index = i + horizontal_find_first(bwdRet);
										const auto in = input[index];
										const auto ref = outputBwdRef[index];
										const auto out = outputBwd[index];

										lmsg.append(
											std::string(activation) + std::string(" backward pass not passed") + nwl +
											std::string("In:") + tab + std::to_string(in) + nwl +
											std::string("Ref:") + tab + std::to_string(ref) + nwl +
											std::string("Out:") + tab + std::to_string(out) + nwl);
									}

									if (fwdErr || bwdErr)
									{
										if (ret.load())
											ret.store(false);

										break;
									}
								}
							}
							
							if (!ret.load())
							{
								lock.lock();
								msg.append(lmsg);
								lock.unlock();
							}
						}
					}
				});
			}

			return ret.load();
		}

		Activation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector<Layer*>& inputs, const Float alpha = Float(0), const Float beta = Float(0)) :
			Layer(device, format, name, LayerTypes::Activation, 0, 0, inputs[0]->C, inputs[0]->D, inputs[0]->H, inputs[0]->W, 0, 0, 0, inputs, false),
			ActivationFunction(activation),
			Alpha(Activation::GetAlpha(activation, alpha, beta)),
			Beta(Activation::GetBeta(activation, alpha, beta)),
			Func(Activation::GetActivation(activation)),
			algorithm(dnnl::algorithm::eltwise_linear),
			reorderFwdSrc(false),
			reorderBwdSrc(false),
			reorderBwdDiffSrc(false)
		{
			assert(Inputs.size() == 1);

			FwdZeroGradient = Float(1);
			FwdInferenceWeight = Float(5);
			FwdTrainingWeight = Float(10);
			BwdTrainingWeight = Float(10);
		}
			
		void UpdateResolution() final override
		{
			H = InputLayer->H;
			W = InputLayer->W;
		}

		std::string GetDescription() const final override
		{
			auto description = GetDescriptionHeader();

			description.append(nwl + std::string(" Activation: ") + tab + std::string(magic_enum::enum_name<Activations>(ActivationFunction)));
			description.append(nwl + std::string(" Alpha:  ") + dtab + FloatToString(Alpha));
			description.append(nwl + std::string(" Beta:   ") + dtab + FloatToString(Beta));

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
			auto alpha = Alpha;
			auto beta = Beta;

			switch (ActivationFunction)
			{
				case Activations::ASinh:
				case Activations::Selu:
				case Activations::SoftPlus:
				case Activations::SoftSign:
				case Activations::TanhExp:
				    break;

				case Activations::Abs:
					algorithm = dnnl::algorithm::eltwise_abs;
					break;
				case Activations::BoundedRelu:
					algorithm = dnnl::algorithm::eltwise_clip;
					beta = alpha;
					alpha = Float(0);
					break;
				case Activations::Clip:
					algorithm = dnnl::algorithm::eltwise_clip;
					break;
				case Activations::ClipV2:
					algorithm = dnnl::algorithm::eltwise_clip_v2;
					break;
				case Activations::Elu:
					algorithm = dnnl::algorithm::eltwise_elu;
					break;
				case Activations::Exp:
					algorithm = dnnl::algorithm::eltwise_exp;
					break;
				case Activations::GeluErf:
					algorithm = dnnl::algorithm::eltwise_gelu_erf;
					break;
				case Activations::GeluTanh:
					algorithm = dnnl::algorithm::eltwise_gelu_tanh;
					break;
				case Activations::HardSigmoid:
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
				case Activations::Sigmoid:
					algorithm = dnnl::algorithm::eltwise_logistic;
					break;
				case Activations::LogSigmoid:
					algorithm = dnnl::algorithm::eltwise_soft_relu;
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

			if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
			{
				ChosenFormat = dnnl::memory::format_tag::nc;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C) }), dnnl::memory::data_type::f32, ChosenFormat));
			}
			else
			{
				if (NeuronsFormat == dnnl::memory::format_tag::any)
					ChosenFormat = LayerBeforeCost || IsPlainDataFmt(*InputLayer->DstMemDesc) ? PlainFmt : GetMemoryFormat(*InputLayer->DstMemDesc);
				else
					ChosenFormat = PlainFmt;

				DstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
				DiffDstMemDesc = std::make_unique<dnnl::memory::desc>(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(batchSize), dnnl::memory::dim(C), dnnl::memory::dim(H), dnnl::memory::dim(W) }), dnnl::memory::data_type::f32, ChosenFormat));
			}

			fwdDesc = std::make_unique<dnnl::eltwise_forward::primitive_desc>(dnnl::eltwise_forward::primitive_desc(Device.engine, dnnl::prop_kind::forward, algorithm, *InputLayer->DstMemDesc, *DstMemDesc, alpha, beta));
			bwdDesc = std::make_unique<dnnl::eltwise_backward::primitive_desc>(dnnl::eltwise_backward::primitive_desc(Device.engine, algorithm, *InputLayer->DiffDstMemDesc, *DiffDstMemDesc, *DstMemDesc, alpha, beta, *fwdDesc));

			bwdAddDesc = std::make_unique<dnnl::binary::primitive_desc>(dnnl::binary::primitive_desc(Device.engine, dnnl::algorithm::binary_add, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc, *InputLayer->DiffDstMemDesc));

			reorderFwdSrc = fwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdSrc = bwdDesc->src_desc() != *InputLayer->DstMemDesc;
			reorderBwdDiffSrc = bwdDesc->diff_src_desc() != *InputLayer->DiffDstMemDesc;

#ifdef DNN_CACHE_PRIMITIVES
			fwd = std::make_unique<dnnl::eltwise_forward>(dnnl::eltwise_forward(*fwdDesc));
			bwd = std::make_unique<dnnl::eltwise_backward>(dnnl::eltwise_backward(*bwdDesc));
			bwdAdd = std::make_unique<dnnl::binary>(dnnl::binary(*bwdAddDesc));
#endif
		}

		void ForwardProp(const UInt batchSize, const bool training) final override
		{
			switch (ActivationFunction)
			{
			case Activations::ASinh:
			case Activations::Selu:
			case Activations::SoftPlus:
			case Activations::SoftSign:
			case Activations::TanhExp:
			{
				const auto plain = IsPlainFormat();
				const auto strideHW = HW() * VectorSize;

				if (GetMemoryNDims(*InputLayer->DstMemDesc) == 2)
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
									Func.fVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta).store_a(&Neurons[c]);
#ifndef DNN_LEAN
									if (!InplaceBwd)
										VecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
								}
							}
							else
								for (auto c = 0ull; c < C; c++)
								{
									Neurons[c] = Func.f(InputLayer->Neurons[c], Alpha, Beta);
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
									Func.fVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta).store_a(&Neurons[c]);
							else
								for (auto c = 0ull; c < C; c++)
									Neurons[c] = Func.f(InputLayer->Neurons[c], Alpha, Beta);
						}
					}
					else
					{
#endif
						if (training)
						{
							const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * PaddedC;
									for (auto c = start; c < start + PaddedC; c += VectorSize)
									{
										Func.fVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta).store_a(&Neurons[c]);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											VecZero.store_nt(&NeuronsD1[c]);
#endif // DNN_LEAN
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * C;
									for (auto c = start; c < start + C; c++)
									{
										Neurons[c] = Func.f(InputLayer->Neurons[c], Alpha, Beta);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											NeuronsD1[c] = Float(0);
#endif // DNN_LEAN
									}
								});
						}
						else
						{
							const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdInferenceWeight);

							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * PaddedC;
									for (auto c = start; c < start + PaddedC; c += VectorSize)
										Func.fVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta).store_a(&Neurons[c]);
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * C;
									for (auto c = start; c < start + C; c++)
										Neurons[c] = Func.f(InputLayer->Neurons[c], Alpha, Beta);
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
									const auto start = c * HW();
									for (auto hw = start; hw < start + strideHW; hw += VectorSize)
									{
										Func.fVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											VecZero.store_nt(&NeuronsD1[hw]);
#endif // DNN_LEAN
									}
								}
							else
								for (auto c = 0ull; c < C; c++)
								{
									const auto start = c * HW();
									for (auto hw = start; hw < start + HW(); hw++)
									{
										Neurons[hw] = Func.f(InputLayer->Neurons[hw], Alpha, Beta);
#ifndef DNN_LEAN
										if (!InplaceBwd)
											NeuronsD1[hw] = Float(0);
#endif // DNN_LEAN
									}
								}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto start = c * HW();
									for (auto hw = start; hw < start + strideHW; hw += VectorSize)
										Func.fVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
								}
							else
								for (auto c = 0ull; c < C; c++)
								{
									const auto start = c * HW();
									for (auto hw = start; hw < start + HW(); hw++)
										Neurons[hw] = Func.f(InputLayer->Neurons[hw], Alpha, Beta);
								}
						}
					}
					else
					{
#endif
						if (training)
						{
							const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdTrainingWeight);

							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto start = n * PaddedCDHW() + c * HW();
										for (auto hw = start; hw < start + strideHW; hw += VectorSize)
										{
											Func.fVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
#ifndef DNN_LEAN
											if (!InplaceBwd)
												VecZero.store_nt(&NeuronsD1[hw]);
#endif // DNN_LEAN
										}
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto start = n * CDHW() + c * HW();
										for (auto hw = start; hw < start + HW(); hw++)
										{
											Neurons[hw] = Func.f(InputLayer->Neurons[hw], Alpha, Beta);
#ifndef DNN_LEAN
											if (!InplaceBwd)
												NeuronsD1[hw] = Float(0);
#endif // DNN_LEAN
										}
									}
								});
						}
						else
						{
							const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * GetElementsCount(), FwdInferenceWeight);

							if (!plain)
							{
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < PaddedC; c += VectorSize)
									{
										const auto start = n * PaddedCDHW() + c * HW();
										for (auto hw = start; hw < start + strideHW; hw += VectorSize)
											Func.fVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta).store_a(&Neurons[hw]);
									}
								});
							}
							else
							{
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto start = n * CDHW() + c * HW();
										for (auto hw = start; hw < start + HW(); hw++)
											Neurons[hw] = Func.f(InputLayer->Neurons[hw], Alpha, Beta);
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
				const auto& memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
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
					InitArray<Float>(NeuronsD1.data(), PaddedCDHW(), batchSize, FwdZeroGradient);
#endif
			}
			}
		}

		void BackwardProp(const UInt batchSize) final override
		{
#ifdef DNN_LEAN
			ZeroGradient(batchSize);
#endif // DNN_LEAN

			switch (ActivationFunction)
			{
			case Activations::ASinh:
			case Activations::Selu:
			case Activations::SoftPlus:
			case Activations::SoftSign:
			case Activations::TanhExp:
			{
				const auto plain = IsPlainFormat();
				const auto threads = batchSize == 1 ? 1ull : GetThreads(batchSize * GetElementsCount(), BwdTrainingWeight);
				const auto strideHW = HW() * VectorSize;

				if (GetMemoryNDims(*InputLayerBwd->DstMemDesc) == 2)
				{
#ifdef DNN_STOCHASTIC
					if (batchSize == 1)
					{
						if (InplaceBwd)
						{
							if (!plain)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
									(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta), VecFloat().load_a(&InputLayerBwd->NeuronsD1[c])).store_a(&InputLayerBwd->NeuronsD1[c]);
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
									InputLayerBwd->NeuronsD1[c] = Func.df(InputLayer->Neurons[c], Alpha, Beta) * InputLayerBwd->NeuronsD1[c];
							}
						}
						else
						{
							if (!plain)
							{
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
									mul_add(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayerBwd->NeuronsD1[c])).store_a(&InputLayerBwd->NeuronsD1[c]);
							}
							else
							{
								for (auto c = 0ull; c < C; c++)
									InputLayerBwd->NeuronsD1[c] += Func.df(InputLayer->Neurons[c], Alpha, Beta) * NeuronsD1[c];
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
									const auto start = n * PaddedC;
									for (auto c = start; c < start + PaddedC; c += VectorSize)
										(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta), VecFloat().load_a(&InputLayerBwd->NeuronsD1[c])).store_a(&InputLayerBwd->NeuronsD1[c]);
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * C;
									for (auto c = start; c < start + C; c++)
										InputLayerBwd->NeuronsD1[c] = Func.df(InputLayer->Neurons[c], Alpha, Beta) * InputLayerBwd->NeuronsD1[c];
								});
						}
						else
						{
							if (!plain)
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * PaddedC;
									for (auto c = start; c < start + PaddedC; c += VectorSize)
										mul_add(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[c]), Alpha, Beta), VecFloat().load_a(&NeuronsD1[c]), VecFloat().load_a(&InputLayerBwd->NeuronsD1[c])).store_a(&InputLayerBwd->NeuronsD1[c]);
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									const auto start = n * C;
									for (auto c = start; c < start + C; c++)
										InputLayerBwd->NeuronsD1[c] += Func.df(InputLayer->Neurons[c], Alpha, Beta) * NeuronsD1[c];
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
									const auto start = c * HW();
									for (auto hw = start; hw < start + strideHW; hw += VectorSize)
										(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta), VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw])).store_a(&InputLayerBwd->NeuronsD1[hw]);
								}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto start = c * HW();
									for (auto hw = start; hw < start + HW(); hw++)
										InputLayerBwd->NeuronsD1[hw] = Func.df(InputLayer->Neurons[hw], Alpha, Beta) * InputLayerBwd->NeuronsD1[hw];
								}
							}
						}
						else
						{
							if (!plain)
								for (auto c = 0ull; c < PaddedC; c += VectorSize)
								{
									const auto start = c * HW();
									for (auto hw = start; hw < start + strideHW; hw += VectorSize)
										mul_add(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw])).store_a(&InputLayerBwd->NeuronsD1[hw]);
								}
							else
							{
								for (auto c = 0ull; c < C; c++)
								{
									const auto start = c * HW();
									for (auto hw = start; hw < start + HW(); hw++)
										InputLayerBwd->NeuronsD1[hw] += Func.df(InputLayer->Neurons[hw], Alpha, Beta) * NeuronsD1[hw];
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
										const auto start = n * PaddedCDHW() + c * HW();
										for (auto hw = start; hw < start + strideHW; hw += VectorSize)
											(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta), VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw])).store_a(&InputLayerBwd->NeuronsD1[hw]);
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto start = n * CDHW() + c * HW();
										for (auto hw = start; hw < start + HW(); hw++)
											InputLayerBwd->NeuronsD1[hw] = Func.df(InputLayer->Neurons[hw], Alpha, Beta) * InputLayerBwd->NeuronsD1[hw];
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
										const auto start = n * PaddedCDHW() + c * HW();
										for (auto hw = start; hw < start + strideHW; hw += VectorSize)
											mul_add(Func.dfVec(VecFloat().load_a(&InputLayer->Neurons[hw]), Alpha, Beta), VecFloat().load_a(&NeuronsD1[hw]), VecFloat().load_a(&InputLayerBwd->NeuronsD1[hw])).store_a(&InputLayerBwd->NeuronsD1[hw]);
									}
								});
							else
								for_i(batchSize, threads, [=](UInt n)
								{
									for (auto c = 0ull; c < C; c++)
									{
										const auto start = n * CDHW() + c * HW();
										for (auto hw = start; hw < start + HW(); hw++)
											InputLayerBwd->NeuronsD1[hw] += Func.df(InputLayer->Neurons[hw], Alpha, Beta) * NeuronsD1[hw];
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
				const auto& memSrc = dnnl::memory(*InputLayer->DstMemDesc, Device.engine, InputLayer->Neurons.data());
				auto srcMem = reorderBwdSrc ? dnnl::memory(bwdDesc->src_desc(), Device.engine) : memSrc;
				if (reorderBwdSrc)
				{
					dnnl::reorder(memSrc, srcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memSrc}, { DNNL_ARG_TO, srcMem } });
					Device.stream.wait();
				}

				const auto& diffDstMem = dnnl::memory(bwdDesc->diff_dst_desc(), Device.engine, NeuronsD1.data());

				auto memDiffSrc = SharesInput ? dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine) : dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data());
				auto diffSrcMem = reorderBwdDiffSrc ? dnnl::memory(bwdDesc->diff_src_desc(), Device.engine) : memDiffSrc;

				//if (reorderBwdDiffSrc)
				//{
				//	dnnl::reorder(memDiffSrc, diffSrcMem).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_FROM, memDiffSrc}, { DNNL_ARG_TO, diffSrcMem } });
				//	Device.stream.wait();
				//}

#ifdef DNN_CACHE_PRIMITIVES
				bwd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
#else
				dnnl::eltwise_backward(*bwdDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ {DNNL_ARG_SRC, srcMem}, { DNNL_ARG_DIFF_DST, InplaceBwd ? diffSrcMem : diffDstMem }, { DNNL_ARG_DIFF_SRC, diffSrcMem } });
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
					bwdAdd->execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) } });
#else
					dnnl::binary(*bwdAddDesc).execute(Device.stream, std::unordered_map<int, dnnl::memory>{ { DNNL_ARG_SRC_0, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) }, { DNNL_ARG_SRC_1, memDiffSrc }, { DNNL_ARG_DST, dnnl::memory(*InputLayerBwd->DiffDstMemDesc, Device.engine, InputLayerBwd->NeuronsD1.data()) } });
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