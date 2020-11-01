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
		inline static VecFloat fVec(const VecFloat& x) noexcept { return min(VecFloat(1), max(VecFloat(0), x * VecFloat(0.2) + VecFloat(0.5))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x < VecFloat(-2.5) | x > VecFloat(2.5), VecFloat(0), VecFloat(0.2)); }
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
		inline static Float df(const Float& x) noexcept { return x < Float(-3.0) ? Float(0) : x > Float(3.0) ? Float(1) : (Float(1.0 / 3.0) * x + Float(0.5)); }
		inline static VecFloat fVec(const VecFloat& x) noexcept { return x * Relu6::fVec(x + VecFloat(3)) * VecFloat(1.0 / 6.0); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x < VecFloat(-3.0), VecFloat(0), select(x > VecFloat(3.0), VecFloat(1), (VecFloat(1.0 / 3.0)* x + VecFloat(0.5)))); }
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
		inline static Float f(const Float& x) noexcept { return x * std::tanhf(std::log1p(std::exp(x))); }
		inline static Float df(const Float& x) noexcept { const Float tmpExp = std::expf(x); const Float tmpSoftplus = std::log1p(tmpExp); const Float tmpSech = Float(1) / std::cosh(tmpSoftplus); return std::tanh(tmpSoftplus) + x * tmpExp * FloatSquare(tmpSech) / (tmpExp + Float(1)); }
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
		inline static VecFloat fVec(const VecFloat& x) noexcept { return VecFloat(1.0507009873554804934193349852946) * select(x > VecFloat(0), x, VecFloat(1.6732632423543772848170429916717) * (exp(x) - VecFloat(1))); }
		inline static VecFloat dfVec(const VecFloat& x) noexcept { return select(x > VecFloat(0), VecFloat(1.0507009873554804934193349852946), VecFloat(1.7580993408473768599402175208123) * exp(x)); }
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
		Relu = 16,
		Round = 17,
		Softmax = 18,
		SoftRelu = 19,
		Sqrt = 20,
		Square = 21,
		Swish = 22,
		Tanh = 23
	};

	class Activation final : public Layer
	{
	public:

		Activation(const dnn::Device& device, const dnnl::memory::format_tag format, const std::string& name, const Activations activation, const std::vector<Layer*>& inputs, const Float alpha = Float(0), const Float beta = Float(0));

		const Activations ActivationFunction;
		const Float Alpha;
		const Float Beta;

		std::string GetDescription() const final override;

		size_t FanIn() const final override;
		size_t FanOut() const final override;

		void InitializeDescriptors(const size_t batchSize) final override;

		void ForwardProp(const size_t batchSize, const bool training) final override;
		void BackwardProp(const size_t batchSize) final override;

	private:
		std::unique_ptr<dnnl::logsoftmax_forward::primitive_desc> fwdDescLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_backward::primitive_desc> bwdDescLogSoftmax;
		std::unique_ptr<dnnl::softmax_forward::primitive_desc> fwdDescSoftmax;
		std::unique_ptr<dnnl::softmax_backward::primitive_desc> bwdDescSoftmax;
		std::unique_ptr<dnnl::eltwise_forward::primitive_desc> fwdDesc;
		std::unique_ptr<dnnl::eltwise_backward::primitive_desc> bwdDesc;
		std::unique_ptr<dnnl::binary::primitive_desc> bwdAddDesc;

		std::unique_ptr<dnnl::logsoftmax_forward> fwdLogSoftmax;
		std::unique_ptr<dnnl::logsoftmax_backward> bwdLogSoftmax;
		std::unique_ptr<dnnl::softmax_forward> fwdSoftmax;
		std::unique_ptr<dnnl::softmax_backward> bwdSoftmax;
		std::unique_ptr<dnnl::eltwise_forward> fwd;
		std::unique_ptr<dnnl::eltwise_backward> bwd;
		std::unique_ptr<dnnl::binary> bwdAdd;

		dnnl::algorithm algorithm;
		bool reorderFwdSrc;
		bool reorderBwdSrc;
		bool reorderBwdDiffSrc;
	};
}