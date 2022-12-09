#pragma once
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
#include "stdafx.h"
#else
#include <sys/sysinfo.h>
#endif

#ifdef NDEBUG
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

#ifndef MAX_VECTOR_SIZE
#ifdef DNN_SSE41
#define INSTRSET 5
#define MAX_VECTOR_SIZE 128
#endif // DNN_SSE41

#ifdef DNN_SSE42
#define INSTRSET 6
#define MAX_VECTOR_SIZE 128
#endif // DNN_SSE42

#ifdef DNN_AVX
#define INSTRSET 7
#define MAX_VECTOR_SIZE 256
#endif //DNN_AVX

#ifdef DNN_AVX2
#define INSTRSET 8
#define MAX_VECTOR_SIZE 256
#endif //DNN_AVX2

#ifdef DNN_AVX512
#define INSTRSET 9
#define MAX_VECTOR_SIZE 512
#endif //DNN_AVX512

#ifdef DNN_AVX512BW
#define INSTRSET 10
#define MAX_VECTOR_SIZE 512
#endif //DNN_AVX512BW
#endif // MAX_VECTOR_SIZE

#include "instrset.h"
#include "vectorclass.h"
#include "vectormath_common.h"
#include "vectormath_exp.h"
#include "vectormath_hyp.h"
#include "vectormath_trig.h"
#include "add-on/random/ranvec1.h"

#include <algorithm>
#include <array>
#include <atomic>
//#include <bit>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional> 
#include <future>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <locale>
#include <clocale>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <utility>

#include "dnnl.hpp"
#include "dnnl_debug.h"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

#include "AlignedAllocator.h"
#include "ParallelFor.h"

#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 255
#include "magic_enum.hpp"

//#include "csv.hpp"

using namespace dnn;

namespace
{
#ifdef _MSC_VER
#define DNN_ALIGN(alignment) __declspec(align(alignment))
#else
#define DNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#endif
#define DNN_SIMD_ALIGN DNN_ALIGN(64)

	constexpr auto UseInplace = true;

	typedef float Float;
	typedef std::size_t UInt;
	typedef unsigned char Byte;

	auto GetThreads(const UInt elements, const Float weight = 1) NOEXCEPT
	{
		const auto ULTRALIGHT_THRESHOLD =   2097152ull;	// minimum threshold for ULTRALIGHT load
		const auto LIGHT_THRESHOLD =        8338608ull;
		const auto MEDIUM_THRESHOLD =      68338608ull;
		const auto HEAVY_THRESHOLD =      120338608ull;
		const auto MAXIMUM_THRESHOLD =    187538608ull;

		const auto MAXIMUM = omp_get_max_threads();
		const auto ULTRALIGHT = MAXIMUM >= 32 ?  2ull : MAXIMUM >= 24 ?  2ull :  2ull;
		const auto LIGHT      = MAXIMUM >= 32 ?  4ull : MAXIMUM >= 24 ?  4ull :  4ull;
		const auto MEDIUM     = MAXIMUM >= 32 ? 8ull : MAXIMUM >= 24 ?  8ull :  8ull;
		const auto HEAVY      = MAXIMUM >= 32 ? 16ull : MAXIMUM >= 24 ? 12ll : 12ull;
		const auto ULTRAHEAVY = MAXIMUM >= 32 ? 24ull : MAXIMUM >= 24 ? 24ull : 16ull;

		return
			elements < static_cast<UInt>(weight * Float(ULTRALIGHT_THRESHOLD)) ? ULTRALIGHT :
			elements < static_cast<UInt>(weight * Float(LIGHT_THRESHOLD)) ? LIGHT :
			elements < static_cast<UInt>(weight * Float(MEDIUM_THRESHOLD)) ? MEDIUM :
			elements < static_cast<UInt>(weight * Float(HEAVY_THRESHOLD)) ? HEAVY :
			elements < static_cast<UInt>(weight * Float(MAXIMUM_THRESHOLD)) ? ULTRAHEAVY : MAXIMUM;
	}
	// weight > 1   less threads   
	// weight < 1   more threads

	struct LabelInfo
	{
		UInt LabelA;
		UInt LabelB;
		Float Lambda;	
	};

#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
	typedef Vec16f VecFloat;
	typedef Vec16fb VecFloatBool;
	constexpr auto VectorSize = 16ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw16c;
#elif defined(DNN_AVX2) || defined(DNN_AVX)
	typedef Vec8f VecFloat;
	typedef Vec8fb VecFloatBool;
	constexpr auto VectorSize = 8ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw8c;
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
	typedef Vec4f VecFloat;
	typedef Vec4fb VecFloatBool;
	constexpr auto VectorSize = 4ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw4c;
#endif
	
	

	constexpr auto GetVectorPart(const UInt& elements) NOEXCEPT { return (elements / VectorSize) * VectorSize; }
	constexpr auto DivUp(const UInt& c) NOEXCEPT { return (((c - 1) / VectorSize) + 1) * VectorSize; }
	auto IsPlainDataFmt(const dnnl::memory::desc& md) NOEXCEPT { return md.get_format_kind() == dnnl::memory::format_kind::blocked && md.get_inner_nblks() == 0; }
	auto IsBlockedDataFmt(const dnnl::memory::desc& md) NOEXCEPT { return md.get_format_kind() == dnnl::memory::format_kind::blocked && md.get_inner_nblks() == 1 && md.get_inner_idxs()[0] == 1 && (md.get_inner_blks()[0] == 4 || md.get_inner_blks()[0] == 8 || md.get_inner_blks()[0] == 16); }
	constexpr auto PlainFmt = dnnl::memory::format_tag::nchw; // equals dnnl::memory::format_tag::abcd
	auto GetDataFmt(const dnnl::memory::desc& md)
	{
		if (md.get_format_kind() == dnnl::memory::format_kind::blocked)
		{
			switch (md.get_ndims())
			{
			case 1:
				if (md.get_inner_nblks() == 0)
					return dnnl::memory::format_tag::a;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			case 2:
				if (md.get_inner_nblks() == 0)
					return dnnl::memory::format_tag::ab;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			case 3:
				if (md.get_inner_nblks() == 0)
					return dnnl::memory::format_tag::abc;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			case 4:
				if (md.get_inner_nblks() == 0)
				{
					if (md.get_strides()[1] == 1)
						return dnnl::memory::format_tag::acdb;
					else
						if (md.get_strides()[3] == 1)
							return dnnl::memory::format_tag::abcd;

					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				}
				else
				{
					if (md.get_inner_nblks() == 1 && md.get_inner_idxs()[0] == 1)
					{
						switch (md.get_inner_blks()[0])
						{
						case 4:
							return dnnl::memory::format_tag::nChw4c;
						case 8:
							return dnnl::memory::format_tag::nChw8c;
						case 16:
							return dnnl::memory::format_tag::nChw16c;
						default:
							throw std::invalid_argument("Unsupported format in GetDataFmt function");
						}
					}
					else
						throw std::invalid_argument("Unsupported format in GetDataFmt function");
				}
				break;
			case 5:
				if (md.get_inner_nblks() == 0)
					return dnnl::memory::format_tag::abcde;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			default:
				throw std::invalid_argument("Unsupported number of dimensions in GetDataFmt function");
			}
		}

		return dnnl::memory::format_tag::undef;
	}

	template<typename T>
	void InitArray(T* destination, const std::size_t elements, const int initValue = 0) NOEXCEPT
	{
		if (elements < 1048576ull)
			::memset(destination, initValue, elements * sizeof(T));
		else
		{
			const auto threads = GetThreads(elements);
			const auto part = elements / threads;
			for_i(threads, [=](const std::size_t thread) { ::memset(destination + part * thread, initValue, part * sizeof(T)); });
			if (elements % threads != 0)
				::memset(destination + part * threads, initValue, (elements - part * threads) * sizeof(T));
		}
	}

	struct aligned_free
	{
		void operator()(void* p)
		{
#if defined(_WIN32) || defined(__CYGWIN__)
			::_aligned_free(p);
#elif defined(__MINGW32__)
			::_mm_free(p);
#else
			::free(p);
#endif
		}
	};
	
	template<typename T>
	T* aligned_malloc(std::size_t size, std::size_t alignment) 
	{ 
#if defined(_WIN32) || defined(__CYGWIN__)
		return static_cast<T*>(::_aligned_malloc(size * sizeof(T), alignment));
#elif defined(__ANDROID__)
		return static_cast<T*>(::memalign(size * sizeof(T), alignment));
#elif defined(__MINGW32__)
		return  static_cast<T*>(::_mm_malloc(size * sizeof(T), alignment));
#else  // posix assumed
		return static_cast<T*>(::aligned_alloc(alignment, size * sizeof(T)));
#endif
	}

	template<class T> using unique_ptr_aligned = std::unique_ptr<T, aligned_free>;

	template<class T, std::size_t alignment> 
	unique_ptr_aligned<T> aligned_unique_ptr(std::size_t size, std::size_t align) { return unique_ptr_aligned<T>(static_cast<T*>(aligned_malloc<T>(size, align))); }

	template <typename T, std::size_t alignment> class AlignedArray
	{
		typedef typename std::size_t size_type;

	protected:
		unique_ptr_aligned<T> arrPtr = nullptr;
		T* dataPtr = nullptr;
		size_type nelems = 0;

	public:
		inline void release() NOEXCEPT
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;
		}
		AlignedArray() NOEXCEPT
		{
			AlignedArray::release();
		}
		AlignedArray(const size_type elements, const T value = T()) NOEXCEPT
		{
			AlignedArray::release();

			arrPtr = aligned_unique_ptr<T, alignment>(elements, alignment);
			if (arrPtr)
			{
				dataPtr = arrPtr.get();
				nelems = elements;

				if constexpr (std::is_floating_point_v<T>)
				{
					if constexpr (value == T(0))
						InitArray<T>(dataPtr, nelems, 0);
					else
						PRAGMA_OMP_SIMD()
						for (auto i = 0ull; i < nelems; i++)
							dataPtr[i] = value;
				}
				else
					for (auto i = 0ull; i < nelems; i++)
						dataPtr[i] = value;
			}
		}
		inline T* data() noexcept { return dataPtr; }
		inline const T* data() const noexcept { return dataPtr; }
		inline size_type size() const noexcept { return nelems; }
		inline void resize(size_type elements, const T value = T()) NOEXCEPT
		{ 
			if (elements == nelems)
				return;

			AlignedArray::release();
			
			if (elements > 0)
			{
				arrPtr = aligned_unique_ptr<T, alignment>(elements, alignment);
				if (arrPtr)
				{
					dataPtr = arrPtr.get();
					nelems = elements;
					if constexpr (std::is_floating_point_v<T>)
					{
						if constexpr (value == T(0))
							InitArray<T>(dataPtr, nelems, 0);
						else
							PRAGMA_OMP_SIMD()
							for (auto i = 0ull; i < nelems; i++)
								dataPtr[i] = value;
					}
					else
						for (auto i = 0ull; i < nelems; i++)
							dataPtr[i] = value;
				}
			}		
		}
		inline T& operator[] (size_type i) NOEXCEPT { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const NOEXCEPT { return dataPtr[i]; }
		inline bool empty() const noexcept { return nelems == 0; }
	};

	template <typename T> class AlignedMemory
	{
		typedef typename std::size_t size_type;

	protected:
		std::unique_ptr<dnnl::memory> arrPtr = nullptr;
		T* dataPtr = nullptr;
		size_type nelems = 0;
		dnnl::memory::desc description;

	public:
		inline void release() NOEXCEPT
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;			
		}
		AlignedMemory() NOEXCEPT
		{
			AlignedMemory::release();
		}
		AlignedMemory(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			if (md)
			{
				AlignedMemory::release();

				arrPtr = std::make_unique<dnnl::memory>(md, engine);
				if (arrPtr)
				{
					dataPtr = static_cast<T*>(arrPtr->get_data_handle());
					nelems = md.get_size() / sizeof(T);

					if constexpr (std::is_floating_point_v<T>)
					{
						if constexpr (value == T(0))
							InitArray<T>(dataPtr, nelems, 0);
						else
							PRAGMA_OMP_SIMD()
							for (auto i = 0ull; i < nelems; i++)
								dataPtr[i] = value;
					}
					else
						for (auto i = 0ull; i < nelems; i++)
							dataPtr[i] = value;

					description = md;
				}
			}
		}
		inline T* data() noexcept { return dataPtr; }
		inline const T* data() const noexcept { return dataPtr; }
		inline size_type size() const noexcept { return nelems; }
		inline dnnl::memory::desc desc() { return description; }
		inline void resizeMem(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			if (md)
			{
				if (md.get_size() / sizeof(T) == nelems)
					return;

				AlignedMemory::release();

				if (md.get_size() / sizeof(T) > 0)
				{
					arrPtr = std::make_unique<dnnl::memory>(md, engine);
					if (arrPtr)
					{
						dataPtr = static_cast<T*>(arrPtr->get_data_handle());
						nelems = md.get_size() / sizeof(T);
						if constexpr (std::is_floating_point_v<T>)
						{
							if (value == T(0))
								InitArray<T>(dataPtr, nelems, 0);
							else
								PRAGMA_OMP_SIMD()
								for (auto i = 0ull; i < nelems; i++)
									dataPtr[i] = value;
						}
						else
							for (auto i = 0ull; i < nelems; i++)
								dataPtr[i] = value;

						description = md;
					}
				}
			}
		}
		void resize(const size_type n, const size_type c, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type h, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(h), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type d, const size_type h, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T()) NOEXCEPT
		{
			AlignedMemory::resizeMem(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(d), dnnl::memory::dim(h), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		inline T& operator[] (size_type i) NOEXCEPT { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const NOEXCEPT { return dataPtr[i]; }
		inline bool empty() const noexcept { return nelems == 0; }
	};

	typedef AlignedMemory<Float> FloatArray;
	typedef AlignedArray<Byte, 64ull> ByteArray;
	typedef std::vector<Float, AlignedAllocator<Float, 64ull>> FloatVector;
	//constexpr bool IS_LITTLE_ENDIAN = std::endian::native == std::endian::little;
	constexpr auto NEURONS_LIMIT = Float(5000);	// limit for all the neurons and derivative [-NEURONS_LIMIT,NEURONS_LIMIT]
	constexpr auto WEIGHTS_LIMIT = Float(500);	// limit for all the weights and biases [-WEIGHTS_LIMIT,WEIGHTS_LIMIT]
	
	template<typename T>
	constexpr auto inline Square(const T& value) NOEXCEPT { return (value * value); }
	template<typename T>
	constexpr auto inline Clamp(const T& v, const T& lo, const T& hi) NOEXCEPT { return (v < lo) ? lo : (hi < v) ? hi : v; }
	template<typename T>
	constexpr auto inline Saturate(const T& value) NOEXCEPT { return (value > T(255)) ? Byte(255) : (value < T(0)) ? Byte(0) : Byte(value); }
	template<typename T>
	constexpr auto inline GetColorFromRange(const T& range, const T& minimum, const T& value) NOEXCEPT { return Saturate<T>(T(255) - ((value - minimum) * range)); }
	template<typename T>
	constexpr auto inline GetColorRange(const T& min, const T& max) NOEXCEPT { return (min == max) ? T(0) : T(255) / ((std::signbit(min) && std::signbit(max)) ? -(min + max) : (max - min)); }
	
	/* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
	template<typename T>
	inline void KahanSum(const T& value, T& sum, T& correction) NOEXCEPT
	{
		const auto y = value - correction;
		const auto t = sum + y;
		correction = (t - sum) - y;
		sum = t;
	}

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
	const auto nwl = std::string("\r\n");
#elif defined(__APPLE__)
	const auto nwl = std::string("\r");
#else // assuming Linux
	const auto nwl = std::string("\n");
#endif
	const auto tab = std::string("\t");
	const auto dtab = std::string("\t\t");	
	
#ifdef DNN_FAST_SEED
	template<typename T>
	inline T Seed() NOEXCEPT
	{
		return static_cast<T>(__rdtsc());
	}
#else
	int GetPhysicalSeedType() NOEXCEPT
	{
		int abcd[4];						// return values from cpuid instruction
		
		cpuid(abcd, 7);						// call cpuid function 7
		if (abcd[1] & (1 << 18)) 
			return 3; // ebx bit 18: RDSEED available
		cpuid(abcd, 1);						// call cpuid function 1
		if (abcd[2] & (1 << 30)) 
			return 2; // ecx bit 30: RDRAND available
		if (abcd[3] & (1 << 4)) 
			return 1; // edx bit  4: RDTSC available

		return 0;
	}
	
	int PhysicalSeedType = -1;
	template<typename T>
	inline T Seed() NOEXCEPT
	{
		if (PhysicalSeedType < 0)
			PhysicalSeedType = GetPhysicalSeedType();
		
		uint32_t ran = 0;					// random number
		switch (PhysicalSeedType) 
		{
		case 1:								// use RDTSC instruction
			ran = static_cast<uint32_t>(__rdtsc());
			break;
		case 2:								// use RDRAND instruction
			while (_rdrand32_step(&ran) == 0)
			{ }
			break;
		case 3:								// use RDSEED instruction */
			while (_rdseed32_step(&ran) == 0)
			{ }
			break;
		}
		
		return static_cast<T>(ran);			// return random number
	}
#endif

	inline static auto BernoulliVecFloat(const Float prob = Float(0.5)) noexcept
	{
		static thread_local auto generator = Ranvec1(3, Seed<int>(), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())));
#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
		return select(generator.random16f() < prob, VecFloat(1), VecFloat(0));
#elif defined(DNN_AVX2) || defined(DNN_AVX)
		return select(generator.random8f() < prob, VecFloat(1), VecFloat(0));
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
		return select(generator.random4f() < prob, VecFloat(1), VecFloat(0));
#endif
	}

	template<typename T>
	inline auto Bernoulli(const Float p = Float(0.5)) NOEXCEPT
	{
#ifndef NDEBUG
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in Bernoulli function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return static_cast<T>(std::bernoulli_distribution(static_cast<double>(p))(generator));
	}
	
	template<typename T>
	inline auto Bernoulli(std::mt19937& generator, const Float p = Float(0.5)) NOEXCEPT
	{
#ifndef NDEBUG
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in Bernoulli function");
#endif
		return static_cast<T>(std::bernoulli_distribution(static_cast<double>(p))(generator));
	}

	template<typename T>
	inline auto UniformInt(std::mt19937& generator, const T min, const T max) NOEXCEPT
	{
		static_assert(std::is_integral<T>::value, "Only integral type supported in UniformInt function");
#ifndef NDEBUG
		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformInt function");
#endif
		return std::uniform_int_distribution<T>(min, max)(generator);
	}

	template<typename T>
	inline auto UniformReal(std::mt19937& generator, const T min, const T max) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in UniformReal function");
#ifndef NDEBUG
		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformReal function");
#endif
		return std::uniform_real_distribution<T>(min, max)(generator);
	}

	template<typename T>
	auto TruncatedNormal(std::mt19937& generator, const T m, const T s, const T limit) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in TruncatedNormal function");
#ifndef NDEBUG
		if (limit < s)
	     throw std::invalid_argument("limit out of range in TruncatedNormal function");
#endif
		T x;
		do { x = std::normal_distribution<T>(T(0), s)(generator); }
		while (std::abs(x) > limit); // reject if beyond limit
		
		return x + m;
	}

	// from Stack Overflow https://stackoverflow.com/questions/15165202/random-number-generator-with-beta-distribution
	template <typename RealType = double>
	class beta_distribution
	{
	public:
		typedef RealType result_type;

		class param_type
		{
		public:
			typedef beta_distribution distribution_type;

			explicit param_type(RealType a = 2.0, RealType b = 2.0) : a_param(a), b_param(b) { }

			RealType a() const noexcept { return a_param; }
			RealType b() const noexcept { return b_param; }

			bool operator==(const param_type& other) const noexcept
			{
				return (a_param == other.a_param && b_param == other.b_param);
			}

			bool operator!=(const param_type& other) const noexcept
			{
				return !(*this == other);
			}

		private:
			RealType a_param, b_param;
		};

		explicit beta_distribution(RealType a = 2.0, RealType b = 2.0) noexcept  : a_gamma(a), b_gamma(b) { }
		explicit beta_distribution(const param_type& param) noexcept : a_gamma(param.a()), b_gamma(param.b()) { }

		void reset() { }

		param_type param() const noexcept
		{
			return param_type(a(), b());
		}

		void param(const param_type& param) noexcept
		{
			a_gamma = gamma_dist_type(param.a());
			b_gamma = gamma_dist_type(param.b());
		}

		template <typename URNG>
		inline result_type operator()(URNG& engine) noexcept
		{
			return generate(engine, a_gamma, b_gamma);
		}

		template <typename URNG>
		inline result_type operator()(URNG& engine, const param_type& param) noexcept
		{
			gamma_dist_type a_param_gamma(param.a()), b_param_gamma(param.b());
			return generate(engine, a_param_gamma, b_param_gamma);
		}

		result_type min() const noexcept { return 0.0; }
		result_type max() const noexcept { return 1.0; }

		result_type a() const noexcept { return a_gamma.alpha(); }
		result_type b() const noexcept { return b_gamma.alpha(); }

		bool operator==(const beta_distribution<result_type>& other) const noexcept
		{
			return (param() == other.param() &&	a_gamma == other.a_gamma &&	b_gamma == other.b_gamma);
		}

		bool operator!=(const beta_distribution<result_type>& other) const noexcept
		{
			return !(*this == other);
		}

	private:
		typedef std::gamma_distribution<result_type> gamma_dist_type;

		gamma_dist_type a_gamma, b_gamma;

		template <typename URNG>
		inline result_type generate(URNG& engine, gamma_dist_type& x_gamma, gamma_dist_type& y_gamma) noexcept
		{
			result_type x = x_gamma(engine);
			return x / (x + y_gamma(engine));
		}
	};

	template<typename T>
	inline auto BetaDistribution(std::mt19937& generator, const T a, const T b) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in BetaDistribution function");

		return beta_distribution<T>(a, b)(generator);
	}

	auto FloatToString(const Float value, const std::streamsize precision = 8)
	{
		std::stringstream stream; 
		stream << std::setprecision(precision) << value;
		return stream.str();
	}

	auto FloatToStringFixed(const Float value, const std::streamsize precision = 8)
	{
		std::stringstream stream; 
		stream << std::setprecision(precision) << std::fixed << value;
		return stream.str();
	}

	auto FloatToStringScientific(const Float value, const std::streamsize precision = 4)
	{
		std::stringstream stream; 
		stream << std::setprecision(precision) << std::scientific << value;
		return stream.str();
	}

   	auto GetFileSize(std::string fileName)
	{
		auto file = std::ifstream(fileName, std::ifstream::in | std::ifstream::binary);

		if (!file.is_open() || file.bad())
			return std::streamsize(-1);

		file.seekg(0, std::ios::beg);
		const auto start = file.tellg();
		file.seekg(0, std::ios::end);
		const auto end = file.tellg();
		file.close();
		
		return static_cast<std::streamsize>(end - start);
	}

	auto StringToLower(std::string text)
	{
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		return text;
	}

	auto IsStringBool(std::string text)
	{
		const auto textLower = StringToLower(text);
		
		if (textLower == "true" || textLower == "yes" || textLower == "false" || textLower == "no")
			return true;

		return false;
	}

	auto StringToBool(std::string text)
	{
		const auto textLower = StringToLower(text);
		
		if (textLower == "true" || textLower == "yes")
			return true;

		return false;
	}

	auto GetTotalFreeMemory()
	{
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
		MEMORYSTATUSEX statusEx;
		statusEx.dwLength = sizeof(MEMORYSTATUSEX);
		GlobalMemoryStatusEx(&statusEx);
		std::cout << std::string("Available memory: ") << std::to_string(statusEx.ullAvailPhys/1024/1024) << std::string("/") << std::to_string(statusEx.ullTotalPhys/1024/1024) << " MB" << std::endl;
		return statusEx.ullAvailPhys;
#else        
		struct sysinfo info;
		if (sysinfo(&info) == 0)
		{
			std::cout << std::string("Available memory: ") << std::to_string(info.freeram*info.mem_unit/1024/1024) << std::string("/") << std::to_string(info.totalram*info.mem_unit/1024/1024) << " MB" << std::endl;
			return static_cast<UInt>(info.freeram * info.mem_unit);
		}
		else
			return static_cast<UInt>(0);
#endif
	}
	
	auto CaseInsensitiveReplace(std::string::const_iterator begin, std::string::const_iterator end, const std::string& before, const std::string& after)
	{
		auto retval = std::string("");
		auto dest = std::back_insert_iterator<std::string>(retval);
		auto current = begin;
		auto next = std::search(current, end, before.begin(), before.end(), [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); });

		while (next != end)
		{
			std::copy(current, next, dest);
			std::copy(after.begin(), after.end(), dest);
			current = next + before.size();
			next = std::search(current, end, before.begin(), before.end(), [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); });
		}

		std::copy(current, next, dest);

		return retval;
	}

	// From Stack Overflow https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
	auto Trim(std::string text)
	{
		text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](int ch) { return !std::isspace(ch); }));
		text.erase(std::find_if(text.rbegin(), text.rend(), [](int ch) { return !std::isspace(ch); }).base(), text.end());
		return text;
	}

	// from Stack Overflow https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
	auto& SafeGetline(std::istream& is, std::string& line)
	{
		line.clear();

		// The characters in the stream are read one-by-one using a std::streambuf.
		// That is faster than reading them one-by-one using the std::istream.
		// Code that uses streambuf this way must be guarded by a sentry object.
		// The sentry object performs various tasks,
		// such as thread synchronization and updating the stream state.

		std::istream::sentry se(is, true);
		auto sb = is.rdbuf();

		for (;;) 
		{
			auto c = sb->sbumpc();
			switch (c) 
			{
			case '\n':
				return is;
			case '\r':
				if (sb->sgetc() == '\n')
					sb->sbumpc();
				return is;
			case std::streambuf::traits_type::eof():
				// Also handle the case when the last line has no line ending
				if (line.empty())
					is.setstate(std::ios::eofbit);
				return is;
			default:
				line += static_cast<char>(c);
			}
		}
	}

	template <typename T>
	constexpr void SwapEndian(T& buffer) NOEXCEPT
	{
		static_assert(std::is_standard_layout<T>::value, "SwapEndian supports standard layout types only");
		auto startIndex = static_cast<char*>((void*)buffer.data());
		auto endIndex = startIndex + sizeof(buffer);
		std::reverse(startIndex, endIndex);
	}
}