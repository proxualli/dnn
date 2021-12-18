#pragma once
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
#include "stdafx.h"
#else
#include <sys/sysinfo.h>
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

namespace dnn
{
#ifdef _MSC_VER
#define DNN_ALIGN(alignment) __declspec(align(alignment))
#else
#define DNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#endif
#define DNN_SIMD_ALIGN DNN_ALIGN(64)

	constexpr auto UseInplace = true;

	typedef float Float;
	typedef size_t UInt;
	typedef unsigned char Byte;

	constexpr auto ULTRALIGHT_COMPUTE = 3ull;	// number of threads
	constexpr auto LIGHT_COMPUTE = 6ull;
	constexpr auto MEDIUM_COMPUTE = 12ull;
	constexpr auto HEAVY_COMPUTE = 16ull;
	constexpr auto ULTRALIGHT_COMPUTE_ELEMENTSTHRESHOLD = 2097152ull;	// minimum element threshold for ULTRALIGHT_COMPUTE
	constexpr auto LIGHT_COMPUTE_ELEMENTSTHRESHOLD = 8338608ull;
	constexpr auto MEDIUM_COMPUTE_ELEMENTSTHRESHOLD = 68338608ull;
	constexpr auto GetThreads(const UInt elements, const Float weight = 1) 
	{
		if (weight <= Float(0.01) || weight > Float(100))
			throw std::invalid_argument("Weight is out of range in GetThreads function");
		
		return elements < static_cast<UInt>(weight * ULTRALIGHT_COMPUTE_ELEMENTSTHRESHOLD) ? ULTRALIGHT_COMPUTE : elements < static_cast<UInt>(weight * LIGHT_COMPUTE_ELEMENTSTHRESHOLD) ? LIGHT_COMPUTE : elements < static_cast<UInt>(weight * MEDIUM_COMPUTE_ELEMENTSTHRESHOLD) ? MEDIUM_COMPUTE : HEAVY_COMPUTE;
	}

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
	constexpr auto GetVectorPart(const UInt& elements) { return (elements / VectorSize) * VectorSize; }
	constexpr auto DivUp(const UInt& c) noexcept { return (((c - 1) / VectorSize) + 1) * VectorSize; }
	constexpr auto IsPlainDataFmt(const dnnl::memory::desc& md) noexcept { return md.data.format_kind == dnnl_blocked && md.data.format_desc.blocking.inner_nblks == 0; }
	constexpr auto IsBlockedDataFmt(const dnnl::memory::desc& md) noexcept { return md.data.format_kind == dnnl_blocked && md.data.format_desc.blocking.inner_nblks == 1 && md.data.format_desc.blocking.inner_idxs[0] == 1 && (md.data.format_desc.blocking.inner_blks[0] == 4 || md.data.format_desc.blocking.inner_blks[0] == 8 || md.data.format_desc.blocking.inner_blks[0] == 16); }
	constexpr auto PlainFmt = dnnl::memory::format_tag::nchw; // equals dnnl::memory::format_tag::abcd
	constexpr auto GetDataFmt(const dnnl::memory::desc& md)
	{
		if (md.data.format_kind == dnnl_blocked)
		{
			switch (md.data.ndims)
			{
			case 1:
				if (md.data.format_desc.blocking.inner_nblks == 0)
					return dnnl::memory::format_tag::a;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			case 2:
				if (md.data.format_desc.blocking.inner_nblks == 0)
					return dnnl::memory::format_tag::ab;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			case 3:
				if (md.data.format_desc.blocking.inner_nblks == 0)
					return dnnl::memory::format_tag::abc;
				else
					throw std::invalid_argument("Unsupported format in GetDataFmt function");
				break;
			case 4:
				if (md.data.format_desc.blocking.inner_nblks == 0)
					return dnnl::memory::format_tag::abcd;
				else
				{
					if (md.data.format_desc.blocking.inner_nblks == 1 && md.data.format_desc.blocking.inner_idxs[0] == 1)
					{
						switch (md.data.format_desc.blocking.inner_blks[0])
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
				if (md.data.format_desc.blocking.inner_nblks == 0)
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
	static void InitArray(T* destination, const std::size_t elements, const int initValue = 0) noexcept
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
		void release() noexcept
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;
		}
		AlignedArray()
		{
			release();
		}
		AlignedArray(const size_type elements, const T value = T()) 
		{
			release();

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
		inline void resize(size_type elements, const T value = T())
		{ 
			if (elements == nelems)
				return;

			release();
			
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
		inline T& operator[] (size_type i) noexcept { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const noexcept { return dataPtr[i]; }
		inline bool empty() const noexcept { return nelems == 0; }
	};

	template <typename T> class AlignedMemory
	{
		typedef typename std::size_t size_type;

	protected:
		std::unique_ptr<dnnl::memory> arrPtr = nullptr;
		T* dataPtr = nullptr;
		size_type nelems = 0;
		
	public:
		void release() noexcept
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;
		}
		AlignedMemory()
		{
			release();
		}
		AlignedMemory(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T())
		{
			if (md)
			{
				release();

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
				}
			}
		}
		inline T* data() noexcept { return dataPtr; }
		inline const T* data() const noexcept { return dataPtr; }
		inline size_type size() const noexcept { return nelems; }
		void resize(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T())
		{
			if (md)
			{
				if (md.get_size() / sizeof(T) == nelems)
					return;

				release();

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
					}
				}
			}
		}
		void resize(const size_type n, const size_type c, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T())
		{
			resize(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T())
		{
			resize(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type h, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T())
		{
			resize(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(h), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		void resize(const size_type n, const size_type c, const size_type d, const size_type h, const size_type w, const dnnl::memory::data_type dtype, const dnnl::memory::format_tag format, const dnnl::engine& engine, const T value = T())
		{
			resize(dnnl::memory::desc(dnnl::memory::dims({ dnnl::memory::dim(n), dnnl::memory::dim(c), dnnl::memory::dim(d), dnnl::memory::dim(h), dnnl::memory::dim(w) }), dtype, format), engine, value);
		}
		inline T& operator[] (size_type i) noexcept { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const noexcept { return dataPtr[i]; }
		inline bool empty() const noexcept { return nelems == 0; }
	};

	typedef AlignedMemory<Float> FloatArray;
	typedef AlignedArray<Byte, 64ull> ByteArray;
	typedef std::vector<Float, AlignedAllocator<Float, 64ull>> FloatVector;
	//constexpr bool IS_LITTLE_ENDIAN = std::endian::native == std::endian::little;
	constexpr auto NEURONS_LIMIT = Float(10000);		// limit for all the neurons and derivative [-NEURONS_LIMIT,NEURONS_LIMIT]
	constexpr auto WEIGHTS_LIMIT = Float(100);		// limit for all the weights and biases [-WEIGHTS_LIMIT,WEIGHTS_LIMIT]
	
	constexpr auto FloatSquare(const Float& value) noexcept { return (value * value); }
	template<typename T>
	constexpr auto Clamp(const T& v, const T& lo, const T& hi) noexcept { return (v < lo) ? lo : (hi < v) ? hi : v; }
	template<typename T>
	constexpr auto Saturate(const T& value) noexcept { return (value > T(255)) ? Byte(255) : (value < T(0)) ? Byte(0) : Byte(value); }
	constexpr auto GetColorFromRange(const Float& range, const Float& minimum, const Float& value) noexcept { return Saturate<Float>(Float(255) - ((value - minimum) * range)); }
	constexpr auto GetColorRange(const Float& min, const Float& max) noexcept { return (min == max) ? Float(0) : Float(255) / ((std::signbit(min) && std::signbit(max)) ? -(min + max) : (max - min)); }

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
	static const auto nwl = std::string("\r\n");
#else // assuming Linux
	static const auto nwl = std::string("\n");
#endif
	static const auto tab = std::string("\t");
	static const auto dtab = std::string("\t\t");	
	
#ifdef DNN_FAST_SEED
	template<typename T>
	inline static T Seed() noexcept
	{
		return static_cast<T>(__rdtsc());
	}
#else
	static int GetPhysicalSeedType() noexcept
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
	
	static int PhysicalSeedType = -1;
	template<typename T>
	T Seed() noexcept
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
			while (_rdrand32_step(&ran) == 0) {}
			break;
		case 3:								// use RDSEED instruction */
			while (_rdseed32_step(&ran) == 0) {}
			break;
		}
		
		return static_cast<T>(ran);			// return random number
	}
#endif

	inline static auto BernoulliVecFloat(const Float p = Float(0.5))
	{
		if (p < 0 || p > 1) 
			throw std::invalid_argument("Parameter out of range in BernoulliVecFloat function");

		static thread_local auto generator = Ranvec1(Seed<int>(), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())), 3);
#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
		return select(generator.random16f() < p, VecFloat(1), VecFloat(0));
#elif defined(DNN_AVX2) || defined(DNN_AVX)
		return select(generator.random8f() < p, VecFloat(1), VecFloat(0));
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
		return select(generator.random4f() < p, VecFloat(1), VecFloat(0));
#endif
	}

	template<typename T>
	static auto Bernoulli(const Float p = Float(0.5))
	{
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in Bernoulli function");

		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return static_cast<T>(std::bernoulli_distribution(static_cast<double>(p))(generator));
	}

	template<typename T>
	static auto UniformInt(const T min, const T max)
	{
		static_assert(std::is_integral<T>::value, "Only integral type supported in UniformInt function");

		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformInt function");

		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return std::uniform_int_distribution<T>(min, max)(generator);
	}

	template<typename T>
	static auto UniformReal(const T min, const T max)
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in UniformReal function");

		if(min > max)
			throw std::invalid_argument("Parameter out of range in UniformReal function");

		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return std::uniform_real_distribution<T>(min, max)(generator);
	}

	template<typename T>
	static auto TruncatedNormal(const T m, const T s, const T limit)
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in TruncatedNormal function");

		if (limit < s)
			throw std::invalid_argument("limit out of range in TruncatedNormal function");

		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		
		T x;
		do 
		{
			x = std::normal_distribution<T>(T(0), s)(generator);
		} while (std::abs(x) > limit); // reject if beyond limit
		
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

			RealType a() const { return a_param; }
			RealType b() const { return b_param; }

			bool operator==(const param_type& other) const
			{
				return (a_param == other.a_param && b_param == other.b_param);
			}

			bool operator!=(const param_type& other) const
			{
				return !(*this == other);
			}

		private:
			RealType a_param, b_param;
		};

		explicit beta_distribution(RealType a = 2.0, RealType b = 2.0) : a_gamma(a), b_gamma(b) { }
		explicit beta_distribution(const param_type& param) : a_gamma(param.a()), b_gamma(param.b()) { }

		void reset() { }

		param_type param() const
		{
			return param_type(a(), b());
		}

		void param(const param_type& param)
		{
			a_gamma = gamma_dist_type(param.a());
			b_gamma = gamma_dist_type(param.b());
		}

		template <typename URNG>
		result_type operator()(URNG& engine)
		{
			return generate(engine, a_gamma, b_gamma);
		}

		template <typename URNG>
		result_type operator()(URNG& engine, const param_type& param)
		{
			gamma_dist_type a_param_gamma(param.a()), b_param_gamma(param.b());
			return generate(engine, a_param_gamma, b_param_gamma);
		}

		result_type min() const { return 0.0; }
		result_type max() const { return 1.0; }

		result_type a() const { return a_gamma.alpha(); }
		result_type b() const { return b_gamma.alpha(); }

		bool operator==(const beta_distribution<result_type>& other) const
		{
			return (param() == other.param() &&	a_gamma == other.a_gamma &&	b_gamma == other.b_gamma);
		}

		bool operator!=(const beta_distribution<result_type>& other) const
		{
			return !(*this == other);
		}

	private:
		typedef std::gamma_distribution<result_type> gamma_dist_type;

		gamma_dist_type a_gamma, b_gamma;

		template <typename URNG>
		result_type generate(URNG& engine, gamma_dist_type& x_gamma, gamma_dist_type& y_gamma)
		{
			result_type x = x_gamma(engine);
			return x / (x + y_gamma(engine));
		}
	};

	template<typename T>
	inline static auto BetaDistribution(const T a, const T b) noexcept
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in BetaDistribution function");

		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return beta_distribution<T>(a, b)(generator);
	}

	static auto FloatToString(const Float value, const std::streamsize precision = 8)
	{
		std::stringstream stream; 
		stream << std::setprecision(precision) << value;
		return stream.str();
	}

	static auto FloatToStringFixed(const Float value, const std::streamsize precision = 8)
	{
		std::stringstream stream; 
		stream << std::setprecision(precision) << std::fixed << value;
		return stream.str();
	}

	static auto FloatToStringScientific(const Float value, const std::streamsize precision = 4)
	{
		std::stringstream stream; 
		stream << std::setprecision(precision) << std::scientific << value;
		return stream.str();
	}

   	static auto GetFileSize(std::string fileName)
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

	static auto StringToLower(std::string text)
	{
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		return text;
	}

	static auto IsStringBool(std::string text)
	{
		const auto textLower = StringToLower(text);
		
		if (textLower == "true" || textLower == "yes" || textLower == "false" || textLower == "no")
			return true;

		return false;
	}

	static auto StringToBool(std::string text)
	{
		const auto textLower = StringToLower(text);
		
		if (textLower == "true" || textLower == "yes")
			return true;

		return false;
	}

	static auto GetTotalFreeMemory()
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
	
	static auto CaseInsensitiveReplace(std::string::const_iterator begin, std::string::const_iterator end, const std::string& before, const std::string& after)
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
	static auto Trim(std::string text)
	{
		text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](int ch) { return !std::isspace(ch); }));
		text.erase(std::find_if(text.rbegin(), text.rend(), [](int ch) { return !std::isspace(ch); }).base(), text.end());
		return text;
	}

	// from Stack Overflow https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
	static auto& SafeGetline(std::istream& is, std::string& line)
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
	constexpr void SwapEndian(T& buffer)
	{
		static_assert(std::is_standard_layout<T>::value, "SwapEndian supports standard layout types only");
		auto startIndex = static_cast<char*>((void*)buffer.data());
		auto endIndex = startIndex + sizeof(buffer);
		std::reverse(startIndex, endIndex);
	}

	// from oneDNN sample code
#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

	static dnnl_engine_kind_t validate_engine_kind(dnnl_engine_kind_t akind) {
		// Checking if a GPU exists on the machine
		if (akind == dnnl_gpu) {
			if (!dnnl_engine_get_count(dnnl_gpu)) {
				printf("Application couldn't find GPU, please run with CPU "
					"instead.\n");
				exit(0);
			}
		}
		return akind;
	}

#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

	static inline dnnl_engine_kind_t parse_engine_kind(int argc, char** argv) {
		// Returns default engine kind, i.e. CPU, if none given
		if (argc == 1) {
			return validate_engine_kind(dnnl_cpu);
		}
		else if (argc == 2) {
			// Checking the engine type, i.e. CPU or GPU
			char* engine_kind_str = argv[1];
			if (!strcmp(engine_kind_str, "cpu")) {
				return validate_engine_kind(dnnl_cpu);
			}
			else if (!strcmp(engine_kind_str, "gpu")) {
				return validate_engine_kind(dnnl_gpu);
			}
		}

		// If all above fails, the example should be run properly
		COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
			"inappropriate engine kind.\n"
			"Please run the example like this: %s [cpu|gpu].",
			argv[0]);
		}

	static inline const char* engine_kind2str_upper(dnnl_engine_kind_t kind) {
		if (kind == dnnl_cpu) return "CPU";
		if (kind == dnnl_gpu) return "GPU";
		return "<Unknown engine>";
	}

	// Read from memory, write to handle
	static inline void read_from_dnnl_memory(void* handle, dnnl_memory_t mem) {
		dnnl_engine_t eng;
		dnnl_engine_kind_t eng_kind;
		const dnnl_memory_desc_t* md;

		if (!handle) COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s", "handle is NULL.");

		CHECK(dnnl_memory_get_engine(mem, &eng));
		CHECK(dnnl_engine_get_kind(eng, &eng_kind));
		CHECK(dnnl_memory_get_memory_desc(mem, &md));
		size_t bytes = dnnl_memory_desc_get_size(md);

		bool is_cpu_sycl
			= (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_cpu);

		if (eng_kind == dnnl_gpu || is_cpu_sycl) {
			void* mapped_ptr = NULL;
			CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
			if (mapped_ptr) memcpy(handle, mapped_ptr, bytes);
			CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
		}

		if (eng_kind == dnnl_cpu) {
			void* ptr = NULL;
			CHECK(dnnl_memory_get_data_handle(mem, &ptr));
			if (ptr) {
				for (size_t i = 0; i < bytes; ++i) {
					((char*)handle)[i] = ((char*)ptr)[i];
				}
			}
			return;
		}

		assert(!"not expected");
	}

	// Read from handle, write to memory
	static inline void write_to_dnnl_memory(void* handle, dnnl_memory_t mem) {
		dnnl_engine_t eng;
		dnnl_engine_kind_t eng_kind;
		const dnnl_memory_desc_t* md;

		if (!handle) COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s", "handle is NULL.");

		CHECK(dnnl_memory_get_engine(mem, &eng));
		CHECK(dnnl_engine_get_kind(eng, &eng_kind));
		CHECK(dnnl_memory_get_memory_desc(mem, &md));
		size_t bytes = dnnl_memory_desc_get_size(md);

#ifdef DNNL_WITH_SYCL
		bool is_cpu_sycl
			= (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_cpu);
		bool is_gpu_sycl
			= (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_gpu);
		if (is_cpu_sycl || is_gpu_sycl) {
			void* mapped_ptr = NULL;
			CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
			if (mapped_ptr) {
				for (size_t i = 0; i < bytes; ++i) {
					((char*)mapped_ptr)[i] = ((char*)handle)[i];
				}
			}
			CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
			return;
		}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
		if (eng_kind == dnnl_gpu) {
			void* mapped_ptr = NULL;
			CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
			if (mapped_ptr) memcpy(mapped_ptr, handle, bytes);
			CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
			return;
		}
#endif

		if (eng_kind == dnnl_cpu) {
			void* ptr = NULL;
			CHECK(dnnl_memory_get_data_handle(mem, &ptr));
			if (ptr) {
				for (size_t i = 0; i < bytes; ++i) {
					((char*)ptr)[i] = ((char*)handle)[i];
				}
			}
			return;
		}

		assert(!"not expected");
	}
}