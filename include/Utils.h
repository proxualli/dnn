#pragma once
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
#include "stdafx.h"
#else
#include <sys/sysinfo.h>
#endif

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
#include <unordered_map>
#include <vector>
#include <utility>


#ifndef MAX_VECTOR_SIZE
#ifdef DNN_SSE41
#define INSTRSET 5
#define MAX_VECTOR_SIZE 128
#endif
#ifdef DNN_AVX2
#define INSTRSET 8
#define MAX_VECTOR_SIZE 256
#endif
#ifdef DNN_AVX512
#define INSTRSET 9
#define MAX_VECTOR_SIZE 512
#endif
#endif

#include "vectorclass.h"
#include "vectormath_hyp.h"
#include "vectormath_exp.h"
#include "vectormath_trig.h"
#include "add-on/random/ranvec1.h"

#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 255

#include "magic_enum.hpp"

#ifdef DNN_OMP
#include <omp.h>
#endif

#include "dnnl.hpp"
#include "dnnl_debug.h"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)
#define CHAIn2(a, b) a b
#define CHAIN2(a, b) CHAIn2(a, b)

#ifdef _MSC_VER
#define PRAGMA_MACRo(x) __pragma(x)
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#else
#define PRAGMA_MACRo(x) _Pragma(#x)
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n) PRAGMA_MACRO(omp parallel for collapse(n))
#define PRAGMA_OMP_PARALLEL_THREADS(n) PRAGMA_MACRO(omp parallel num_threads(n))
#define PRAGMA_OMP_FOR_SCHEDULE_STATIC(n) PRAGMA_MACRO(omp for schedule(static,n))
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#define collapse(x)
#define PRAGMA_OMP(...)
#define PRAGMA_OMP_SIMD(...)
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)
#define PRAGMA_OMP_PARALLEL_THREADS(n)
#define PRAGMA_OMP_FOR_SCHEDULE_STATIC(n)
#define OMP_GET_THREAD_NUM() 0
#define OMP_GET_NUM_THREADS() 1
#endif

#if (defined(__clang_major__) \
        && (__clang_major__ < 3 \
                || (__clang_major__ == 3 && __clang_minor__ < 9))) \
        || (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
        || (!defined(__INTEL_COMPILER) && !defined(__clang__) \
                && (defined(_MSC_VER) || __GNUC__ < 6 \
                        || (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif // long simdlen if

#include "AlignedAllocator.h"
#include "ParallelFor.h"

namespace dnn
{
#ifdef _MSC_VER
	#define DNN_ALIGN(alignment) __declspec(align(alignment))
#else
	#define DNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#endif
	#define DNN_SIMD_ALIGN DNN_ALIGN(64)

	typedef float Float;
	typedef unsigned char Byte;
	typedef std::vector<Float, AlignedAllocator<Float, 64ull>> FloatVector;
	typedef std::vector<Byte, AlignedAllocator<Byte, 64ull>> ByteVector;

    //constexpr bool IS_LITTLE_ENDIAN = std::endian::native == std::endian::little;
	constexpr auto NEURONS_LIMIT = Float(1000);   // limit for all the value of the neurons and its derivatives [-NEURONS_LIMIT,NEURONS_LIMIT]
	constexpr auto WEIGHTS_LIMIT = Float(100);   // limit for all the value of the weights and biases [-WEIGHTS_LIMIT,WEIGHTS_LIMIT]
	constexpr auto LIGHT_COMPUTE = 4ull;         // number of threads
	constexpr auto MEDIUM_COMPUTE = 8ull;
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
	
	constexpr auto PlainFmt = dnnl::memory::format_tag::nchw;

#if defined(DNN_AVX512)
	typedef Vec16f VecFloat;
	typedef Vec16fb VecFloatBool;
	constexpr auto VectorSize = 16ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw16c;
#elif defined(DNN_AVX2)
	typedef Vec8f VecFloat;
	typedef Vec8fb VecFloatBool;
	constexpr auto VectorSize = 8ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw8c;
#elif defined(DNN_SSE41)
	typedef Vec4f VecFloat;
	typedef Vec4fb VecFloatBool;
	constexpr auto VectorSize = 4ull;
	constexpr auto BlockedFmt = dnnl::memory::format_tag::nChw4c;
#endif

	constexpr auto DivUp(const size_t& c) noexcept { return (((c - 1) / VectorSize) + 1) * VectorSize; }
	constexpr auto IsPlainDataFmt(const dnnl::memory::desc& md) noexcept { return md.data.format_kind == dnnl_blocked && md.data.format_desc.blocking.inner_nblks == 0; }
	constexpr auto IsBlockedDataFmt(const dnnl::memory::desc& md) noexcept { return md.data.format_kind == dnnl_blocked && md.data.format_desc.blocking.inner_nblks == 1 && md.data.format_desc.blocking.inner_idxs[0] == 1 && (md.data.format_desc.blocking.inner_blks[0] == 4 || md.data.format_desc.blocking.inner_blks[0] == 8 || md.data.format_desc.blocking.inner_blks[0] == 16); }

	constexpr auto GetDataFmt(const dnnl::memory::desc& md) noexcept
	{
		if (md.data.format_kind == dnnl_blocked)
		{
			if (md.data.format_desc.blocking.inner_nblks == 0)
				return PlainFmt;
			else 
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
						return dnnl::memory::format_tag::undef;
					}
				}
		}

		return dnnl::memory::format_tag::undef;
	}

	inline static void ZeroFloatVector(Float* destination, const size_t elements) noexcept
	{
		if (elements < 1048576ull)
			::memset(destination, 0, elements * sizeof(Float));
		else
		{
			const auto threads = elements < 2097152ull ? 2ull : elements < 8338608ull ? LIGHT_COMPUTE : MEDIUM_COMPUTE;
			const auto part = elements / threads;
			for_i(threads, [=](const size_t thread) { ::memset(destination + part * thread, 0, part * sizeof(Float)); });
			if (elements % threads != 0)
				::memset(destination + part * threads, 0, (elements - part * threads) * sizeof(Float));
		}
	}

	inline static void ZeroFloatVectorAllocate(FloatVector& destination, const size_t elements) noexcept
	{
		if (destination.size() < elements)
			destination = FloatVector(elements);

		ZeroFloatVector(destination.data(), elements);
	}
	
#ifdef _MSC_VER
#pragma intrinsic(__rdtsc)
#endif
	inline static auto BernoulliVecFloat(const Float prob = Float(0.5)) noexcept
	{
		static thread_local auto generator = Ranvec1(3);

		generator.init(static_cast<int>(__rdtsc()), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())));
#if defined(DNN_AVX512)
		return select(generator.random16f() < prob, VecFloat(1), VecFloat(0));
#elif defined(DNN_AVX2)
		return select(generator.random8f() < prob, VecFloat(1), VecFloat(0));
#elif defined(DNN_SSE41)
		return select(generator.random4f() < prob, VecFloat(1), VecFloat(0));
#endif
	}

#ifdef _MSC_VER
#pragma intrinsic(__rdtsc)
#endif
	template<typename T>
	static auto Bernoulli(const Float prob = Float(0.5)) noexcept
	{
		static thread_local auto generator = std::mt19937(static_cast<unsigned>(__rdtsc()));
		return static_cast<T>(std::bernoulli_distribution(double(prob))(generator));
	}

#ifdef _MSC_VER
#pragma intrinsic(__rdtsc)
#endif
	template<typename T>
	static auto UniformInt(const T min, const T max) noexcept
	{
		static thread_local auto generator = std::mt19937(static_cast<unsigned>(__rdtsc()));
		return std::uniform_int_distribution<T>(min, max)(generator);
	}

#ifdef _MSC_VER
#pragma intrinsic(__rdtsc)
#endif
	template<typename T>
	static auto UniformReal(const T min, const T max) noexcept
	{
		static thread_local auto generator = std::mt19937(static_cast<unsigned>(__rdtsc()));
		return std::uniform_real_distribution<T>(min, max)(generator);
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

	inline static auto StringToLower(std::string text)
	{
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		return text;
	};

	static auto IsStringBool(std::string text)
	{
		auto textLower = StringToLower(text);
		
		if (textLower == "true" || textLower == "yes" || textLower == "false" || textLower == "no")
			return true;

		return false;
	}

	static auto StringToBool(std::string text)
	{
		auto textLower = StringToLower(text);
		
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
			return static_cast<size_t>(info.freeram * info.mem_unit);
		}
		else
			return static_cast<size_t>(0);
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

		auto se = std::istream::sentry(is, true);
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
}
