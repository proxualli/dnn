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
#include <cfenv>
#include <cmath>
#include <cstring>
#include <exception>
#include <execution>
#include <filesystem>
#include <functional> 
#include <future>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <locale>
#include <clocale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <utility>

#ifndef NDEBUG
#ifdef _WIN32
#pragma fenv_access (on)
#else
#pragma STDC FENV_ACCESS on
#endif
#endif

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

#include <bitsery/bitsery.h>
#include <bitsery/adapter/stream.h>
#include <bitsery/traits/string.h>
#include <bitsery/traits/vector.h>
//#include <bitsery/ext/std_atomic.h>
//#include <bitsery/ext/growable.h>

#include "CsvFile.h"

using namespace dnn;

namespace
{
	constexpr auto DefaultDatasetMeanStdDev = false;
	constexpr auto Inplace = true;
	constexpr auto Kahan = true;
	constexpr auto Reference = false;
	constexpr auto SingleMeanVariancePass = true;
	constexpr auto TestActivations = false;
	constexpr auto TestBatchNormalization = false;
	constexpr auto TestConcat = false;

	typedef float Float;
	typedef std::size_t UInt;
	typedef unsigned char Byte;
	
	static const auto MAX_THREADS = static_cast<UInt>(omp_get_max_threads());
	auto GetThreads(const UInt elements, const Float weight = Float(1)) NOEXCEPT
	{
		const auto load = static_cast<UInt>(Float(elements) * weight);

		constexpr auto ULTRALIGHT_THRESHOLD =   2097152ull;
		constexpr auto LIGHT_THRESHOLD =        8338608ull;
		constexpr auto MEDIUM_THRESHOLD =      68338608ull;
		constexpr auto HEAVY_THRESHOLD =      120338608ull;
		constexpr auto MAXIMUM_THRESHOLD =    187538608ull;
		
		const auto ULTRALIGHT = 2ull;
		const auto LIGHT      = MAX_THREADS >=  6ull ?  4ull : 2ull;
		const auto MEDIUM     = MAX_THREADS >=  8ull ?  8ull : MAX_THREADS >=  6ull ?  6ull : MAX_THREADS >=  4ull ?  4ull : 2ull;
		const auto HEAVY      = MAX_THREADS >= 32ull ? 16ull : MAX_THREADS >= 24ull ?  16ll : MAX_THREADS >= 16ull ? 16ull : MAX_THREADS >= 12ull ? 12ull : MAX_THREADS >= 8ull ? 8ull : MAX_THREADS >= 6ull ? 6ull : MAX_THREADS >= 4ull ? 4ull : 2ull;
		const auto ULTRAHEAVY = MAX_THREADS >= 32ull ? 32ull : MAX_THREADS >= 24ull ? 24ull : MAX_THREADS >= 16ull ? 16ull : MAX_THREADS >= 12ull ? 12ull : MAX_THREADS >= 8ull ? 8ull : MAX_THREADS >= 6ull ? 6ull : MAX_THREADS >= 4ull ? 4ull : 2ull;

		return
			load < ULTRALIGHT_THRESHOLD ? ULTRALIGHT :
			load < LIGHT_THRESHOLD ?           LIGHT :
			load < MEDIUM_THRESHOLD ?         MEDIUM :
			load < HEAVY_THRESHOLD ?           HEAVY :
			load < MAXIMUM_THRESHOLD ?    ULTRAHEAVY : MAX_THREADS;
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
	inline const auto VecZero = VecFloat(Float(0));

	constexpr auto GetVectorPart(const UInt& elements) NOEXCEPT { return (elements / VectorSize) * VectorSize; }
	constexpr auto DivUp(const UInt& c) NOEXCEPT { if (c == 0ull) return 0ull; else return (((c - 1) / VectorSize) + 1) * VectorSize; }
	auto IsPlainDataFmt(const dnnl::memory::desc& md) NOEXCEPT { return md.get_format_kind() == dnnl::memory::format_kind::blocked && md.get_inner_nblks() == 0; }
	constexpr auto PlainFmt = dnnl::memory::format_tag::nchw; // equals dnnl::memory::format_tag::abcd
	auto GetDataFmt(const dnnl::memory::desc& md) NOEXCEPT
	{
		using format_tag = dnnl::memory::format_tag;
		
		if (!md.is_zero())
		{
			const auto data_type = md.get_data_type();
			const auto dims = md.get_dims();
			const auto format_kind = md.get_format_kind();
			const auto inner_blks = md.get_inner_blks();
			const auto inner_idxs = md.get_inner_idxs();
			const auto inner_nblks = md.get_inner_nblks();
			const auto ndims = md.get_ndims();
			const auto padded_dims = md.get_padded_dims();
			const auto padded_offsets = md.get_padded_offsets();
			const auto strides = md.get_strides();

			if (format_kind == dnnl::memory::format_kind::blocked)
			{
				if (inner_nblks == 0)
				{
					if (ndims == 1)
					{
						return format_tag::a;
					}
					if (ndims == 2)
					{

						if (strides[0] == 1)
							return format_tag::ba;
						else if (strides[1] == 1)
							return format_tag::ab;
					}
					if (ndims == 3)
					{
						if (strides[0] == 1)
							return format_tag::bca;
						else if (strides[1] == 1)
							return format_tag::acb;
						else if (strides[2] == 1)
							return format_tag::abc;
					}
					if (ndims == 4)
					{
						if (strides[0] == 1)
							return format_tag::bcda;
						else if (strides[1] == 1)
							return format_tag::acdb;
						else if (strides[2] == 1)
							return format_tag::abdc;
						else if (strides[3] == 1)
							return format_tag::abcd;
					}
					if (ndims == 5)
					{
						if (strides[0] == 1)
							return format_tag::bcdea;
						else if (strides[1] == 1)
							return format_tag::acdeb;
						else if (strides[2] == 1)
							return format_tag::abdec;
						else if (strides[3] == 1)
							return format_tag::abced;
						else if (strides[4] == 1)
							return format_tag::abcde;
					}
				}
				else
				{
					if (ndims == 2)
					{
						return format_tag::ab;
					}
					if (ndims == 3)
					{
						return format_tag::abc;
					}
					if (ndims == 4)
					{
						if (inner_nblks == 1 && inner_idxs[0] == 0)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::Abcd4a;
							case 8:
								return format_tag::Abcd8a;
							case 16:
								return format_tag::Abcd16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 1 && inner_idxs[0] == 1)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::aBcd4b;
							case 8:
								return format_tag::aBcd8b;
							case 16:
								return format_tag::aBcd16b;
							default:
								return format_tag::undef;
							}
						}
						//else if (inner_nblks == 1 && inner_idxs[0] == 2)
						//{
						//	switch (inner_blks[0])
						//	{
						//	case 4:
						//		return format_tag::abCd4c;
						//	/*case 8:
						//		return format_tag::abCd4a8c;
						//	case 16:
						//		return format_tag::abCd16c;*/
						//	default:
						//		return format_tag::undef;
						//	}
						//}
						else if (inner_nblks == 2 && inner_idxs[0] == 1 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcd4b4a;
							case 8:
								return format_tag::ABcd8b8a;
							case 16:
								return format_tag::ABcd16b16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 2 && inner_idxs[0] == 0 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcd4a4b;
							case 8:
								return format_tag::ABcd8a8b;
							case 16:
								return format_tag::ABcd16a16b;
							default:
								return format_tag::undef;
							}
						}
					}
					if (ndims == 5)
					{
						if (inner_nblks == 1 && inner_idxs[0] == 0)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::Abcde4a;
							case 8:
								return format_tag::Abcde8a;
							case 16:
								return format_tag::Abcde16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 1 && inner_idxs[0] == 1)
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::aBcde4b;
							case 8:
								return format_tag::aBcde8b;
							case 16:
								return format_tag::aBcde16b;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 2 && inner_idxs[0] == 1 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcde4b4a;
							case 8:
								return format_tag::ABcde8b8a;
							case 16:
								return format_tag::ABcde16b16a;
							default:
								return format_tag::undef;
							}
						}
						else if (inner_nblks == 2 && inner_idxs[0] == 0 && (inner_blks[0] == inner_blks[1]))
						{
							switch (inner_blks[0])
							{
							case 4:
								return format_tag::ABcde4a4b;
							case 8:
								return format_tag::ABcde8a8b;
							case 16:
								return format_tag::ABcde16a16b;
							default:
								return format_tag::undef;
							}
						}

						/*

						/// 5D tensor blocked by 1st dimension with block size 16
						dnnl_ABcde4b16a4b,
						/// 5D tensor blocked by 1st dimension with block size 8
						dnnl_ABcde2b8a4b,
						/// 5D tensor blocked by 2nd dimension with block size 16
						dnnl_aBcde16b,
						dnnl_ABcde16b16a,
						dnnl_aBCde16b16c,
						dnnl_aBCde16c16b,
						dnnl_aBCde2c8b4c,
						dnnl_Abcde4a,
						/// 5D tensor blocked by 2nd dimension with block size 32
						dnnl_aBcde32b,
						/// 5D tensor blocked by 2nd dimension with block size 4
						dnnl_aBcde4b,
						dnnl_ABcde4b4a,
						dnnl_ABcde4a4b,
						dnnl_aBCde4b4c,
						dnnl_aBCde2c4b2c,
						dnnl_aBCde4b8c2b,
						dnnl_aBCde4c16b4c,
						dnnl_aBCde16c16b4c,
						dnnl_aBCde16c16b2c,
						dnnl_aBCde4c4b,
						dnnl_Abcde8a,
						dnnl_ABcde8a8b,
						dnnl_ABcde8a4b,
						dnnl_BAcde16b16a,
						/// 5D tensor blocked by 2nd dimension with block size 8
						dnnl_aBcde8b,
						dnnl_ABcde8b16a2b,
						dnnl_aBCde8b16c2b,
						dnnl_aBCde4c8b2c,
						dnnl_aCBde8b16c2b,
						dnnl_ABcde8b8a,
						dnnl_ABcde32a32b,
						dnnl_aBCde8b8c,
						dnnl_aBCde8b4c,
						dnnl_ABc4a8b8a4b,
						dnnl_ABcd4a8b8a4b,
						dnnl_ABcde4a8b8a4b,
						dnnl_BAc4b8a8b4a,
						dnnl_BAcd4b8a8b4a,
						dnnl_BAcde4b8a8b4a,
						dnnl_ABcd2a8b8a2b,
						dnnl_aBCd4b8c8b4c,
						dnnl_aBCde4b8c8b4c,
						dnnl_aBCde2b8c8b2c,
						dnnl_aBCde8c16b2c,
						dnnl_aBCde8c8b,
						/// 5D tensor blocked by 3rd dimension with block size 4
						dnnl_aBCde2b4c2b,

						*/
					}
				}
			}
		}
		
		return format_tag::undef;
	}
		
	template<typename T>
	static void InitArray(T* destination, const std::size_t elements, const int initValue = 0) NOEXCEPT
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
		void release() NOEXCEPT
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;
		}
		AlignedArray() NOEXCEPT	{ }
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
		inline auto data() noexcept { return dataPtr; }
		inline const auto data() const noexcept { return dataPtr; }
		inline auto size() const noexcept { return nelems; }
		void resize(size_type elements, const T value = T()) NOEXCEPT
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
		inline T& operator[] (size_type i) NOEXCEPT { return dataPtr[i]; }
		inline const T& operator[] (size_type i) const NOEXCEPT { return dataPtr[i]; }
		inline auto empty() const noexcept { return nelems == 0; }
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
		void release() NOEXCEPT
		{
			if (arrPtr)
				arrPtr.reset();

			nelems = 0;
			arrPtr = nullptr;
			dataPtr = nullptr;			
		}
		AlignedMemory() NOEXCEPT { }
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
		inline auto memory() noexcept { return arrPtr.get(); }
		inline auto data() noexcept { return dataPtr; }
		inline const auto data() const noexcept { return dataPtr; }
		inline auto size() const noexcept { return nelems; }
		auto desc() { return description; }
		void resizeMem(const dnnl::memory::desc& md, const dnnl::engine& engine, const T value = T()) NOEXCEPT
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
		inline auto empty() const noexcept { return nelems == 0; }
	};

	typedef AlignedMemory<Float> FloatArray;
	typedef AlignedArray<Byte, 64ull> ByteArray;
	typedef std::vector<Float, AlignedAllocator<Float, 64ull>> FloatVector;
	//constexpr bool IS_LITTLE_ENDIAN = std::endian::native == std::endian::little;
	constexpr auto NEURONS_LIMIT = Float(5000);	// limit for all the neurons and derivatives [-NEURONS_LIMIT,NEURONS_LIMIT]
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
		if constexpr (Kahan)
		{
			const auto y = value - correction;
			const auto t = sum + y;
			correction = (t - sum) - y;
			sum = t;
		}
		else
			sum += value;
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
	static T Seed() NOEXCEPT
	{
		return static_cast<T>(__rdtsc());
	}
#else
	template<typename T>
	static T Seed() NOEXCEPT
	{
		return static_cast<T>(physicalSeed());
	}
#endif

	static auto BernoulliVecFloat(const Float p = Float(0.5)) NOEXCEPT
	{
#ifndef NDEBUG
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in BernoulliVecFloat function");
#endif
		static thread_local auto generator = Ranvec1(3, Seed<int>(), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())));

#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
		return select(generator.random16f() < p, Float(1), Float(0));
#elif defined(DNN_AVX2) || defined(DNN_AVX)
		return select(generator.random8f() < p, Float(1), Float(0));
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
		return select(generator.random4f() < p, Float(1), Float(0));
#endif
	}

	static auto UniformVecFloat(const Float min = Float(0), const Float max = Float(1)) NOEXCEPT
	{
#ifndef NDEBUG
		if (min >= max)
			throw std::invalid_argument("Parameter out of range in UniformVecFloat function");
#endif
		static thread_local auto generator = Ranvec1(3, Seed<int>(), static_cast<int>(std::hash<std::thread::id>()(std::this_thread::get_id())));
		const auto scale = std::abs(max - min);
	
#if defined(DNN_AVX512BW) || defined(DNN_AVX512)
		return (generator.random16f() * scale) + min;
#elif defined(DNN_AVX2) || defined(DNN_AVX)
		return (generator.random8f() * scale) + min;
#elif defined(DNN_SSE42) || defined(DNN_SSE41)
		return (generator.random4f() * scale) + min;
#endif
	}

	template<typename T>
	static auto Bernoulli(const Float p = Float(0.5)) NOEXCEPT
	{
#ifndef NDEBUG
		if (p < 0 || p > 1)
			throw std::invalid_argument("Parameter out of range in Bernoulli function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return static_cast<T>(std::bernoulli_distribution(static_cast<double>(p))(generator));
	}

	template<typename T>
	static auto UniformInt(const T min, const T max) NOEXCEPT
	{
		static_assert(std::is_integral<T>::value, "Only integral type supported in UniformInt function");
#ifndef NDEBUG
		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformInt function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return std::uniform_int_distribution<T>(min, max)(generator);
	}

	template<typename T>
	static auto UniformReal(const T min, const T max) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in UniformReal function");
#ifndef NDEBUG
		if (min > max)
			throw std::invalid_argument("Parameter out of range in UniformReal function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		return std::uniform_real_distribution<T>(min, max)(generator);
	}

	template<typename T>
	static auto TruncatedNormal(const T m, const T s, const T limit) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in TruncatedNormal function");
#ifndef NDEBUG
		if (limit < s)
	     throw std::invalid_argument("limit out of range in TruncatedNormal function");
#endif
		static thread_local auto generator = std::mt19937(Seed<unsigned>());
		T x;
		do { x = std::normal_distribution<T>(T(0), s)(generator); }
		while (std::abs(x) > limit); // reject if beyond limit
		
		return x + m;
	}

	/* https://stackoverflow.com/questions/15165202/random-number-generator-with-beta-distribution */
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
	auto BetaDistribution(const T a, const T b) NOEXCEPT
	{
		static_assert(std::is_floating_point<T>::value, "Only Floating point type supported in BetaDistribution function");
		static thread_local auto generator = std::mt19937(Seed<unsigned>());

		return ::beta_distribution<T>(a, b)(generator);
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

	/* https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring */
	auto Trim(std::string text)
	{
		text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](int ch) { return !std::isspace(ch); }));
		text.erase(std::find_if(text.rbegin(), text.rend(), [](int ch) { return !std::isspace(ch); }).base(), text.end());
		return text;
	}

	/* https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf */
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

	/*
	auto RelativeError(const Float reference, const Float actual)
	{
		return std::abs(reference - actual) / std::max(std::numeric_limits<Float>().min(), std::abs(reference));
	}
	
	auto auto Median(FloatVector& array)
	{
		std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
		return array[array.size() / 2];
	}
	*/

	/* https://stackoverflow.com/questions/64169258/c-print-days-hours-minutes-etc-of-a-chronoduration */
	template <class Rep, std::intmax_t num, std::intmax_t denom>
	auto ChronoBurst(std::chrono::duration<Rep, std::ratio<num, denom>> d)
	{
		const auto hrs = std::chrono::duration_cast<std::chrono::hours>(d);
		const auto mins = std::chrono::duration_cast<std::chrono::minutes>(d - hrs);
		const auto secs = std::chrono::duration_cast<std::chrono::seconds>(d - hrs - mins);
		const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(d - hrs - mins - secs);

		return std::make_tuple(hrs, mins, secs, ms);
	}
}