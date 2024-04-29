#pragma once
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
#include <omp.h>
#else
#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <future>
#include <thread>
#endif

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)
#define CHAIn2(a, b) a b
#define CHAIN2(a, b) CHAIn2(a, b)

#ifdef _MSC_VER
#define PRAGMA_MACRo(x) __pragma(x)
#else
#define PRAGMA_MACRo(x) _Pragma(#x)
#endif
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n) PRAGMA_MACRO(omp parallel for collapse(n))
#define PRAGMA_OMP_PARALLEL_THREADS(n) PRAGMA_MACRO(omp parallel num_threads(n))
#define PRAGMA_OMP_FOR_SCHEDULE_STATIC(n) PRAGMA_MACRO(omp for schedule(static,n))
#define PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(n) PRAGMA_MACRO(omp for schedule(dynamic,n))
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#define PRAGMA_OMP(...)
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)
#define PRAGMA_OMP_PARALLEL_THREADS(n)
#define PRAGMA_OMP_FOR_SCHEDULE_STATIC(n)
#define PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(n)
#define OMP_GET_THREAD_NUM() 0
#define OMP_GET_NUM_THREADS() 1
#endif

// MSVC still supports omp 2.0 only
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#define PRAGMA_OMP_SIMD(...)
#else
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)


#if (defined(__clang_major__) \
        && (__clang_major__ < 3 \
                || (__clang_major__ == 3 && __clang_minor__ < 9))) \
        || (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
        || (!defined(__INTEL_COMPILER) && !defined(__clang__) \
                && (defined(_MSC_VER) || __GNUC__ < 6 \
                        || (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif // long simdlen if

#ifndef DNN_UNREF_PAR
  #ifdef _MSC_VER
	#define DNN_UNREF_PAR(P) (P)
  #else
	#define DNN_UNREF_PAR(P) (void)P
  #endif
#endif

namespace dnn
{
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP
	struct blocked_range 
	{
		typedef size_t const_iterator;

		blocked_range(const size_t begin, const size_t end) : begin_(begin), end_(end) {}
		blocked_range(const int& begin, const int& end) : begin_(begin), end_(end) {}

		const_iterator begin() const { return begin_; }
		const_iterator end() const { return end_; }

	private:
		const size_t begin_;
		const size_t end_;
	};

	template <typename Func>
	void xparallel_for(const size_t begin, const size_t end, const Func& f)
	{
		blocked_range r(begin, end);
		f(r);
	}

	template <typename Func>
	void parallel_for(const size_t begin, const size_t end, const Func& f) 
	{
		assert(end >= begin);

		const auto nthreads = std::thread::hardware_concurrency();
		auto blockSize = (end - begin) / nthreads;
		if (blockSize * nthreads < end - begin)
			blockSize++;

		std::vector<std::future<void>> futures;

		auto blockBegin = begin;
		auto blockEnd = blockBegin + blockSize;
		if (blockEnd > end) 
			blockEnd = end;

		for (auto i = 0ull; i < nthreads; i++) 
		{
			futures.push_back(std::move(std::async(std::launch::async, [blockBegin, blockEnd, &f] {	f(blocked_range(blockBegin, blockEnd));	})));

			blockBegin += blockSize;
			if (blockBegin >= end) 
				break;

			blockEnd = blockBegin + blockSize;
			if (blockEnd > end) 
				blockEnd = end;
		}

		for (auto &future : futures) 
			future.wait();
	}

	template <typename T, typename U>
	bool value_representation(U const &value) { return static_cast<U>(static_cast<T>(value)) == value; }

	template <typename Func>
	inline void for_(const size_t begin, const size_t end, const Func& f)
	{
		value_representation<size_t>(end) ?	parallel_for(begin, end, f)	: xparallel_for(begin, end, f);
	}
#endif

	template <typename Func>
	inline void for_i(const size_t range, const Func& f)
	{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
	#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
		PRAGMA_OMP_PARALLEL_THREADS(omp_get_max_threads())
		{
			PRAGMA_OMP_FOR_SCHEDULE_STATIC(1)
			for (auto i = 0ll; i < static_cast<long long>(range); i++)
				f(i);
		}
	#else
		#pragma omp parallel for shared(f) schedule(static,1) num_threads(omp_get_max_threads())
		for (auto i = 0ull; i < range; i++)
			f(i);
	#endif
	
#else
		for_(0ull, range, [&](const blocked_range& r)
		{
			for (auto i = r.begin(); i < r.end(); i++)
				f(i);
		});
#endif
	}

	template <typename Func>
	inline void for_i(const size_t range, const size_t threads, const Func& f)
	{
		if (threads > 1)
		{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
	#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
			PRAGMA_OMP_PARALLEL_THREADS(static_cast<int>(threads))
			{
				PRAGMA_OMP_FOR_SCHEDULE_STATIC(1)
				for (auto i = 0ll; i < static_cast<long long>(range); i++)
					f(i);
			}
	#else
            #pragma omp parallel for shared(f) schedule(static,1) num_threads(static_cast<int>(threads))
			for (auto i = 0ull; i < range; i++)
				f(i);
	#endif
#else
			DNN_UNREF_PAR(threads);
			for_(0ull, range, [&](const blocked_range& r)
			{
				for (auto i = r.begin(); i < r.end(); i++)
				f(i);
			});
#endif
		}
		else
			for (auto i = 0ull; i < range; i++)
				f(i);
	}

	template <typename Func>
	inline void for_i_dynamic(const size_t range, const Func& f)
	{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
	#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
		PRAGMA_OMP_PARALLEL_THREADS(omp_get_max_threads())
		{
			PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(1)
			for (auto i = 0ll; i < static_cast<long long>(range); i++)
				f(i);
		}
	#else
		#pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_max_threads())
		for (auto i = 0ull; i < range; i++)
			f(i);
	#endif
#else
		for_(0ull, range, [&](const blocked_range& r)
		{
			for (auto i = r.begin(); i < r.end(); i++)
				f(i);
		});
#endif
	}

	template <typename Func>
	inline void for_i_dynamic(const size_t range, const size_t threads, const Func& f)
	{
		if (threads > 1)
		{
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
	#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
			PRAGMA_OMP_PARALLEL_THREADS(static_cast<int>(threads))
			{
				PRAGMA_OMP_FOR_SCHEDULE_DYNAMIC(1)
				for (auto i = 0ll; i < static_cast<long long>(range); i++)
					f(i);
			}
	#else
			#pragma omp parallel for schedule(dynamic,1) num_threads(static_cast<int>(threads))
			for (auto i = 0ull; i < range; i++)
				f(i);
	#endif
#else
			DNN_UNREF_PAR(threads);
			for_(0ull, range, [&](const blocked_range& r)
			{
				for (auto i = r.begin(); i < r.end(); i++)
					f(i);
			});
#endif
		}
		else
			for (auto i = 0ull; i < range; i++)
				f(i);
	}
}