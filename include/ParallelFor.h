#pragma once

#if defined DNN_OMP
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

#ifndef DNN_UNREF_PAR
  #if defined(_MSC_VER)
	#define DNN_UNREF_PAR(P) UNREFERENCED_PARAMETER(P)
  #else
	#define DNN_UNREF_PAR(P) (void)P
  #endif
#endif

namespace dnn
{
#if !defined DNN_OMP
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
#if defined DNN_OMP
		#pragma omp parallel num_threads(omp_get_max_threads())
		{
			#pragma omp for schedule(static,1)
			for (auto i = 0ll; i < static_cast<long long>(range); i++)
				f(i);
		}
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
#if defined DNN_OMP
		#pragma omp parallel num_threads(static_cast<int>(threads))
		{
			#pragma omp for schedule(static,1)
			for (auto i = 0ll; i < static_cast<long long>(range); i++)
				f(i);
		}
#else
		DNN_UNREF_PAR(threads);

		for_(0ull, range, [&](const blocked_range& r)
		{
			for (auto i = r.begin(); i < r.end(); i++)
				f(i);
		});
#endif
	}
}