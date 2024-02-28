/* This code is part of the demos used in the course
 *
 *    ``High Performance Scientific Computing - Part 1''
 *          at Politecnico di Torino - Italy.
 *
 * Copyright (c) 2024, Matteo Cicuttin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#include <iostream>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <array>
#ifdef HAVE_AVX
#include <immintrin.h>
#endif
#include "utils.h"

struct alloc_unaligned {};
struct alloc_aligned {};

struct xnpy_impl_naive : public alloc_unaligned {} impl_naive;
struct xnpy_impl_unroll4 : public alloc_unaligned {} impl_unroll4;
struct xnpy_impl_unroll8 : public alloc_unaligned {} impl_unroll8;
struct xnpy_impl_unroll16 : public alloc_unaligned {} impl_unroll16;
struct xnpy_impl_unroll32 : public alloc_unaligned {} impl_unroll32;
#ifdef HAVE_AVX
struct xnpy_impl_avx_unalign : public alloc_unaligned {} impl_avx_unalign;
struct xnpy_impl_avx_unalign_unroll : public alloc_unaligned {} impl_avx_unalign_unroll;
struct xnpy_impl_avx_align : public alloc_aligned {} impl_avx_align;
struct xnpy_impl_avx_align_unroll : public alloc_aligned {} impl_avx_align_unroll;
#endif

template<typename T>
__attribute__((noinline))
static void
xnpy(xnpy_impl_naive, size_t N, size_t pow, const T * __restrict__ x, T * __restrict__ y)
{
    for (size_t i = 0; i < N; i++) {
        T xpow = 1.0;
        for (size_t p = 0; p < pow; p++)
            xpow *= x[i];
        y[i] = xpow + y[i];
    }
}

template<size_t N, typename T>
auto make_array_of(T v) {
    return [&]<size_t ...Idx>(std::index_sequence<Idx...>) {
        return std::array{ (Idx, v)... };
    } (std::make_index_sequence<N>{});
}

/* Loop unrolling: we unroll the loop in order to allow the compiler to
 * better understand memory access patterns */
template<size_t UNROLL, typename T>
__attribute__((noinline))
static void
xnpy_unroll(size_t N, size_t pow, const T * __restrict__ x, T * __restrict__ y)
{
    if (N < UNROLL) {
        for (size_t i = 0; i < N; i++) {
            double xpow = 1.0;
            for (size_t p = 0; p < pow; p++)
                xpow *= x[i];
            y[i] = xpow + y[i];
        }
        return;
    }

    size_t NN = (N/UNROLL)*UNROLL;
    for (size_t i = 0; i < NN; i += UNROLL)
    {
        auto xpow = make_array_of<UNROLL>(1.0);

        for (size_t p = 0; p < pow; p++) {
            for (size_t u = 0; u < UNROLL; u++)
                xpow[u] *= x[i+u];
        }
        
        for (size_t u = 0; u < UNROLL; u++)
            y[i+u] = xpow[u] + y[i+u];
    }
    
    for (size_t i = NN; i < N; i++) {
        T xpow = 1.0;
        for (size_t p = 0; p < pow; p++)
            xpow *= x[i];
        y[i] = xpow + y[i];
    }
}

template<typename T>
__attribute__((noinline))
static void
xnpy(xnpy_impl_unroll4, size_t N, size_t pow, const T * __restrict__ x, T * __restrict__ y)
{
    return xnpy_unroll<4>(N, pow, x, y);
}

template<typename T>
__attribute__((noinline))
static void
xnpy(xnpy_impl_unroll8, size_t N, size_t pow, const T * __restrict__ x, T * __restrict__ y)
{
    return xnpy_unroll<8>(N, pow, x, y);
}

template<typename T>
__attribute__((noinline))
static void
xnpy(xnpy_impl_unroll16, size_t N, size_t pow, const T * __restrict__ x, T * __restrict__ y)
{
    return xnpy_unroll<16>(N, pow, x, y);
}

template<typename T>
__attribute__((noinline))
static void
xnpy(xnpy_impl_unroll32, size_t N, size_t pow, const T * __restrict__ x, T * __restrict__ y)
{
    return xnpy_unroll<32>(N, pow, x, y);
}

#ifdef HAVE_AVX
/* We use AVX vector instructions: process the data in blocks of 4 */
__attribute__((noinline))
static void
xnpy(xnpy_impl_avx_unalign, size_t N, size_t pow, const double * __restrict x, double * __restrict__ y)
{
    if (N < 4) {
        for (size_t i = 0; i < N; i++) {
            double xpow = 1.0;
            for (size_t p = 0; p < pow; p++)
                xpow *= x[i];
            y[i] = xpow + y[i];
        }
        return;
    }

    const double one = 1.0;
    size_t NN = (N/4)*4;
    for (size_t i = 0; i < NN; i+= 4) {
        __m256d xpows = _mm256_broadcast_sd(&one);
        __m256d yis = _mm256_loadu_pd(&y[i]);
        __m256d xis = _mm256_loadu_pd(&x[i]);
#ifdef HAVE_FMA
        for (size_t p = 0; pow > 0 && p < pow-1; p++)
            xpows = _mm256_mul_pd(xpows, xis);
        _mm256_storeu_pd( &y[i], _mm256_fmadd_pd(xpows, xis, yis) );
#else
        for (size_t p = 0; p < pow; p++)
            xpows = _mm256_mul_pd(xpows, xis);
        _mm256_storeu_pd( &y[i], _mm256_add_pd(xpows, yis) );
#endif
    }

    for (size_t i = NN; i < N; i++) {
        double xpow = 1.0;
        for (size_t p = 0; p < pow; p++)
            xpow *= x[i];
        y[i] = xpow + y[i];
    }
}

/* AVX & unroll: we process in blocks of 32, making sure that all the YMM
 * registers are used */
__attribute__((noinline))
static void
xnpy(xnpy_impl_avx_unalign_unroll, size_t N, size_t pow, const double * __restrict x,
    double * __restrict__ y)
{
    if (N < 32) {
        for (size_t i = 0; i < N; i++) {
            double xpow = 1.0;
            for (size_t p = 0; p < pow; p++)
                xpow *= x[i];
            y[i] = xpow + y[i];
        }
        return;
    }

    const double one = 1.0;
    
    size_t NN = (N/32)*32;
    for (size_t i = 0; i < NN; i+= 32) {
        __m256d xpow0 = _mm256_broadcast_sd(&one);
        __m256d xpow1 = xpow0;
        __m256d xpow2 = xpow0;
        __m256d xpow3 = xpow0;
        __m256d xpow4 = xpow0;
        __m256d xpow5 = xpow0;
        __m256d xpow6 = xpow0;
        __m256d xpow7 = xpow0;
    
        __m256d xis0 = _mm256_loadu_pd(&x[i+0]);
        __m256d xis1 = _mm256_loadu_pd(&x[i+4]);
        __m256d xis2 = _mm256_loadu_pd(&x[i+8]);
        __m256d xis3 = _mm256_loadu_pd(&x[i+12]);
        __m256d xis4 = _mm256_loadu_pd(&x[i+16]);
        __m256d xis5 = _mm256_loadu_pd(&x[i+20]);
        __m256d xis6 = _mm256_loadu_pd(&x[i+24]);
        __m256d xis7 = _mm256_loadu_pd(&x[i+28]);

        for (size_t p = 0; p < pow; p++) {
            xpow0 = _mm256_mul_pd(xpow0, xis0);
            xpow1 = _mm256_mul_pd(xpow1, xis1);
            xpow2 = _mm256_mul_pd(xpow2, xis2);
            xpow3 = _mm256_mul_pd(xpow3, xis3);
            xpow4 = _mm256_mul_pd(xpow4, xis4);
            xpow5 = _mm256_mul_pd(xpow5, xis5);
            xpow6 = _mm256_mul_pd(xpow6, xis6);
            xpow7 = _mm256_mul_pd(xpow7, xis7);
        }

        xis0 = _mm256_loadu_pd(&y[i+0]);
        xis1 = _mm256_loadu_pd(&y[i+4]);
        xis2 = _mm256_loadu_pd(&y[i+8]);
        xis3 = _mm256_loadu_pd(&y[i+12]);
        xis4 = _mm256_loadu_pd(&y[i+16]);
        xis5 = _mm256_loadu_pd(&y[i+20]);
        xis6 = _mm256_loadu_pd(&y[i+24]);
        xis7 = _mm256_loadu_pd(&y[i+28]);

        xpow0 = _mm256_add_pd(xpow0, xis0);
        xpow1 = _mm256_add_pd(xpow1, xis1);
        xpow2 = _mm256_add_pd(xpow2, xis2);
        xpow3 = _mm256_add_pd(xpow3, xis3);
        xpow4 = _mm256_add_pd(xpow4, xis4);
        xpow5 = _mm256_add_pd(xpow5, xis5);
        xpow6 = _mm256_add_pd(xpow6, xis6);
        xpow7 = _mm256_add_pd(xpow7, xis7);

        _mm256_storeu_pd( &y[i+0], xpow0 );
        _mm256_storeu_pd( &y[i+4], xpow1 );
        _mm256_storeu_pd( &y[i+8], xpow2 );
        _mm256_storeu_pd( &y[i+12], xpow3 );
        _mm256_storeu_pd( &y[i+16], xpow4 );
        _mm256_storeu_pd( &y[i+20], xpow5 );
        _mm256_storeu_pd( &y[i+24], xpow6 );
        _mm256_storeu_pd( &y[i+28], xpow7 );
    }

    for (size_t i = NN; i < N; i++) {
        double xpow = 1.0;
        for (size_t p = 0; p < pow; p++)
            xpow *= x[i];
        y[i] = xpow + y[i];
    }
}

/*
template<typename T>
static void
align(size_t align, size_t& elems, T& ptr)
{
    // Don't use this in real world. Checks on T must be enforced.
    //static_assert(std::is_pointer_v<T>, "T must be a pointer");
    T align_ptr = (T)((((size_t)ptr) & ~(align-1)) + align);
    ptrdiff_t delta_elems = align_ptr - ptr;
    
    if (delta_elems < 0)
        elems = 0;

    elems -= delta_elems;
    ptr += delta_elems;
}
*/

/* AVX on aligned memory */
__attribute__((noinline))
static void
xnpy(xnpy_impl_avx_align, size_t N, size_t pow, const double * __restrict x, double * __restrict__ y)
{
    if (N < 4) {
        /* N less than 4: vector too small for the AVX loop,
         * compute directly and return. */
        for (size_t i = 0; i < N; i++) {
            double xpow = 1.0;
            for (size_t p = 0; p < pow; p++)
                xpow *= x[i];
            y[i] = xpow + y[i];
        }
        return;
    }

    /* Use AVX in the aligned part */
    const double one = 1.0;
    size_t NN = (N/4)*4;
    for (size_t i = 0; i < NN; i+= 4) {
        __m256d xpows = _mm256_broadcast_sd(&one);
        __m256d yis = _mm256_load_pd(&y[i]);
        __m256d xis = _mm256_load_pd(&x[i]);
#ifdef HAVE_FMA
        for (size_t p = 0; pow > 0 && p < pow-1; p++)
            xpows = _mm256_mul_pd(xpows, xis);
        _mm256_store_pd( &y[i], _mm256_fmadd_pd(xpows, xis, yis) );
#else
        for (size_t p = 0; p < pow; p++)
            xpows = _mm256_mul_pd(xpows, xis);
        _mm256_store_pd( &y[i], _mm256_add_pd(xpows, yis) );
#endif
    }

    /* And finish the remaining elements */
    for (size_t i = NN; i < N; i++) {
        double xpow = 1.0;
        for (size_t p = 0; p < pow; p++)
            xpow *= x[i];
        y[i] = xpow + y[i];
    }
}

/* AVX & unroll on aligned memory */
__attribute__((noinline))
static void
xnpy(xnpy_impl_avx_align_unroll, size_t N, size_t pow, const double * __restrict x,
    double * __restrict__ y)
{
    if (N < 32) {
        /* N less than 32: vector too small for the AVX loop,
         * compute directly and return. */
        for (size_t i = 0; i < N; i++) {
            double xpow = 1.0;
            for (size_t p = 0; p < pow; p++)
                xpow *= x[i];
            y[i] = xpow + y[i];
        }
        return;
    }

    /* Use AVX in the aligned part */
    const double one = 1.0;
    size_t NN = (N/32)*32;
    #pragma omp parallel for
    for (size_t i = 0; i < NN; i+= 32) {
        __m256d xpow0 = _mm256_broadcast_sd(&one);
        __m256d xpow1 = xpow0;
        __m256d xpow2 = xpow0;
        __m256d xpow3 = xpow0;
        __m256d xpow4 = xpow0;
        __m256d xpow5 = xpow0;
        __m256d xpow6 = xpow0;
        __m256d xpow7 = xpow0;
    
        __m256d xis0 = _mm256_load_pd(&x[i+0]);
        __m256d xis1 = _mm256_load_pd(&x[i+4]);
        __m256d xis2 = _mm256_load_pd(&x[i+8]);
        __m256d xis3 = _mm256_load_pd(&x[i+12]);
        __m256d xis4 = _mm256_load_pd(&x[i+16]);
        __m256d xis5 = _mm256_load_pd(&x[i+20]);
        __m256d xis6 = _mm256_load_pd(&x[i+24]);
        __m256d xis7 = _mm256_load_pd(&x[i+28]);

        for (size_t p = 0; p < pow; p++) {
            xpow0 = _mm256_mul_pd(xpow0, xis0);
            xpow1 = _mm256_mul_pd(xpow1, xis1);
            xpow2 = _mm256_mul_pd(xpow2, xis2);
            xpow3 = _mm256_mul_pd(xpow3, xis3);
            xpow4 = _mm256_mul_pd(xpow4, xis4);
            xpow5 = _mm256_mul_pd(xpow5, xis5);
            xpow6 = _mm256_mul_pd(xpow6, xis6);
            xpow7 = _mm256_mul_pd(xpow7, xis7);
        }

        xis0 = _mm256_load_pd(&y[i+0]);
        xis1 = _mm256_load_pd(&y[i+4]);
        xis2 = _mm256_load_pd(&y[i+8]);
        xis3 = _mm256_load_pd(&y[i+12]);
        xis4 = _mm256_load_pd(&y[i+16]);
        xis5 = _mm256_load_pd(&y[i+20]);
        xis6 = _mm256_load_pd(&y[i+24]);
        xis7 = _mm256_load_pd(&y[i+28]);

        xpow0 = _mm256_add_pd(xpow0, xis0);
        xpow1 = _mm256_add_pd(xpow1, xis1);
        xpow2 = _mm256_add_pd(xpow2, xis2);
        xpow3 = _mm256_add_pd(xpow3, xis3);
        xpow4 = _mm256_add_pd(xpow4, xis4);
        xpow5 = _mm256_add_pd(xpow5, xis5);
        xpow6 = _mm256_add_pd(xpow6, xis6);
        xpow7 = _mm256_add_pd(xpow7, xis7);

        _mm256_stream_pd( &y[i+0], xpow0 );
        _mm256_stream_pd( &y[i+4], xpow1 );
        _mm256_stream_pd( &y[i+8], xpow2 );
        _mm256_stream_pd( &y[i+12], xpow3 );
        _mm256_stream_pd( &y[i+16], xpow4 );
        _mm256_stream_pd( &y[i+20], xpow5 );
        _mm256_stream_pd( &y[i+24], xpow6 );
        _mm256_stream_pd( &y[i+28], xpow7 );
    }


    /* And finish the remaining elements */
    for (size_t i = NN; i < N; i++) {
        double xpow = 1.0;
        for (size_t p = 0; p < pow; p++)
            xpow *= x[i];
        y[i] = xpow + y[i];
    }
}
#endif /* HAVE_AVX */

template<typename T>
auto alloc_for_xnpy(alloc_unaligned, size_t testsize)
{
    size_t array_size = testsize*sizeof(T);

    T *x = (T *) utils::checked_malloc(array_size);
    T *y = (T *) utils::checked_malloc(array_size);
    
    return std::pair(x,y);
}

template<typename T>
auto alloc_for_xnpy(alloc_aligned, size_t testsize)
{
    size_t array_size = testsize*sizeof(T);

    T *x = (T *) utils::checked_aligned_alloc(32, array_size);
    T *y = (T *) utils::checked_aligned_alloc(32, array_size);
    
    return std::pair(x,y);
}

template<typename T, typename IMPL>
static void
test_xnpy(IMPL impl, size_t testsize, size_t power, FILE *log)
{
    /* Generate some test data */
    const T xval = T(1.1);
    const T yval = T(1.2);
    
    auto [x, y] = alloc_for_xnpy<T>(impl, testsize);

    for (size_t i = 0; i < testsize; i++) {
        x[i] = xval;
        y[i] = yval;
    }

    utils::clock clk;
    clk.start();
    xnpy(impl, testsize, power, x, y);
    const double xnpy_time = clk.stop();

    const double xnpy_bytes = testsize*3*sizeof(T);
    const double xnpy_flops = testsize*(power+1);
    const double xnpy_gflops_s = xnpy_flops*1e-9/xnpy_time;
    const double xnpy_gb_s = xnpy_bytes*1e-9/xnpy_time;
    const double xnpy_ai = xnpy_flops/xnpy_bytes;
    
    printf("XNPY runtime:   %g seconds\n", xnpy_time);
    printf("XNPY flops:     %g GFLOPS/s\n", xnpy_gflops_s);
    //printf("XNPY bandwidth: %g GB/s\n", xnpy_gb_s);
    //printf("XNPY AI:        %g FLOPS/byte\n", xnpy_ai);
    
    if (log) {
        fprintf(log, "%ld %g %g %g %g\n", power, xnpy_time, xnpy_gflops_s,
            xnpy_gb_s, xnpy_ai);
    }

    T sum = 0.0;
    for (size_t i = 0; i < testsize; i++)
        sum += y[i];
    
    auto refpow = [](size_t ts, size_t pow, double x, double y) {
        T ret = 0.0, xpow = 1.0;
        for (size_t p = 0; p < pow; p++) xpow *= x;
        for (size_t i = 0; i < ts; i++) ret += (y + xpow);
        return ret;
    };
    printf("%g\n", sum-refpow(testsize, power, xval, yval));

    free(x);
    free(y);
}

template<typename T, typename IMPL>
void
test_xnpy(IMPL impl, const std::string& impl_name)
{
    const size_t testsize = 80*1000*1000;
    
    std::string filename = "xnpy_" + impl_name + ".log";
    
    FILE *log = fopen(filename.c_str(), "w+");
    if (!log) {
        perror("fopen");
        log = NULL;
    }
    
    for (size_t i = 1; i < 50; i++)
        test_xnpy<T>(impl, testsize, i, log);
    
    if (log)
        fclose(log);
}

int main(int argc, char **argv)
{
    using T = double;
    
    if (argc > 1) {
        std::string impl_name;
        impl_name = argv[1];
        if (impl_name == "naive")
            test_xnpy<T>(impl_naive, impl_name);
        else if (impl_name == "unroll4")
            test_xnpy<T>(impl_unroll4, impl_name);
        else if (impl_name == "unroll8")
            test_xnpy<T>(impl_unroll8, impl_name);
        else if (impl_name == "unroll16")
            test_xnpy<T>(impl_unroll16, impl_name);
        else if (impl_name == "unroll32")
            test_xnpy<T>(impl_unroll32, impl_name);
        #ifdef HAVE_AVX
        else if (impl_name == "avx_unalign")
            test_xnpy<T>(impl_avx_unalign, impl_name);
        else if (impl_name == "avx_unalign_unroll")
            test_xnpy<T>(impl_avx_unalign_unroll, impl_name);
        else if (impl_name == "avx_align")
            test_xnpy<T>(impl_avx_align, impl_name);
        else if (impl_name == "avx_align_unroll")
            test_xnpy<T>(impl_avx_align_unroll, impl_name);
        #endif
        else {
            printf("Avail implementations: naive unroll4 unroll8 unroll16 unroll32 ");
            #ifdef HAVE_AVX
            printf("avx_unalign avx_unalign_unroll avx_align avx_align_unroll ");
            #endif
            printf("\n");
            return 1;
        }
    }
    return 0;
}
