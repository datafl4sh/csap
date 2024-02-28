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

#include <cstdio>
#include <cmath>
#include <string>
#include <cblas.h>
#include <limits>
#ifdef HAVE_AVX
#include <immintrin.h>
#endif
#include "utils.h"

/* Plain AXPY */
template<typename T>
__attribute__((noinline))
static void
axpy(size_t N, T alpha, const T * __restrict__ x, T * __restrict__ y)
{
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++)
        y[i] = alpha*x[i] + y[i];
}

/* BLAS AXPY, float */
static void
axpy_blas(size_t N, float alpha, const float * __restrict__ x, float * __restrict__ y)
{
    if (N > std::numeric_limits<int>::max()) {
        fprintf(stderr, "axpy_blas: array size too big (%lu).\n", N);
        return;
    }
    int iN = N;
    cblas_saxpy(N, alpha, x, 1, y, 1);
}

/* BLAS AXPY, double */
static void
axpy_blas(size_t N, double alpha, const double * __restrict__ x, double * __restrict__ y)
{
    if (N > std::numeric_limits<int>::max()) {
        fprintf(stderr, "axpy_blas: array size too big (%lu).\n", N);
        return;
    }
    int iN = N;
    cblas_daxpy(N, alpha, x, 1, y, 1);
}

#ifdef HAVE_AVX
/* AVX single precision */
__attribute__((noinline))
static void
axpy_avx(size_t N, float alpha, const float * __restrict__ x,
    float * __restrict__ y)
{
    if (N < 8) {
        for (size_t i = 0; i < N; i++)
            y[i] = alpha*x[i] + y[i];
        return;
    }

    __m256 alphas = _mm256_broadcast_ss(&alpha);
    
    size_t NN = (N/8)*8;
    for (size_t i = 0; i < NN; i += 8) {
        __m256 xs = _mm256_load_ps(&x[i]);
        __m256 ys = _mm256_load_ps(&y[i]);
#ifdef HAVE_FMA
        _mm256_stream_ps(&y[i], _mm256_fmadd_ps(alphas, xs, ys) );
#else
        xs = _mm256_mul_ps(alphas, xs);
        _mm256_stream_ps(&y[i], _mm256_add_ps(xs, ys) );
#endif

    }
    _mm_sfence();
    
    for (size_t i = NN; i < N; i++)
        y[i] = alpha*x[i] + y[i];
}

/* AVX double precision */
__attribute__((noinline))
static void
axpy_avx(size_t N, double alpha, const double * __restrict__ x,
    double * __restrict__ y)
{
    if (N < 4) {
        for (size_t i = 0; i < N; i++)
            y[i] = alpha*x[i] + y[i];
        return;
    }

    __m256d alphas = _mm256_broadcast_sd(&alpha);
    
    size_t NN = (N/4)*4;
    for (size_t i = 0; i < NN; i += 4) {
        __m256d xs = _mm256_load_pd(&x[i]);
        __m256d ys = _mm256_load_pd(&y[i]);
#ifdef HAVE_FMA
        _mm256_stream_pd(&y[i], _mm256_fmadd_pd(alphas, xs, ys) );
#else
        xs = _mm256_mul_pd(alphas, xs);
        _mm256_stream_pd(&y[i], _mm256_add_pd(xs, ys) );
#endif
    }
    _mm_sfence();
    
    for (size_t i = NN; i < N; i++)
        y[i] = alpha*x[i] + y[i];
}
#endif /* HAVE_AVX */

enum class axpy_impl {
    naive,
#ifdef HAVE_AVX
    avx,
#endif
    blas
};

template<typename T>
static void
test_axpy(size_t testsize, const axpy_impl impl)
{
    /* Generate some test data */
    const T xval = T(1);
    const T yval = T(1);
    const T alpha = T(1);
    
    const size_t align = 32; /* AVX requires 32 byte alignment */
    size_t array_size = testsize*sizeof(T);
    if (array_size % align != 0) /* aligned_alloc() requires size multiple of align */
        array_size = (array_size & ~(align - 1)) + align;

    T *x = (T *) utils::checked_aligned_alloc(align, array_size);
    T *y = (T *) utils::checked_aligned_alloc(align, array_size);

    for (size_t i = 0; i < testsize; i++) {
        x[i] = xval;
        y[i] = yval;
    }

    utils::clock clk;

    if (axpy_impl::naive == impl) {
        clk.start();
        axpy(testsize, alpha, x, y);
    }

#ifdef HAVE_AVX
    if (axpy_impl::avx == impl) {
        clk.start();
        axpy_avx(testsize, alpha, x, y);
    }
#endif

    if (axpy_impl::blas == impl) {
        clk.start();
        axpy_blas(testsize, alpha, x, y);
    }

    const double axpy_time = clk.stop();

    /* Compute the number of bytes trasferred to and from memory:
     * each iteration of the for cycle reads two operands (x and y)
     * and writes the result, therefore for each cycle we transfer
     * 3*sizeof(T) bytes.
     */
    const double axpy_bytes = testsize*3*sizeof(T);
    
    /* Compute the number of floating point operations: for each iteration
     * of the for cycle we do one multiplication and one sum. */
    const double axpy_flops = testsize*2;
    
    printf("AXPY runtime:   %g seconds\n", axpy_time);
    printf("AXPY flops:     %g GFLOPS/s\n", axpy_flops*1e-9/axpy_time);
    //printf("AXPY bandwidth: %g GB/s\n", axpy_bytes*1e-9/axpy_time);
    
    T sum = 0.0;
    for (size_t i = 0; i < testsize; i++)
        sum += y[i];
    
    printf("Sum: %g\n", sum);

    free(x);
    free(y);
}

int main(int argc, const char **argv)
{
    const size_t testsize = 32*1024*1024;
    
    axpy_impl impl = axpy_impl::naive;
    std::string impl_name = "naive";

    if (argc > 1) {
        impl_name = argv[1];
        if (impl_name == "naive")
            impl = axpy_impl::naive;
        #ifdef HAVE_AVX
        else if (impl_name == "avx")
            impl = axpy_impl::avx;
        #endif
        else if (impl_name == "blas")
            impl = axpy_impl::blas;
        else {
            printf("Avail implementations: naive avx blas\n");
            return 1;
        }
    }

    printf("Testing single precision of %s implementation\n", impl_name.c_str());
    for (int i = 0; i < 4; i++)
        test_axpy<float>(testsize, impl);

    printf("Testing double precision of %s implementation\n", impl_name.c_str());
    for (int i = 0; i < 4; i++)
        test_axpy<double>(testsize, impl);

    return 0;
}
