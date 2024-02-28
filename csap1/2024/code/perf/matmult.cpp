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

#include "utils.h"

#define I(m,n) (((m)*(N))+(n))

template<typename T>
__attribute__((noinline))
static void
matmult_naive(T * __restrict__ c, const T * __restrict__ a,
        const T * __restrict__ b, size_t N)
{
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                c[I(i,j)] += a[I(i,k)] * b[I(k,j)];
}

template<typename T>
__attribute__((noinline))
static void
matmult_reorder(T * __restrict__ c, const T * __restrict__ a,
        const T * __restrict__ b, size_t N)
{
    for (size_t i = 0; i < N; i++)
        for (size_t k = 0; k < N; k++)      // swapped this
            for (size_t j = 0; j < N; j++)  // with this
                c[I(i,j)] += a[I(i,k)] * b[I(k,j)];
}

/* Probable GCC bug: The 'static' qualifier kills performance under GCC-12.
 * My theory is that when the funcion is declared 'static', GCC inlines but
 * forgets the __restrict__ qualifiers, so performance is destroyed. 'noinline'
 * recovers all the performance, supporting the theory (no, I don't want to go
 * and check out the ASM now). Under Clang there is no difference. */

template<typename T>
__attribute__((noinline))
static void
matmult_blocked(T * __restrict__ c, const T * __restrict__ a,
    const T * __restrict__ b, size_t N)
{
    const size_t BF = 8;
    const size_t NN = BF*(N/BF);

    
    for (size_t ii = 0; ii < NN; ii += BF)
    {
        const size_t iiBF = ii + BF;
        const size_t i_lim = iiBF;
        for (size_t kk = 0; kk < NN; kk += BF)
        {
            const size_t kkBF = kk + BF;
            const size_t k_lim = kkBF;
            for (size_t jj = 0; jj < NN; jj += BF)
            {
                const size_t jjBF = jj + BF;
                const size_t j_lim = jjBF;
                
                for (size_t i = ii; i < i_lim; i++)
                    for (size_t k = kk; k < k_lim; k++)
                        for (size_t j = jj; j < j_lim; j++)
                            c[I(i,j)] += a[I(i,k)]*b[I(k,j)];
            }
        }
    }

    for (size_t i = NN; i < N; i++)
        for (size_t k = 0; k < N; k++)
            for (size_t j = 0; j < NN; j++)
                c[I(i,j)] += a[I(i,k)] * b[I(k,j)];
    
    for (size_t i = 0; i < N; i++)
        for (size_t k = 0; k < N; k++)
            for (size_t j = NN; j < N; j++)
                c[I(i,j)] += a[I(i,k)] * b[I(k,j)];
        
    for (size_t i = 0; i < NN; i++)
        for (size_t k = NN; k < N; k++)
            for (size_t j = 0; j < NN; j++)
                c[I(i,j)] += a[I(i,k)] * b[I(k,j)];
}

template<typename T>
static void
test_matmult(size_t N)
{
    T *a = (T *) utils::checked_malloc(N*N*sizeof(T));
    T *b = (T *) utils::checked_malloc(N*N*sizeof(T));
    T *c = (T *) utils::checked_malloc(N*N*sizeof(T));
    
    for (size_t i = 0; i < N*N; i++) {
        a[i] = 1;
        b[i] = 1;
        c[i] = 0;
    }
    
    utils::clock clk;
    clk.start();
    matmult_blocked(c,a,b,N);

    const double mm_time = clk.stop();
    const double mm_flops = 2*N*N*N;
    const double mm_gflops_s = mm_flops*1e-9/mm_time;

    printf("%g %lu %g ", mm_time, N, mm_gflops_s);
    
    
    T sum = 0.0;
    for (size_t i = 0; i < N*N; i++)
        sum += c[i];
    printf("%g\n", sum - N*N*N);

    free(a);
    free(b);
    free(c);
}

int main(void)
{
    //test_matmult<double>(2501);
    for (size_t i = 100; i <= 1600; i+= 100)
        test_matmult<double>(i);
}
