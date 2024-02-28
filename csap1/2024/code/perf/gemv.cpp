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

#include <cblas.h>

#include "utils.h"

/* Plain GEMV */
template<typename T>
__attribute__((noinline))
static void
gemv(int N, T alpha, T beta, const T * __restrict__ A,
    const T * __restrict__ x, T * __restrict__ y)
{
    for (int i = 0; i < N; i++)
        y[i] = beta*y[i];               // 2N mem, N flops
        
    for (int i = 0; i < N; i++)         // 2N mem (y read and write)
        for (int j = 0; j < N; j++) 
            y[i] += alpha*A[i*N+j]*x[j];    //N*N reads on x, but x is likely
                                            //to be always in fast cache, N*N
                                            //reads on A and 3*N*N flops
}

template<typename T>
__attribute__((noinline))
static void
gemv_unroll(int N, T alpha, T beta, const T * __restrict__ A,
    const T * __restrict__ x, T * __restrict__ y)
{
    for (int i = 0; i < N; i++)
        y[i] = beta*y[i];
        
    size_t NN = (N/8)*8;
    for (size_t i = 0; i < NN; i+=8) {
        for (size_t j = 0; j < N; j++) {
            const T xj = x[j];
            y[i+0] += alpha*A[(i+0)*N+j]*xj;
            y[i+1] += alpha*A[(i+1)*N+j]*xj;
            y[i+2] += alpha*A[(i+2)*N+j]*xj;
            y[i+3] += alpha*A[(i+3)*N+j]*xj;
            y[i+4] += alpha*A[(i+4)*N+j]*xj;
            y[i+5] += alpha*A[(i+5)*N+j]*xj;
            y[i+6] += alpha*A[(i+6)*N+j]*xj;
            y[i+7] += alpha*A[(i+7)*N+j]*xj;
        }
    }

    for (size_t i = NN; i < N; i++)
        for (size_t j = 0; j < N; j++) 
            y[i] += alpha*A[i*N+j]*x[j];
}

template<typename T>
static void
test_gemv(int N)
{
    /* Generate some test data */
    const T xval = (T) 4.2;
    const T yval = (T) 8.4;
    const T alpha = (T) M_PI;
    const T beta = (T) M_E;
    
    T *A = (T *) utils::checked_malloc(N*N*sizeof(T));
    T *x = (T *) utils::checked_malloc(N*sizeof(T));
    T *y = (T *) utils::checked_malloc(N*sizeof(T));
    
    for (size_t i = 0; i < N; i++) {
        x[i] = xval;
        y[i] = 0;
        for (size_t j = 0; j < N; j++)
            A[i*N+j] = 1;
    }
    
    utils::clock clk;
    clk.start();
    //gemv(N, alpha, beta, A, x, y);
    gemv_unroll(N, alpha, beta, A, x, y);
    //cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, alpha, A, N, x, 1, beta, y, 1);
    const double gemv_time = clk.stop();
    
    /* Compute the number of bytes trasferred to and from memory:
     * each iteration of the for cycle reads two operands (x and y)
     * and writes the result, therefore for each cycle we transfer
     * 3*sizeof(T) bytes.
     */
    const double gemv_bytes = (4*N + (2.*N)*N)*sizeof(T);
    
    /* Compute the number of floating point operations: for each iteration
     * of the for cycle we do one multiplication and one sum. */
    const double gemv_flops = (3.*N)*N + N;
    
    printf("GEMV runtime:   %g seconds\n", gemv_time);
    printf("GEMV flops:     %g GFLOPS/s\n", gemv_flops*1e-9/gemv_time);
    printf("GEMV bandwidth: %g GB/s\n", gemv_bytes*1e-9/gemv_time);
    
    T sum = 0.0;
    for (size_t i = 0; i < N; i++)
        sum += y[i];
    
    printf("Sum: %g\n", sum);
    
    free(A);
    free(x);
    free(y);
}

int main(void)
{
    test_gemv<double>(20000);
}