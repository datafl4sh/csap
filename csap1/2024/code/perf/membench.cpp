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
#include <fstream>
#include <cstring>
#include <immintrin.h>

#include "utils.h"

static void
memcpy_naive(char * __restrict__ dst, const char * __restrict__ src, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = src[i];
}

static void
memcpy_avx(char * __restrict__ dst, const char * __restrict__ src, size_t n)
{
    for (size_t i = 0; i < n; i += 32) {
        __m256i mem = _mm256_load_si256((__m256i*)&src[i]);
        _mm256_store_si256((__m256i*)&dst[i], mem);
    }
}

static void
benchmark_memcpy(void)
{
    std::ofstream ofs("memcpy.log");

    for (size_t size = (1<<6), reps = (1<<28); size < (1<<27)+1; size *= 2, reps /= 2)
    {
        std::cout << "Testing size " << utils::size_to_human(size) << " with ";
        std::cout << utils::size_to_human(reps) << " repetitions: ";
        std::cout.flush();

        char *a = (char *) utils::checked_aligned_alloc(32, size);
        char *b = (char *) utils::checked_aligned_alloc(32, size);

        utils::clock clk;
        clk.start();

        for (size_t rep = 0; rep < reps; rep++) {
            memcpy(b, a, size);
            //memcpy_naive(b, a, size);
            //memcpy_avx(b, a, size); 
            char *t = a;
            a = b;
            b = t;
        }
    
        const double memcpy_time = clk.stop()/reps;
        const double memcpy_bw = 2*size/(memcpy_time*1e9);

        std::cout << memcpy_bw << " GB/s" << std::endl;

        if (ofs.is_open())
            ofs << size << " " << memcpy_time << " " << memcpy_bw << std::endl;

        free(a);
        free(b);
    }
}

static void
benchmark_memset(void)
{
    std::ofstream ofs("memset.log");

    for (size_t size = (1<<6), reps = (1<<28); size < (1<<27)+1; size *= 2, reps /= 2)
    {
        std::cout << "Testing size " << utils::size_to_human(size) << " with ";
        std::cout << utils::size_to_human(reps) << " repetitions: ";
        std::cout.flush();

        char *a = (char *) utils::checked_malloc(size);

        utils::clock clk;
        clk.start();

        for (size_t rep = 0; rep < reps; rep++)
            memset(a, rep%256, size);
    
        const double memset_time = clk.stop()/reps;
        const double memset_bw = size/(memset_time*1e9);

        std::cout << memset_bw << " GB/s" << std::endl;

        if (ofs.is_open())
            ofs << size << " " << memset_time << " " << memset_bw << std::endl;

        free(a);
    }
}

int main(void)
{
    std::cout << " ==== MEMCPY benchmark ====" << std::endl;
    benchmark_memcpy();
    std::cout << " ==== MEMSET benchmark ====" << std::endl;
    benchmark_memset();
    return 0;
}