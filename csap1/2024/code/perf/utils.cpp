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
#include <cstdlib>
#include "utils.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#define timespecsub(a, b, result)                           \
    do {                                                    \
        (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;       \
        (result)->tv_nsec = (a)->tv_nsec - (b)->tv_nsec;    \
        if ((result)->tv_nsec < 0) {                        \
            --(result)->tv_sec;                             \
            (result)->tv_nsec += 1000000000;                \
        }                                                   \
    } while (0)


static void
realtime(struct timespec *ts)
{
#ifdef __MACH__
    clock_serv_t clk;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &clk);
    mach_timespec_t mts;
    clock_get_time(clk, &mts);
    mach_port_deallocate(mach_task_self(), clk);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, ts);
#endif
}

static void
cputime(struct timespec *ts)
{
#ifdef __MACH__
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    ts->ts_sec = ru->ru_utime.tv_sec;
    ts->ts_nsec = ru->ru_utime.tv_usec*1000;
#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ts);
#endif
}


namespace utils {

clock::clock()
{}

void
clock::start()
{
    realtime(&ts_start);
}

double
clock::stop()
{
    realtime(&ts_end);
    struct timespec diff;
    timespecsub(&ts_end, &ts_start, &diff);
    return diff.tv_sec + diff.tv_nsec/1e9;
}

void *
checked_malloc(size_t size)
{
    void *ret = malloc(size);
    if (!ret) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    return ret;
}

void *
checked_aligned_alloc(size_t align, size_t size)
{
    if (size % align != 0)
        size = (size & ~(align-1)) + align;

    void *ret = aligned_alloc(align, size);
    if (!ret) {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }

    return ret;
}

std::string
size_to_human(size_t size)
{
    std::string suffixes = " kMGTPE";
    int index = 0;
    while (size >= 1024) {
        index++;
        size /= 1024;
        /* The division by 1024 is a right shift of 10 bytes. Therefore,
         * with a 64-bit size_t one does at most 6 shifts, so running past
         * the end of suffixes is not possible.
         * Test: size_to_human(~0ULL) == "15E". */
    }

    if (index == 0)
        return std::to_string(size);
    else
        return std::to_string(size) + suffixes[index];
}

} //namespace utils