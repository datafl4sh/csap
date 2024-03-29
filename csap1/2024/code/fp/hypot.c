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


#include <stdio.h>
#include <math.h>
#include <float.h>

static double
hypot_naive(double x, double y)
{
    return sqrt(x * x + y * y);
}

static double
hypot_ok(double x, double y)
{
    x = fabs(x);
    y = fabs(y);
    double max = fmax(x, y);
    double min = fmin(x, y);
    double r = min / max;
    return max * sqrt(1 + r * r);
}

int main(void)
{
    double x = DBL_MAX/2, y = x;
    double res = DBL_MAX/sqrt(2.0);
    double nh = hypot_naive(x,y);
    double bh = hypot_ok(x,y);
    double err = 100.0*fabs(bh - res)/fabs(res);
    printf("Naive: %g\n", nh);
    printf("Ok:    %g, error: %g\n", bh, err);
    return 0;
}

