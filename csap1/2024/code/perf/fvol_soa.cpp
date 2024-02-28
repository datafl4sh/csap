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
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>

#ifdef HAVE_SILO
#include <silo.h>
#endif

#include "utils.h"

using vector_t = std::vector<double>;

class fields {
    vector_t    field_vx;
    vector_t    field_vy;
    vector_t    field_p;
    size_t      N;

public:
    fields();
    fields(size_t);

    double      vx(size_t, size_t) const;
    double&     vx(size_t, size_t);
    double      vy(size_t, size_t) const;
    double&     vy(size_t, size_t);
    double      p(size_t, size_t) const;
    double&     p(size_t, size_t);

    const double *vx_data() const {
        return field_vx.data();
    }

    const double *vy_data() const {
        return field_vy.data();
    }

    const double *p_data() const {
        return field_p.data();
    }

    size_t size() const {
        return N;
    }

    double cell_volume() const {
        return (1./N)*(1./N);
    }

    double face_area() const {
        return (1./N);
    }

    auto center(size_t i, size_t j) const {
        auto cx = (0.5 + i)/N;
        auto cy = (0.5 + j)/N;
        return std::pair(cx, cy);
    }

    fields& operator+=(const fields& other)
    {
        assert(size() == other.size());
        for (size_t i = 0; i < field_vx.size(); i++)
            field_vx[i] += other.field_vx[i];

        for (size_t i = 0; i < field_vy.size(); i++)
            field_vy[i] += other.field_vy[i];

        for (size_t i = 0; i < field_p.size(); i++)
            field_p[i] += other.field_p[i];

        return *this;
    }

    fields& operator*=(double v) {
        for (size_t i = 0; i < field_vx.size(); i++)
            field_vx[i] *= v;

        for (size_t i = 0; i < field_vy.size(); i++)
            field_vy[i] *= v;

        for (size_t i = 0; i < field_p.size(); i++)
            field_p[i] *= v;

        return *this;
    }
};

fields operator+(const fields& a, const fields& b) {
    assert(a.size() == b.size());
    fields ret = a;
    ret += b;
    return ret;
}

fields operator*(double v, const fields& f) {
    fields ret = f;
    ret *= v;
    return ret;
}

fields::fields(size_t pN)
    : N(pN) {
    field_vx.resize(N*N);
    field_vy.resize(N*N);
    field_p.resize(N*N);
}

double fields::vx(size_t i, size_t j) const {
    assert(i < N and j < N);
    return field_vx[N*i+j];
}

double& fields::vx(size_t i, size_t j) {
    assert(i < N and j < N);
    return field_vx[N*i+j];
}

double fields::vy(size_t i, size_t j) const {
    assert(i < N and j < N);
    return field_vy[N*i+j];
}

double& fields::vy(size_t i, size_t j) {
    assert(i < N and j < N);
    return field_vy[N*i+j];
}

double fields::p(size_t i, size_t j) const {
    assert(i < N and j < N);
    return field_p[N*i+j];
}

double& fields::p(size_t i, size_t j) {
    assert(i < N and j < N);
    return field_p[N*i+j];
}

void init_fields(fields& f) {
    for (size_t i = 0; i < f.size(); i++) {
        for (size_t j = 0; j < f.size(); j++) {
            auto [cx, cy] = f.center(i,j);
            //f.p(i,j) = std::sin(M_PI*cx) * std::sin(M_PI*cy);
            f.p(i,j) = std::exp( -((cx-0.3)*(cx-0.3) + (cy-0.2)*(cy-0.2))*1000 );
        }
    }
}

void apply_operator(fields& out, const fields& in) {

    const double alpha = 1.0;

    const double vol = in.cell_volume();
    const double area = in.face_area();


    /* Cells NOT on a boundary */
    for (size_t i = 0; i < in.size(); i++) {
        for (size_t j = 0; j < in.size(); j++) {
            double flux_vx = 0.0;
            double flux_vy = 0.0;
            double flux_p = 0.0;

            /* North */
            {
                //const double nx = 0.0;
                const double ny = 1.0;
                if (i+1 < in.size()) {
                    //const double avg_vx = in.vx(i,j) + in.vx(i+1,j);
                    const double avg_vy = in.vy(i,j) + in.vy(i+1,j);
                    const double avg_p = in.p(i,j) + in.p(i+1,j);
                
                    //const double jmp_vx = in.vx(i,j) - in.vx(i+1,j);
                    const double jmp_vy = in.vy(i,j) - in.vy(i+1,j);
                    const double jmp_p = in.p(i,j) - in.p(i+1,j);

                    /* Commented things do nothing because of zero normal component */
                    //flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (/*nx*avg_vx*/ + ny*avg_vy + alpha*jmp_p);
                }
                else {
                    //const double vx = in.vx(i,j);
                    const double vy = in.vy(i,j);
                    const double p = in.p(i,j);
                    flux_p += (area/vol) * ( /*nx*vx*/ + ny*vy + alpha*p );
                }
            }

            /* West */
            {
                const double nx = -1.0;
                //const double ny = 0.0;
                if (j > 0) {
                    const double avg_vx = in.vx(i,j) + in.vx(i,j-1);
                    //const double avg_vy = in.vy(i,j) + in.vy(i,j-1);
                    const double avg_p = in.p(i,j) + in.p(i,j-1);
                
                    const double jmp_vx = in.vx(i,j) - in.vx(i,j-1);
                    //const double jmp_vy = in.vy(i,j) - in.vy(i,j-1);
                    const double jmp_p = in.p(i,j) - in.p(i,j-1);

                    /* Commented things do nothing because of zero normal component */
                    flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    //flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (nx*avg_vx + /*ny*avg_vy*/ + alpha*jmp_p);
                } else {
                    const double vx = in.vx(i,j);
                    //const double vy = in.vy(i,j);
                    const double p = in.p(i,j);
                    flux_p += (area/vol) * ( nx*vx + /*ny*vy*/ + alpha*p );
                }
            }

            /* East */
            {
                const double nx = 1.0;
                //const double ny = 0.0;
                if (j+1 < in.size()) {
                    const double avg_vx = in.vx(i,j) + in.vx(i,j+1);
                    //const double avg_vy = in.vy(i,j) + in.vy(i,j+1);
                    const double avg_p = in.p(i,j) + in.p(i,j+1);
                
                    const double jmp_vx = in.vx(i,j) - in.vx(i,j+1);
                    //const double jmp_vy = in.vy(i,j) - in.vy(i,j+1);
                    const double jmp_p = in.p(i,j) - in.p(i,j+1);

                    /* Commented things do nothing because of zero normal component */
                    flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    //flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (nx*avg_vx + /*ny*avg_vy*/ + alpha*jmp_p);
                } else {
                    const double vx = in.vx(i,j);
                    //const double vy = in.vy(i,j);
                    const double p = in.p(i,j);
                    flux_vx += (area/vol) * ( nx*p + /*ny*vy*/ + alpha*vx ); 
                    flux_p += (area/vol) * ( nx*vx + /*ny*vy*/ + alpha*p );
                }
            }

            /* South */
            {
                //const double nx = 0.0;
                const double ny = -1.0;
                if (i > 0) {
                    //const double avg_vx = in.vx(i,j) + in.vx(i-1,j);
                    const double avg_vy = in.vy(i,j) + in.vy(i-1,j);
                    const double avg_p = in.p(i,j) + in.p(i-1,j);
                
                    //const double jmp_vx = in.vx(i,j) - in.vx(i-1,j);
                    const double jmp_vy = in.vy(i,j) - in.vy(i-1,j);
                    const double jmp_p = in.p(i,j) - in.p(i-1,j);

                    /* Commented things do nothing because of zero normal component */
                    //flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (/*nx*avg_vx*/ + ny*avg_vy + alpha*jmp_p);
                } else {
                    //const double vx = in.vx(i,j);
                    const double vy = in.vy(i,j);
                    const double p = in.p(i,j);
                    flux_p += (area/vol) * ( /*nx*vx*/ + ny*vy + alpha*p );
                }
            }

            out.vx(i,j) = -flux_vx;
            out.vy(i,j) = -flux_vy;
            out.p(i,j) = -flux_p;
        }
    }
}

#ifdef HAVE_SILO
bool export_to_visit(fields& f, int t, double dt)
{
    std::stringstream fn_ss;
    fn_ss << "wave_" << t << ".silo";

    DBfile *db = DBCreate(fn_ss.str().c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
    if (!db)
        return false;
    DBoptlist *opts = DBMakeOptlist(2);
    DBAddOption(opts, DBOPT_CYCLE, &t);
    double time = t*dt;
    DBAddOption(opts, DBOPT_DTIME, &time);

    int N = f.size();
    std::vector<double> xs(N+1);
    std::vector<double> ys(N+1);
    for (size_t i = 0; i < N+1; i++) {
        xs[i] = double(i)/N;
        ys[i] = double(i)/N;
    }
    int mshdims[] = {N+1,N+1};
    int ndims = 2;
    double *coords[] = {xs.data(), ys.data()};

    DBPutQuadmesh(db, "mesh", nullptr, coords, mshdims, ndims, DB_DOUBLE, DB_COLLINEAR, opts);
    int vardims[] = {N,N};
    DBPutQuadvar1(db, "p", "mesh", f.p_data(), vardims, ndims, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
    DBPutQuadvar1(db, "vx", "mesh", f.vx_data(), vardims, ndims, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
    DBPutQuadvar1(db, "vy", "mesh", f.vy_data(), vardims, ndims, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
    const char *names[] = {"v"};
    const char *defs[] = {"{vx, vy}"};
    int types[] = {DB_VARTYPE_VECTOR};
    DBPutDefvars(db, "defvars", 1, names, types, defs, nullptr);

    DBFreeOptlist(opts);
    DBClose(db);
    return true;
}
#endif

void rk4_step(fields& out, const fields& curr, double dt, const fields& k)
{
    assert(out.size() == curr.size());
    assert(curr.size() == k.size());
    size_t N = out.size();
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.p(i,j) = curr.p(i,j) + dt*k.p(i,j);
    
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.vx(i,j) = curr.vx(i,j) + dt*k.vx(i,j);

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.vy(i,j) = curr.vy(i,j) + dt*k.vy(i,j);
}

void rk4_wsum(fields& out, const fields& curr, double dt, const fields& k1,
    const fields& k2, const fields& k3, const fields& k4)
{
    size_t N = out.size();
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.p(i,j) = curr.p(i,j) + (dt/6.0)*(k1.p(i,j) + 2.0*(k2.p(i,j)+k3.p(i,j)) + k4.p(i,j));
    
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.vx(i,j) = curr.vx(i,j) + (dt/6.0)*(k1.vx(i,j) + 2.0*(k2.vx(i,j)+k3.vx(i,j)) + k4.vx(i,j));

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.vy(i,j) = curr.vy(i,j) + (dt/6.0)*(k1.vy(i,j) + 2.0*(k2.vy(i,j)+k3.vy(i,j)) + k4.vy(i,j));
}

int main(void)
{
    const double dt = 0.001;
    const size_t N = 256;
    const int maxiter = 500;
    const int dump_rate = 10;

    fields curr(N);
    fields next(N);
    fields k1(N);
    fields k2(N);
    fields k3(N);
    fields k4(N);
    fields tmp(N);

    init_fields(curr);

    double itertime = 0.0;

    for (int t = 0; t < maxiter; t++) {
        std::cout << "Timestep " << t << "\r" << std::flush;
        
        utils::clock clk;
        clk.start();

        apply_operator(k1, curr);

        tmp = curr + 0.5*dt*k1;
        //rk4_step(tmp, curr, 0.5*dt, k1);
        apply_operator(k2, tmp);

        tmp = curr + 0.5*dt*k2;
        //rk4_step(tmp, curr, 0.5*dt, k1);
        apply_operator(k3, tmp);

        tmp = curr + dt*k3;
        //rk4_step(tmp, curr, dt, k1);
        apply_operator(k4, tmp);

        next = curr + (dt/6.0)*(k1 + 2.0*(k2+k3) + k4);
        //rk4_wsum(next, curr, dt, k1, k2, k3, k4);
        
        itertime += clk.stop();

        #ifdef HAVE_SILO
        if ( (t%dump_rate) == 0 ) {
            export_to_visit(curr, t, dt);
        }
        #endif

        curr = next;
    }
    std::cout << std::endl;

    double avgitertime = itertime/maxiter;

    std::cout << "Average iteration time: " << avgitertime;
    std::cout << "s, DoFs/s: " << 3*N*N/avgitertime << std::endl;
}