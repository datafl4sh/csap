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

/* Copie, divisioni, bad speculation */

#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>

#ifdef HAVE_SILO
#include <silo.h>
#endif

#include "utils.h"

struct field_components {
    double vx;
    double vy;
    double p;

    field_components& operator+=(const field_components& other)
    {
        vx += other.vx;
        vy += other.vy;
        p += other.p;
        return *this;
    }

    field_components& operator*=(double v)
    {
        vx *= v;
        vy *= v;
        p *= v;
        return *this;
    }
};


field_components operator+(const field_components& a, const field_components& b)
{
    field_components ret = a;
    ret += b;
    return ret;
}

field_components operator*(const field_components& f, double v)
{
    field_components ret = f;
    ret *= v;
    return ret;
}

field_components operator*(double v, const field_components& f)
{
    field_components ret = f;
    ret *= v;
    return ret;
}

class fields {
    std::vector<field_components> fcs;
    size_t N;

public:
    fields();
    fields(size_t);

    field_components at(size_t i, size_t j) const {
        assert(i < N and j < N);
        return fcs[N*i+j];
    }

    field_components& at(size_t i, size_t j) {
        assert(i < N and j < N);
        return fcs[N*i+j];
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
        for (size_t i = 0; i < fcs.size(); i++)
            fcs[i] += other.fcs[i];
        return *this;
    }

    fields& operator*=(double v)
    {
        for (size_t i = 0; i < fcs.size(); i++)
            fcs[i] *= v;
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
    fcs.resize(N*N);
}

void init_fields(fields& f) {
    for (size_t i = 0; i < f.size(); i++) {
        for (size_t j = 0; j < f.size(); j++) {
            auto [cx, cy] = f.center(i,j);
            //f.at(i,j).p = std::sin(M_PI*cx) * std::sin(M_PI*cy);
            f.at(i,j).p = std::exp( -((cx-0.3)*(cx-0.3) + (cy-0.2)*(cy-0.2))*1000 );
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
                    //const double avg_vx = in.at(i,j).vx + in.at(i+1,j).vx;
                    const double avg_vy = in.at(i,j).vy + in.at(i+1,j).vy;
                    const double avg_p = in.at(i,j).p + in.at(i+1,j).p;
                
                    //const double jmp_vx = in.at(i,j).vx - in.at(i+1,j).vx;
                    const double jmp_vy = in.at(i,j).vy - in.at(i+1,j).vy;
                    const double jmp_p = in.at(i,j).p - in.at(i+1,j).p;

                    /* Commented things do nothing because of zero normal component */
                    //flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (/*nx*avg_vx*/ + ny*avg_vy + alpha*jmp_p);
                }
                else {
                    //const double vx = in.at(i,j).vx;
                    const double vy = in.at(i,j).vy;
                    const double p = in.at(i,j).p;
                    flux_p += (area/vol) * ( /*nx*vx*/ + ny*vy + alpha*p );
                }
            }

            /* West */
            {
                const double nx = -1.0;
                //const double ny = 0.0;
                if (j > 0) {
                    const double avg_vx = in.at(i,j).vx + in.at(i,j-1).vx;
                    //const double avg_vy = in.at(i,j).vy + in.at(i,j-1).vy;
                    const double avg_p = in.at(i,j).p + in.at(i,j-1).p;
                
                    const double jmp_vx = in.at(i,j).vx - in.at(i,j-1).vx;
                    //const double jmp_vy = in.at(i,j).vy - in.at(i,j-1).vy;
                    const double jmp_p = in.at(i,j).p - in.at(i,j-1).p;

                    /* Commented things do nothing because of zero normal component */
                    flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    //flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (nx*avg_vx + /*ny*avg_vy*/ + alpha*jmp_p);
                } else {
                    const double vx = in.at(i,j).vx;
                    //const double vy = in.at(i,j).vy;
                    const double p = in.at(i,j).p;
                    flux_p += (area/vol) * ( nx*vx + /*ny*vy*/ + alpha*p );
                }
            }

            /* East */
            {
                const double nx = 1.0;
                //const double ny = 0.0;
                if (j+1 < in.size()) {
                    const double avg_vx = in.at(i,j).vx + in.at(i,j+1).vx;
                    //const double avg_vy = in.at(i,j).vy + in.at(i,j+1).vy.vy;
                    const double avg_p = in.at(i,j).p + in.at(i,j+1).p;
                
                    const double jmp_vx = in.at(i,j).vx - in.at(i,j+1).vx;
                    //const double jmp_vy = in.at(i,j).vy - in.at(i,j+1).vy;
                    const double jmp_p = in.at(i,j).p - in.at(i,j+1).p;

                    /* Commented things do nothing because of zero normal component */
                    flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    //flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (nx*avg_vx + /*ny*avg_vy*/ + alpha*jmp_p);
                } else {
                    const double vx = in.at(i,j).vx;
                    //const double vy = in.at(i,j).vy;
                    const double p = in.at(i,j).p;
                    flux_vx += (area/vol) * ( nx*p + /*ny*vy*/ + alpha*vx ); 
                    flux_p += (area/vol) * ( nx*vx + /*ny*vy*/ + alpha*p );
                }
            }

            /* South */
            {
                //const double nx = 0.0;
                const double ny = -1.0;
                if (i > 0) {
                    //const double avg_vx = in.at(i,j).vx + in.at(i-1,j).vx;
                    const double avg_vy = in.at(i,j).vy + in.at(i-1,j).vy;
                    const double avg_p = in.at(i,j).p + in.at(i-1,j).p;
                
                    //const double jmp_vx = in.at(i,j).vx - in.at(i-1,j).vx;
                    const double jmp_vy = in.at(i,j).vy - in.at(i-1,j).vy;
                    const double jmp_p = in.at(i,j).p - in.at(i-1,j).p;

                    /* Commented things do nothing because of zero normal component */
                    //flux_vx += (0.5*area/vol) * (nx*avg_p + alpha*jmp_vx);
                    flux_vy += (0.5*area/vol) * (ny*avg_p + alpha*jmp_vy);
                    flux_p += (0.5*area/vol) * (/*nx*avg_vx*/ + ny*avg_vy + alpha*jmp_p);
                } else {
                    //const double vx = in.at(i,j).vx;
                    const double vy = in.at(i,j).vy;
                    const double p = in.at(i,j).p;
                    flux_p += (area/vol) * ( /*nx*vx*/ + ny*vy + alpha*p );
                }
            }

            out.at(i,j).vx = -flux_vx;
            out.at(i,j).vy = -flux_vy;
            out.at(i,j).p = -flux_p;
        }
    }
}

#ifdef HAVE_SILO
bool export_to_visit(fields& f, int t, double dt)
{
    int N = f.size();

    std::vector<double> p(N*N);
    std::vector<double> vx(N*N);
    std::vector<double> vy(N*N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            p[i*N+j] = f.at(i,j).p;
            vx[i*N+j] = f.at(i,j).vx;
            vy[i*N+j] = f.at(i,j).vy;
        }
    }

    std::stringstream fn_ss;
    fn_ss << "wave_" << t << ".silo";

    DBfile *db = DBCreate(fn_ss.str().c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
    if (!db)
        return false;
    DBoptlist *opts = DBMakeOptlist(2);
    DBAddOption(opts, DBOPT_CYCLE, &t);
    double time = t*dt;
    DBAddOption(opts, DBOPT_DTIME, &time);

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
    DBPutQuadvar1(db, "p", "mesh", p.data(), vardims, ndims, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
    DBPutQuadvar1(db, "vx", "mesh", vx.data(), vardims, ndims, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
    DBPutQuadvar1(db, "vy", "mesh", vy.data(), vardims, ndims, nullptr, 0, DB_DOUBLE, DB_ZONECENT, nullptr);
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
            out.at(i,j) = curr.at(i,j) + dt*k.at(i,j);
}

void rk4_wsum(fields& out, const fields& curr, double dt, const fields& k1,
    const fields& k2, const fields& k3, const fields& k4)
{
    size_t N = out.size();
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            out.at(i,j) = curr.at(i,j) + (dt/6.0)*(k1.at(i,j) + 2.0*(k2.at(i,j)+k3.at(i,j)) + k4.at(i,j));
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
        apply_operator(k1, curr); //14*4

        tmp = curr + 0.5*dt*k1;
        //rk4_step(tmp, curr, 0.5*dt, k1); //3*2
        apply_operator(k2, tmp); //14*4

        tmp = curr + 0.5*dt*k2;
        //rk4_step(tmp, curr, 0.5*dt, k2); //3*2
        apply_operator(k3, tmp); //14*4

        tmp = curr + dt*k3;
        //rk4_step(tmp, curr, dt, k3); //3*2
        apply_operator(k4, tmp); //14*4
        
        next = curr + (dt/6.0)*(k1 + 2.0*(k2+k3) + k4);
        //rk4_wsum(next, curr, dt, k1, k2, k3, k4); //7*3
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