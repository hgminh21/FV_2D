#ifndef RECONLS_H
#define RECONLS_H

#include <fstream>
#include <iostream>

#include "io/meshread.h"
#include "io/initialize.h"
#include "limiter/squeeze.h"
#include "limiter/venkat.h"
#include "limiter/vanleer.h"
#include "explicit/varini.h"


void reconstruct_leastsquare(const MeshData &mesh,
                            const std::vector<double> &Q,
                            const std::vector<double> &Q_in,
                            const Flow &flow,
                            const Solver &solver,
                            const Reconstruct &recon,
                            reconVars &rv,
                            reconScraps &rs)
{
    // reset accumulators and limiter
    std::fill(rs.Qx1_temp.begin(), rs.Qx1_temp.end(), 0.0);
    std::fill(rs.Qx2_temp.begin(), rs.Qx2_temp.end(), 0.0);
    std::fill(rs.Qy1_temp.begin(), rs.Qy1_temp.end(), 0.0);
    std::fill(rs.Qy2_temp.begin(), rs.Qy2_temp.end(), 0.0);
    rs.Q_max = Q;
    rs.Q_min = Q;
    std::fill(rs.phi.begin(), rs.phi.end(), 2.0);


    // limiter trigger
    // 0 = no limiter, 1 = use limiter
    int lim_trigger;
    if (recon.use_lim == "nolim") {lim_trigger = 0;}
    else {lim_trigger = 1;}

    // Second-order reconstruction
    // accumulate gradient contributions
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;
        // geometry, avoid Vector2d temporaries
        double x1 = mesh.r_c[2*c1], y1 = mesh.r_c[2*c1+1];
        double xf = mesh.r_f[2*i], yf = mesh.r_f[2*i+1];
        double nx = mesh.n_f[2*i], ny = mesh.n_f[2*i+1];
        // cache states
        std::vector<double> Q1(4, 0.0), Q2(4, 0.0);
        Q1[0] = Q[4*c1];
        Q1[1] = Q[4*c1+1];
        Q1[2] = Q[4*c1+2];
        Q1[3] = Q[4*c1+3];
        if (c2 >= 0) {
            Q2[0] = Q[4*c2];
            Q2[1] = Q[4*c2+1];
            Q2[2] = Q[4*c2+2];
            Q2[3] = Q[4*c2+3];
        }
        // compute difference and offsets
        std::vector<double> dQ(4, 0.0);
        double dx, dy;
        if (c2 >= 0) {
            dx = mesh.r_c[2*c2] - x1;
            dy = mesh.r_c[2*c2+1] - y1;
            dQ[0] = Q2[0] - Q1[0];
            dQ[1] = Q2[1] - Q1[1];
            dQ[2] = Q2[2] - Q1[2];
            dQ[3] = Q2[3] - Q1[3];
        } else if (c2 == -2) {
            dx = -2.0*(x1 - xf)*nx;
            dy = -2.0*(y1 - yf)*ny;
            dQ[0] = Q_in[0] - Q1[0];
            dQ[1] = Q_in[1] - Q1[1];
            dQ[2] = Q_in[2] - Q1[2];
            dQ[3] = Q_in[3] - Q1[3];
        } else {
            // wall BC
            double rhoi = Q1[0];
            double ui = Q1[1]/rhoi, vi = Q1[2]/rhoi;
            double pi = (Q1[3] - 0.5*rhoi*(ui*ui+vi*vi))*(flow.gamma-1.0);
            double vn = ui*nx + vi*ny;
            double ug = ui - 2.0*vn*nx;
            double vg = vi - 2.0*vn*ny;
            std::vector<double> Qg(4, 0.0);
            Qg[0] = rhoi; Qg[1] = rhoi*ug; Qg[2] = rhoi*vg;
            Qg[3] = pi/(flow.gamma-1.0) + 0.5*rhoi*(ug*ug+vg*vg);
            dQ[0] = Qg[0] - Q1[0];
            dQ[1] = Qg[1] - Q1[1];
            dQ[2] = Qg[2] - Q1[2];
            dQ[3] = Qg[3] - Q1[3];
            dx = -2.0*(x1 - xf)*nx;
            dy = -2.0*(y1 - yf)*ny;
        }
        // inline update to avoid lambda
        for (int j = 0; j < 4; ++j) {
            rs.Qx1_temp[4*c1+j] += dQ[j] * (dx * mesh.Iyy[c1]);
            rs.Qx2_temp[4*c1+j] += dQ[j] * (dy * mesh.Ixy[c1]);
            rs.Qy1_temp[4*c1+j] += dQ[j] * (dy * mesh.Ixx[c1]);
            rs.Qy2_temp[4*c1+j] += dQ[j] * (dx * mesh.Ixy[c1]);
            if (c2 >= 0) {
                rs.Qx1_temp[4*c2+j] += dQ[j] * (dx * mesh.Iyy[c2]);
                rs.Qx2_temp[4*c2+j] += dQ[j] * (dy * mesh.Ixy[c2]);
                rs.Qy1_temp[4*c2+j] += dQ[j] * (dy * mesh.Ixx[c2]);
                rs.Qy2_temp[4*c2+j] += dQ[j] * (dx * mesh.Ixy[c2]);
            }
        }
        // store min/max for limiters
        if (lim_trigger > 0) {
            for (int j = 0; j < 4; ++j) {
                double Qmax_temp = std::max(Q1[j], rs.Q_max[4*c1+j]);
                double Qmin_temp = std::min(Q1[j], rs.Q_min[4*c1+j]);
                if (c2 >= 0) {
                        rs.Q_max[4*c1+j] = std::max(Qmax_temp, Q2[j]);
                        rs.Q_min[4*c1+j] = std::min(Qmin_temp, Q2[j]);
                        // rs.Q_max[4*c2+j] = rs.Q_max[4*c1+j]; // this might be not true
                        // rs.Q_min[4*c2+j] = rs.Q_min[4*c1+j];
                }
                else if (c2 == -1) { // using Qg from wall BC above
                        rs.Q_max[4*c1+j] = std::max(Qmax_temp, dQ[j]+Q1[j]);
                        rs.Q_min[4*c1+j] = std::min(Qmin_temp, dQ[j]+Q1[j]);
                }
                else { // free-stream
                        rs.Q_max[4*c1+j] = std::max(Qmax_temp, Q_in[j]);
                        rs.Q_min[4*c1+j] = std::min(Qmin_temp, Q_in[j]);
                }
            }
        }
    }

    // final gradient
    for (int i = 0; i < mesh.n_cells; ++i) {
        // compute dQx, dQy
        for (int j = 0; j < 4; ++j) {
            rs.dQx[4*i+j] = rs.Qx1_temp[4*i+j] - rs.Qx2_temp[4*i+j];
            rs.dQy[4*i+j] = rs.Qy1_temp[4*i+j] - rs.Qy2_temp[4*i+j];
        }
    }
    
    // compute left/right states
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i]-1;
        int c2 = mesh.f2c[2*i+1]-1;
        double x1 = mesh.r_c[2*c1], y1 = mesh.r_c[2*c1+1];
        double xf = mesh.r_f[2*i], yf = mesh.r_f[2*i+1];
        // left
        double dfx1 = xf - x1, dfy1 = yf - y1;
        for (int j = 0; j < 4; ++j) {
            rv.Q_L[4*i+j] = Q[4*c1+j]
                        + rs.dQx[4*c1+j] * dfx1
                        + rs.dQy[4*c1+j] * dfy1;
        }
        // right or BC
        if (c2 >= 0) {
            double x2 = mesh.r_c[2*c2], y2 = mesh.r_c[2*c2+1];
            double dfx2 = xf - x2, dfy2 = yf - y2;
            for (int j = 0; j < 4; ++j) {
                rv.Q_R[4*i+j] = Q[4*c2+j]
                            + rs.dQx[4*c2+j] * dfx2
                            + rs.dQy[4*c2+j] * dfy2;
            }
        } else if (c2 == -1) {
            // wall BC same as first-order but on rv.Q_L
            double rhoL = rv.Q_L[4*i];
            double uL = rv.Q_L[4*i+1]/rhoL, vL = rv.Q_L[4*i+2]/rhoL;
            double pL = (rv.Q_L[4*i+3] - 0.5*rhoL*(uL*uL+vL*vL))*(flow.gamma-1.0);
            double nx = mesh.n_f[2*i], ny = mesh.n_f[2*i+1];
            double vn = uL*nx + vL*ny;
            double uR = uL - 2.0*vn*nx;
            double vR = vL - 2.0*vn*ny;
            rv.Q_R[4*i]=rhoL; 
            rv.Q_R[4*i+1]=rhoL*uR;
            rv.Q_R[4*i+2]=rhoL*vR;
            rv.Q_R[4*i+3]=pL/(flow.gamma-1.0)
                            +0.5*rhoL*(uR*uR+vR*vR);
        } else {
            rv.Q_R[4*i] = Q_in[0];
            rv.Q_R[4*i+1] = Q_in[1];
            rv.Q_R[4*i+2] = Q_in[2];
            rv.Q_R[4*i+3] = Q_in[3];
        }
    }

    // apply limiter
    if (lim_trigger > 0) {
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c[2*i]-1;
            int c2 = mesh.f2c[2*i+1]-1;
            std::vector<double> Qmax(4, 0.0), Qmin(4, 0.0), Q1(4, 0.0), Q2(4, 0.0), QL(4, 0.0), QR(4, 0.0), p1(4, 0.0), p2(4, 0.0);
            // cache values
            for (int j = 0; j < 4; ++j) {
                Qmax[j] = rs.Q_max[4*c1+j];
                Qmin[j] = rs.Q_min[4*c1+j];
                Q1[j] = Q[4*c1+j];
                Q2[j] = Q[4*c2+j];
                QL[j] = rv.Q_L[4*i+j];
                QR[j] = rv.Q_R[4*i+j];
                p1[j] = rs.phi[4*c1+j];
                if (c2 >= 0) {p2[j] = rs.phi[4*c2+j];}
            }
            // apply limiter cell 1
            if (recon.use_lim == "squeeze") {
                p1 = squeeze_lim(Qmax, Qmin, Q1, QL, p1, recon);
            } else if (recon.use_lim == "venkat") {
                p1 = venkat_lim(Qmax, Qmin, Q1, QL, p1, recon);
            } else {
                p1 = vanleer_lim(Qmax, Qmin, Q1, QL, p1, recon);
            }
            rs.phi[4*c1] = p1[0];
            rs.phi[4*c1+1] = p1[1];
            rs.phi[4*c1+2] = p1[2];
            rs.phi[4*c1+3] = p1[3];
            // apply limiter cell 2 
            if (c2 < 0) continue; // avoid negative ghost cell index
            else {
                if (recon.use_lim == "squeeze") {
                    p2 = squeeze_lim(Qmax, Qmin, Q2, QR, p2, recon);
                } else if (recon.use_lim == "venkat") {
                    p2 = venkat_lim(Qmax, Qmin, Q2, QR, p2, recon);
                } else {
                    p2 = vanleer_lim(Qmax, Qmin, Q2, QR, p2, recon);
                }
                rs.phi[4*c2] = p2[0];
                rs.phi[4*c2+1] = p2[1];
                rs.phi[4*c2+2] = p2[2];
                rs.phi[4*c2+3] = p2[3];
            }
        }
        // rebuild limited states
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c[2*i] - 1;
            int c2 = mesh.f2c[2*i+1] - 1;
            double x1 = mesh.r_c[2*c1], y1 = mesh.r_c[2*c1+1];
            double xf = mesh.r_f[2*i], yf = mesh.r_f[2*i+1];
            double dfx1 = xf - x1, dfy1 = yf - y1;
            for (int j = 0; j < 4; ++j) {
                // Element-wise multiplication for rv.Q_L
                rv.Q_L[4*i+j] = Q[4*c1+j] + (rs.phi[4*c1+j] * (rs.dQx[4*c1+j] * dfx1 + rs.dQy[4*c1+j] * dfy1));
                if (c2 < 0) continue;  // Skip if c2 is invalid (negative index)
                double x2 = mesh.r_c[2*c2], y2 = mesh.r_c[2*c2+1];
                double dfx2 = xf - x2, dfy2 = yf - y2;
                // Element-wise multiplication for rv.Q_R
                rv.Q_R[4*i+j] = Q[4*c2+j] + (rs.phi[4*c2+j] * (rs.dQx[4*c2+j] * dfx2 + rs.dQy[4*c2+j] * dfy2));
            }
        }
    }
}

#endif // RECONLS_H
