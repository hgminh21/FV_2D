#ifndef VISFLUXCOMP_H
#define VISFLUXCOMP_H

#include <cmath>
#include <algorithm>
#include <iostream> 

#include "io/meshread.h"
#include "io/initialize.h"
#include "flux/fluxcomp.h"
#include "explicit/varini.h"

using std::max;
using std::abs;

void compute_fluxes_vis(const MeshData &mesh,
                        const Flow &flow,
                        const Flux &flux,
                        const reconVars &rv,
                        const reconScraps &rs,
                        fluxVars &fv,
                        fluxScraps &fs,
                        ioVars &iv)
{
    // Inviscid part
    compute_fluxes(mesh, flow, flux, rv, fv);

    std::fill(fs.F_viscous.begin(), fs.F_viscous.end(), 0.0);
    std::fill(fs.Q_f.begin(), fs.Q_f.end(), 0.0);
    std::fill(fs.dQ_fx.begin(), fs.dQ_fx.end(), 0.0);
    std::fill(fs.dQ_fy.begin(), fs.dQ_fy.end(),0.0);
      
    int j = 0;

    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i + 1] - 1;

        double nx = mesh.n_f[2*i];
        double ny = mesh.n_f[2*i + 1];

        double dux, duy, dvx, dvy, dTx, dTy;

        double rho = 0.5 * (rv.Q_L[4*i] + rv.Q_R[4*i]);
        double u = 0.5 * (rv.Q_L[4*i+1] + rv.Q_R[4*i+1]) / rho;
        double v = 0.5 * (rv.Q_L[4*i+2] + rv.Q_R[4*i+2]) / rho;
        double E = 0.5 * (rv.Q_L[4*i+3] + rv.Q_R[4*i+3]);
        double p = (E - 0.5 * rho * (u * u + v * v)) * (flow.gamma - 1.0);
        
        fs.Q_f[4*i] = rho;
        fs.Q_f[4*i+1] = rho * u;
        fs.Q_f[4*i+2] = rho * v;
        fs.Q_f[4*i+3] = E;

        if (c2 >= 0) {
            if (int k = 0; k < 4, ++k) {
                fs.dQ_fx[4*i + k] = 0.5 * (rs.dQx[4*c1 + k] + rs.dQx[4*c2 + k]);
                fs.dQ_fy[4*i + k] = 0.5 * (rs.dQy[4*c1 + k] + rs.dQy[4*c2 + k]);
            }

            double drhox = fs.dQ_fx[4*i];
            double drhoy = fs.dQ_fy[4*i];

            dux = (fs.dQ_fx[4*i + 1] - u * drhox) / rho;
            duy = (fs.dQ_fy[4*i + 1] - u * drhoy) / rho;
            dvx = (fs.dQ_fx[4*i + 2] - v * drhox) / rho;
            dvy = (fs.dQ_fy[4*i + 2] - v * drhoy) / rho;
    
            dTx = ((fs.dQ_fx[4*i+3] - u * fs.dQ_fx[4*i+1] - v * fs.dQ_fx[4*i+2]) * (flow.gamma - 1.0) - p * drhox / rho) / (flow.R * rho);
            dTy = ((fs.dQ_fy[4*i+3] - u * fs.dQ_fy[4*i+1] - v * fs.dQ_fy[4*i+2]) * (flow.gamma - 1.0) - p * drhoy / rho) / (flow.R * rho);
        } else {
            for (int k = 0; k < 4; ++k) {
                fs.dQ_fx[4*i + k] = rs.dQx[4*c1 + k];
                fs.dQ_fy[4*i + k] = rs.dQy[4*c1 + k];
            }

            double drhox = fs.dQ_fx[4*i];
            double drhoy = fs.dQ_fy[4*i];
            dux = (fs.dQ_fx[4*i + 1] - u * drhox) / rho;
            duy = (fs.dQ_fy[4*i + 1] - u * drhoy) / rho;
            dvx = (fs.dQ_fx[4*i + 2] - v * drhox) / rho;
            dvy = (fs.dQ_fy[4*i + 2] - v * drhoy) / rho;
    
            dTx = ((fs.dQ_fx[4*i+3] - u * fs.dQ_fx[4*i+1] - v * fs.dQ_fx[4*i+2]) * (flow.gamma - 1.0) - p * drhox / rho) / (flow.R * rho);
            dTy = ((fs.dQ_fy[4*i+3] - u * fs.dQ_fy[4*i+1] - v * fs.dQ_fy[4*i+2]) * (flow.gamma - 1.0) - p * drhoy / rho) / (flow.R * rho);
        }

        // Wall boundary: Override gradients
        if (c2 == -1) {
            double dx = mesh.r_c[2*c1] - mesh.r_f[2*i];
            double dy = mesh.r_c[2*c1+1] - mesh.r_f[2*i+1];
            double mag = std::sqrt(dx * dx + dy * dy);

            double uL = rv.Q_L[4*i + 1] / rv.Q_L[4*i];
            double vL = rv.Q_L[4*i + 2] / rv.Q_L[4*i];

            double dun = -uL / mag;
            double dvn = -vL / mag;
            double VL = uL * nx + vL * ny;

            {
                iv.dVdn[j++] = -VL / mag;
            }

            dux = dun * nx;
            duy = dun * ny;
            dvx = dvn * nx;
            dvy = dvn * ny;

            dTx = 0.0;
            dTy = 0.0;
        }

        // Compute viscous stress tensor components
        double div_v = dux + dvy;
        double Txx = 2.0 * flow.mu * (dux - div_v / 3.0);
        double Tyy = 2.0 * flow.mu * (dvy - div_v / 3.0);
        double Txy = flow.mu * (duy + dvx);

        // Viscous flux contribution
        fs.F_viscous[4*i] = 0.0;
        fs.F_viscous[4*i + 1] = Txx * nx + Txy * ny;
        fs.F_viscous[4*i + 2] = Txy * nx + Tyy * ny;
        fs.F_viscous[4*i + 3] = (u * (Txx * nx + Txy * ny) +
                                v * (Txy * nx + Tyy * ny) +
                                flow.k * (dTx * nx + dTy * ny));
    }

    // Final flux = inviscid - viscous (note: some literature writes it as inviscid + viscous)
    for (int i = 0; i < mesh.n_faces; ++i) {
        for (int k = 0; k < 4; ++k) {
            fv.F[4*i + k] -= fs.F_viscous[4*i + k];
        }
    }
}

#endif  // VISFLUXCOMP_H
