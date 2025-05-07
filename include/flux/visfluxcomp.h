#ifndef VISFLUXCOMP_H
#define VISFLUXCOMP_H

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream> 

#include <omp.h>
#include "io/meshread.h"
#include "io/initialize.h"
#include "flux/fluxcomp.h"

using namespace Eigen;
using std::max;
using std::abs;

void compute_fluxes_vis(const MeshData &mesh,
                        const MatrixXd &Q_L,
                        const MatrixXd &Q_R,
                        const MatrixXd &dQx,
                        const MatrixXd &dQy,
                        const Flow &flow,
                        const Flux &flux,
                        MatrixXd &F,
                        VectorXd &s_max_all,
                        MatrixXd &F_viscous,
                        MatrixXd &Q_f,
                        MatrixXd &dQ_fx,
                        MatrixXd &dQ_fy,
                        VectorXd &dVdn)
{
    // Inviscid part
    compute_fluxes(mesh, Q_L, Q_R, flow, flux, F, s_max_all);

    F_viscous.setZero();
    Q_f.setZero();
    dQ_fx.setZero();
    dQ_fy.setZero();
      
    int j = 0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;
        int c2 = mesh.f2c(i, 1) - 1;

        double nx = mesh.n_f(i, 0);
        double ny = mesh.n_f(i, 1);

        double dux, duy, dvx, dvy, dTx, dTy;

        Q_f.row(i) = 0.5 * (Q_L.row(i) + Q_R.row(i));
        double rho = Q_f(i, 0);
        double u = Q_f(i, 1) / rho;
        double v = Q_f(i, 2) / rho;
        double E = Q_f(i, 3);
        double p = (E - 0.5 * rho * (u * u + v * v)) * (flow.gamma - 1.0);

        if (c2 >= 0) {
            dQ_fx.row(i) = 0.5 * (dQx.row(c1) + dQx.row(c2));
            dQ_fy.row(i) = 0.5 * (dQy.row(c1) + dQy.row(c2));

            double drhox = dQ_fx(i, 0);
            double drhoy = dQ_fy(i, 0);
            dux = (dQ_fx(i, 1) - u * drhox) / rho;
            duy = (dQ_fy(i, 1) - u * drhoy) / rho;
            dvx = (dQ_fx(i, 2) - v * drhox) / rho;
            dvy = (dQ_fy(i, 2) - v * drhoy) / rho;
    
            dTx = ((dQ_fx(i, 3) - u * dQ_fx(i, 1) - v * dQ_fx(i, 2)) * (flow.gamma - 1.0) - p * drhox / rho) / (flow.R * rho);
            dTy = ((dQ_fy(i, 3) - u * dQ_fy(i, 1) - v * dQ_fy(i, 2)) * (flow.gamma - 1.0) - p * drhoy / rho) / (flow.R * rho);
        } else {
            dQ_fx.row(i) = dQx.row(c1);
            dQ_fy.row(i) = dQy.row(c1);

            double drhox = dQ_fx(i, 0);
            double drhoy = dQ_fy(i, 0);
            dux = (dQ_fx(i, 1) - u * drhox) / rho;
            duy = (dQ_fy(i, 1) - u * drhoy) / rho;
            dvx = (dQ_fx(i, 2) - v * drhox) / rho;
            dvy = (dQ_fy(i, 2) - v * drhoy) / rho;
    
            dTx = ((dQ_fx(i, 3) - u * dQ_fx(i, 1) - v * dQ_fx(i, 2)) * (flow.gamma - 1.0) - p * drhox / rho) / (flow.R * rho);
            dTy = ((dQ_fy(i, 3) - u * dQ_fy(i, 1) - v * dQ_fy(i, 2)) * (flow.gamma - 1.0) - p * drhoy / rho) / (flow.R * rho);
        }

        // Wall boundary: Override gradients
        if (c2 == -1) {
            double dx = mesh.r_c(c1, 0) - mesh.r_f(i, 0);
            double dy = mesh.r_c(c1, 1) - mesh.r_f(i, 1);
            double mag = std::sqrt(dx * dx + dy * dy);

            double uL = Q_L(i, 1) / Q_L(i, 0);
            double vL = Q_L(i, 2) / Q_L(i, 0);

            double dun = -uL / mag;
            double dvn = -vL / mag;
            double VL = uL * nx + vL * ny;

            #pragma omp critical
            {
                dVdn(j++) = -VL / mag;
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
        F_viscous(i, 0) = 0.0;
        F_viscous(i, 1) = Txx * nx + Txy * ny;
        F_viscous(i, 2) = Txy * nx + Tyy * ny;
        F_viscous(i, 3) = (u * (Txx * nx + Txy * ny) +
                           v * (Txy * nx + Tyy * ny) +
                           flow.k * (dTx * nx + dTy * ny));
    }

    // Final flux = inviscid - viscous (note: some literature writes it as inviscid + viscous)
    F -= F_viscous;
}

#endif  // VISFLUXCOMP_H
