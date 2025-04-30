#ifndef RECONGG_H
#define RECONGG_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "io/meshread.h"
#include "io/initialize.h"
#include "limiter/squeeze.h"
#include "limiter/venkat.h"

using namespace Eigen;

void reconstruct_gaussgreen(const MeshData &mesh,
                            const MatrixXd &Q,
                            MatrixXd &Q_L,
                            MatrixXd &Q_R,
                            MatrixXd &dQx,
                            MatrixXd &dQy,
                            const Vector4d &Q_in,
                            const Flow &flow,
                            const Solver &solver,
                            const Reconstruct &recon,
                            MatrixXd &Qx1_temp,
                            MatrixXd &Qy1_temp,
                            MatrixXd &Q_max,
                            MatrixXd &Q_min,
                            MatrixXd &phi)
{
    // reset accumulators and limiter
    Qx1_temp.setZero();
    Qy1_temp.setZero();
    Q_max = Q;
    Q_min = Q;
    phi.setOnes();

    // Second-order reconstruction
    // accumulate gradient contributions
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i,0) - 1;
        int c2 = mesh.f2c(i,1) - 1;
        // geometry, avoid Vector2d temporaries
        double x1 = mesh.r_c(c1,0), y1 = mesh.r_c(c1,1);
        double xf = mesh.r_f(i,0), yf = mesh.r_f(i,1);
        double nx = mesh.n_f(i,0), ny = mesh.n_f(i,1);
        double V1 = mesh.V(c1);
        double A = mesh.A(i);

        // cache states
        RowVector4d Q1 = Q.row(c1);
        RowVector4d Q2 = (c2>=0 ? Q.row(c2).eval() : RowVector4d::Zero());
        // compute difference and offsets
        RowVector4d Qf;
        RowVector4d Qg;
        double dx1, dy1, dx2, dy2;
        dx1 = x1 - xf; dy1 = y1 - yf;
        
        if (c2 >= 0) {
            double x2 = mesh.r_c(c2,0), y2 = mesh.r_c(c2,1);
            dx2 = x2 - xf; dy2 = y2 - yf;
            Qf = (Q1 * sqrt(dx1 * dx1 + dy1 * dy1) + Q2 * sqrt(dx2 * dx2 + dy2 * dy2)) / 
                  (sqrt(dx1 * dx1 + dy1 * dy1) + sqrt(dx2 * dx2 + dy2 * dy2));
        } else if (c2 == -2) {
            dx2 = dx1; dy2 = dy1;
            Qf = (Q1 * sqrt(dx1 * dx1 + dy1 * dy1) + Q_in.transpose() * sqrt(dx2 * dx2 + dy2 * dy2)) / 
                  (sqrt(dx1 * dx1 + dy1 * dy1) + sqrt(dx2 * dx2 + dy2 * dy2));
        } else {
            // wall BC
            double rhoi = Q1(0);
            double ui = Q1(1)/rhoi, vi = Q1(2)/rhoi;
            double pi = (Q1(3) - 0.5*rhoi*(ui*ui+vi*vi))*(flow.gamma-1.0);
            double vn = ui*nx + vi*ny;
            double ug = ui - 2.0*vn*nx;
            double vg = vi - 2.0*vn*ny;
            Qg << rhoi,
                    rhoi*ug,
                    rhoi*vg,
                    pi/(flow.gamma-1.0) + 0.5*rhoi*(ug*ug+vg*vg);
            dx2 = dx1; dy2 = dy1;
            Qf = (Q1 * sqrt(dx1 * dx1 + dy1 * dy1) + Qg * sqrt(dx2 * dx2 + dy2 * dy2)) / 
                  (sqrt(dx1 * dx1 + dy1 * dy1) + sqrt(dx2 * dx2 + dy2 * dy2));
        }
        // inline update to avoid lambda
        Qx1_temp.row(c1) += Qf * nx * A / V1;
        Qy1_temp.row(c1) += Qf * ny * A / V1;
        if (c2 >= 0) {
            double V2 = mesh.V(c2);
            Qx1_temp.row(c2) -= Qf * nx * A / V2;
            Qy1_temp.row(c2) -= Qf * ny * A / V2;
        }
        // store min/max for limiters
        if (recon.use_lim > 0) {
            for (int j = 0; j < 4; ++j) {
                double Qmax_temp = std::max(Q1(j), Q_max(c1,j));
                double Qmin_temp = std::min(Q1(j), Q_min(c1,j));
                if (c2 >= 0) {
                        Q_max(c1,j) = std::max(Qmax_temp, Q2(j));
                        Q_min(c1,j) = std::min(Qmin_temp, Q2(j));
                        Q_max(c2,j) = Q_max(c1,j);
                        Q_min(c2,j) = Q_min(c1,j);
                }
                else if (c2 == -1) {
                    // using Qg from wall BC above
                        Q_max(c1,j) = std::max(Qmax_temp, Qg(j));
                        Q_min(c1,j) = std::min(Qmin_temp, Qg(j));
                }
                else { // free-stream
                        Q_max(c1,j) = std::max(Qmax_temp, Q_in(j));
                        Q_min(c1,j) = std::min(Qmin_temp, Q_in(j));
                }
            }
        }
    }

    // final gradient
    dQx = Qx1_temp;
    dQy = Qy1_temp;

    // compute left/right states
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i,0)-1;
        int c2 = mesh.f2c(i,1)-1;
        double x1 = mesh.r_c(c1,0), y1 = mesh.r_c(c1,1);
        double xf = mesh.r_f(i,0), yf = mesh.r_f(i,1);
        // left
        double dfx1 = xf - x1, dfy1 = yf - y1;
        Q_L.row(i) = Q.row(c1)
                    + dQx.row(c1) * dfx1
                    + dQy.row(c1) * dfy1;
        // right or BC
        if (c2 >= 0) {
            double x2 = mesh.r_c(c2,0), y2 = mesh.r_c(c2,1);
            double dfx2 = xf - x2, dfy2 = yf - y2;
            Q_R.row(i) = Q.row(c2)
                        + dQx.row(c2) * dfx2
                        + dQy.row(c2) * dfy2;
        } else if (c2 == -1) {
            // wall BC same as first-order but on Q_L
            double rhoL = Q_L(i,0);
            double uL = Q_L(i,1)/rhoL, vL = Q_L(i,2)/rhoL;
            double pL = (Q_L(i,3) - 0.5*rhoL*(uL*uL+vL*vL))*(flow.gamma-1.0);
            double nx = mesh.n_f(i,0), ny = mesh.n_f(i,1);
            double vn = uL*nx + vL*ny;
            double uR = uL - 2.0*vn*nx;
            double vR = vL - 2.0*vn*ny;
            Q_R(i,0)=rhoL; Q_R(i,1)=rhoL*uR;
            Q_R(i,2)=rhoL*vR;
            Q_R(i,3)=pL/(flow.gamma-1.0)
                            +0.5*rhoL*(uR*uR+vR*vR);
        } else {
            Q_R.row(i) = Q_in.transpose();
        }
    }

    // apply limiter
    if (recon.use_lim > 0) {
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i,0)-1;
            int c2 = mesh.f2c(i,1)-1;
            if (c2 < 0) continue;
            if (recon.use_lim == 1) {
                phi.row(c1) = squeeze_lim(Q_max.row(c1), Q_min.row(c1), Q.row(c1), Q_L.row(i), phi.row(c1));
                phi.row(c2) = squeeze_lim(Q_max.row(c2), Q_min.row(c2), Q.row(c2), Q_R.row(i), phi.row(c2));
            } else {
                phi.row(c1) = venkat_lim(Q_max.row(c1), Q_min.row(c1), Q.row(c1), Q_L.row(i), phi.row(c1));
                phi.row(c2) = venkat_lim(Q_max.row(c2), Q_min.row(c2), Q.row(c2), Q_R.row(i), phi.row(c2));
            }
        }
        // rebuild limited states
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;
            
            double x1 = mesh.r_c(c1, 0), y1 = mesh.r_c(c1, 1);
            double xf = mesh.r_f(i, 0), yf = mesh.r_f(i, 1);
            double dfx1 = xf - x1, dfy1 = yf - y1;
        
            // Element-wise multiplication for Q_L
            Q_L.row(i) = Q.row(c1) + (phi.row(c1).array() * (dQx.row(c1).array() * dfx1 + dQy.row(c1).array() * dfy1).array()).matrix();
            
            if (c2 < 0) continue;  // Skip if c2 is invalid (negative index)
        
            double x2 = mesh.r_c(c2, 0), y2 = mesh.r_c(c2, 1);
            double dfx2 = xf - x2, dfy2 = yf - y2;
        
            // Element-wise multiplication for Q_R
            Q_R.row(i) = Q.row(c2) + (phi.row(c2).array() * (dQx.row(c2).array() * dfx2 + dQy.row(c2).array() * dfy2).array()).matrix();
        }
    }
}

#endif // RECONGG_H
