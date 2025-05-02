#ifndef RECONLS_H
#define RECONLS_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include <omp.h>
#include "io/meshread.h"
#include "io/initialize.h"
#include "limiter/squeeze.h"
#include "limiter/venkat.h"

using namespace Eigen;

void reconstruct_leastsquare(const MeshData &mesh,
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
                            MatrixXd &Qx2_temp,
                            MatrixXd &Qy1_temp,
                            MatrixXd &Qy2_temp,
                            MatrixXd &Q_max,
                            MatrixXd &Q_min,
                            MatrixXd &phi)
{
    // reset accumulators and limiter
    Qx1_temp.setZero();
    Qx2_temp.setZero();
    Qy1_temp.setZero();
    Qy2_temp.setZero();
    Q_max = Q;
    Q_min = Q;
    phi.setOnes();

    #pragma omp parallel for
    // Second-order reconstruction
    // accumulate gradient contributions
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i,0) - 1;
        int c2 = mesh.f2c(i,1) - 1;
        // geometry, avoid Vector2d temporaries
        double x1 = mesh.r_c(c1,0), y1 = mesh.r_c(c1,1);
        double xf = mesh.r_f(i,0), yf = mesh.r_f(i,1);
        double nx = mesh.n_f(i,0), ny = mesh.n_f(i,1);
        // cache states
        RowVector4d Q1 = Q.row(c1);
        RowVector4d Q2 = (c2>=0 ? Q.row(c2).eval() : RowVector4d::Zero());
        // compute difference and offsets
        RowVector4d dQ;
        double dx, dy;
        if (c2 >= 0) {
            dx = mesh.r_c(c2,0) - x1;
            dy = mesh.r_c(c2,1) - y1;
            dQ = Q2 - Q1;
        } else if (c2 == -2) {
            dx = -2.0*(x1 - xf)*nx;
            dy = -2.0*(y1 - yf)*ny;
            dQ = Q_in.transpose() - Q1;
        } else {
            // wall BC
            double rhoi = Q1(0);
            double ui = Q1(1)/rhoi, vi = Q1(2)/rhoi;
            double pi = (Q1(3) - 0.5*rhoi*(ui*ui+vi*vi))*(flow.gamma-1.0);
            double vn = ui*nx + vi*ny;
            double ug = ui - 2.0*vn*nx;
            double vg = vi - 2.0*vn*ny;
            RowVector4d Qg;
            Qg << rhoi,
                    rhoi*ug,
                    rhoi*vg,
                    pi/(flow.gamma-1.0) + 0.5*rhoi*(ug*ug+vg*vg);
            dQ = Qg - Q1;
            dx = -2.0*(x1 - xf)*nx;
            dy = -2.0*(y1 - yf)*ny;
        }
        // inline update to avoid lambda
        for (int j = 0; j < 4; ++j) {
            #pragma omp atomic
            Qx1_temp(c1, j) += dQ(j) * (dx * mesh.Iyy(c1));
    
            #pragma omp atomic
            Qx2_temp(c1, j) += dQ(j) * (dy * mesh.Ixy(c1));
    
            #pragma omp atomic
            Qy1_temp(c1, j) += dQ(j) * (dy * mesh.Ixx(c1));
    
            #pragma omp atomic
            Qy2_temp(c1, j) += dQ(j) * (dx * mesh.Ixy(c1));
    
            if (c2 >= 0) {
                #pragma omp atomic
                Qx1_temp(c2, j) += dQ(j) * (dx * mesh.Iyy(c2));
                #pragma omp atomic
                Qx2_temp(c2, j) += dQ(j) * (dy * mesh.Ixy(c2));
                #pragma omp atomic
                Qy1_temp(c2, j) += dQ(j) * (dy * mesh.Ixx(c2));
                #pragma omp atomic
                Qy2_temp(c2, j) += dQ(j) * (dx * mesh.Ixy(c2));
            }
        }
        // store min/max for limiters
        if (recon.use_lim > 0) {
            #pragma omp critical
            {
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
                            Q_max(c1,j) = std::max(Qmax_temp, dQ(j)+Q1(j));
                            Q_min(c1,j) = std::min(Qmin_temp, dQ(j)+Q1(j));
                    }
                    else { // free-stream
                            Q_max(c1,j) = std::max(Qmax_temp, Q_in(j));
                            Q_min(c1,j) = std::min(Qmin_temp, Q_in(j));
                    }
                }
            }
        }
    }

    // final gradient
    dQx = Qx1_temp - Qx2_temp;
    dQy = Qy1_temp - Qy2_temp;
    
    #pragma omp parallel for schedule(dynamic)
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
        #pragma omp parallel for schedule(dynamic)
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
        #pragma omp parallel for schedule(dynamic)
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

#endif // RECONLS_H
