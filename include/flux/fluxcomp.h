#ifndef FLUXCOMP_H
#define FLUXCOMP_H

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

#include "io/meshread.h"
#include "io/initialize.h"

using namespace Eigen;
using std::max;
using std::abs;

void compute_fluxes(const MeshData &mesh,
                    const MatrixXd &Q_L,
                    const MatrixXd &Q_R,
                    const Flow &flow,
                    const Flux &flux,
                    MatrixXd &F,
                    VectorXd &s_max_all) // NEW OUTPUT

{
    F.setZero();
    
    for (int i = 0; i < mesh.n_faces; ++i) {
        double nx = mesh.n_f(i, 0);
        double ny = mesh.n_f(i, 1);

        // Left state quantities
        double rhoL = Q_L(i, 0);
        double uL = Q_L(i, 1) / rhoL;
        double vL = Q_L(i, 2) / rhoL;
        double EL = Q_L(i, 3);
        double pL = (EL - 0.5 * rhoL * (uL * uL + vL * vL)) * (flow.gamma - 1);

        // Right state quantities
        double rhoR = Q_R(i, 0);
        double uR = Q_R(i, 1) / rhoR;
        double vR = Q_R(i, 2) / rhoR;
        double ER = Q_R(i, 3);
        double pR = (ER - 0.5 * rhoR * (uR * uR + vR * vR)) * (flow.gamma - 1);

        double vnL = uL * nx + vL * ny;
        double vnR = uR * nx + vR * ny;
        double cL = sqrt(flow.gamma * pL / rhoL);
        double cR = sqrt(flow.gamma * pR / rhoR);

        if (rhoL <= 0 || rhoR <= 0) {
            cerr << "Warning: Non-positive density at face " << i << endl;
            continue;
        }

        if (pL <= 0 || pR <= 0) {
            cerr << "Warning: Non-positive pressure at face " << i << endl;
            continue;
        }

        // Compute fluxes for left state
        double f1L = rhoL * vnL;
        double f2L = rhoL * uL * vnL + pL * nx;
        double f3L = rhoL * vL * vnL + pL * ny;
        double f4L = vnL * (EL + pL);

        // Compute fluxes for right state
        double f1R = rhoR * vnR;
        double f2R = rhoR * uR * vnR + pR * nx;
        double f3R = rhoR * vR * vnR + pR * ny;
        double f4R = vnR * (ER + pR);

        if (flux.method == "lax-friedrichs") {
            // Lax-Friedrichs numerical flux
            double s_max = max(abs(vnL) + cL, abs(vnR) + cR);
            s_max_all(i) = s_max;
            F(i, 0) = 0.5 * (f1L + f1R) - 0.5 * s_max * (Q_R(i, 0) - Q_L(i, 0));
            F(i, 1) = 0.5 * (f2L + f2R) - 0.5 * s_max * (Q_R(i, 1) - Q_L(i, 1));
            F(i, 2) = 0.5 * (f3L + f3R) - 0.5 * s_max * (Q_R(i, 2) - Q_L(i, 2));
            F(i, 3) = 0.5 * (f4L + f4R) - 0.5 * s_max * (Q_R(i, 3) - Q_L(i, 3));
        }
        else if (flux.method == "rusanov") {
            // Lax-Friedrichs numerical flux
            double vn_avg = (abs(vnL) + abs(vnR)) / 2.0;
            double c_avg = (cL + cR) / 2.0;
            double s_max = (vn_avg + c_avg);
            s_max_all(i) = s_max;
            F(i, 0) = 0.5 * (f1L + f1R) - 0.5 * s_max * (Q_R(i, 0) - Q_L(i, 0));
            F(i, 1) = 0.5 * (f2L + f2R) - 0.5 * s_max * (Q_R(i, 1) - Q_L(i, 1));
            F(i, 2) = 0.5 * (f3L + f3R) - 0.5 * s_max * (Q_R(i, 2) - Q_L(i, 2));
            F(i, 3) = 0.5 * (f4L + f4R) - 0.5 * s_max * (Q_R(i, 3) - Q_L(i, 3));
        }
        else if (flux.method == "roe") {
            // Roe numerical flux
            double rho_avg = sqrt(rhoL * rhoR);
            double u_avg = (sqrt(rhoL) * uL + sqrt(rhoR) * uR) / (sqrt(rhoL) + sqrt(rhoR));
            double v_avg = (sqrt(rhoL) * vL + sqrt(rhoR) * vR) / (sqrt(rhoL) + sqrt(rhoR));
            double vn_avg = (sqrt(rhoL) * vnL + sqrt(rhoR) * vnR) / (sqrt(rhoL) + sqrt(rhoR));
            double c_avg = (sqrt(rhoL) * cL + sqrt(rhoR) * cR) / (sqrt(rhoL) + sqrt(rhoR));
            double H_avg = (sqrt(rhoL) * (EL + pL) + sqrt(rhoR) * (ER + pR)) / (sqrt(rhoL) + sqrt(rhoR)) / rho_avg;
            // double c_avg = sqrt((flow.gamma - 1.0) * (H_avg - 0.5 * vn_avg * vn_avg));
            double vn_l = -u_avg * ny + v_avg * nx;
            double vn_s = u_avg * u_avg + v_avg * v_avg;
            double sigma = (flow.gamma - 1.0) / (c_avg * c_avg); 

            MatrixXd R(4 , 4);
            R << 1.0, 1.0, 1.0, 0.0,
                u_avg - c_avg * nx, u_avg, u_avg + c_avg * nx, -ny,
                v_avg - c_avg * ny, v_avg, v_avg + c_avg * ny, nx,
                H_avg - c_avg * vn_avg, vn_s * 0.5, H_avg + c_avg * vn_avg, vn_l;

            MatrixXd A(4 ,4);
            A << abs(vn_avg - c_avg), 0.0, 0.0, 0.0,
                0.0, abs(vn_avg), 0.0, 0.0,
                0.0, 0.0, abs(vn_avg + c_avg), 0.0,
                0.0, 0.0, 0.0, abs(vn_avg);

            MatrixXd L(4, 4);
            L << 0.5 * (0.5 * sigma * vn_s + vn_avg / c_avg), -0.5 * (sigma * u_avg + nx / c_avg), -0.5 * (sigma * v_avg + ny / c_avg), 0.5 * sigma,
                1 - 0.5 * sigma * vn_s, sigma * u_avg, sigma * v_avg, -sigma,
                0.5 * (0.5 * sigma * vn_s - vn_avg / c_avg), -0.5 * (sigma * u_avg - nx / c_avg), -0.5 * (sigma * v_avg - ny / c_avg), 0.5 * sigma,
                -vn_l, -ny, nx, 0.0;

            // MatrixXd A_abs = R * A * R.inverse();
            MatrixXd A_abs = R * A * L;
            s_max_all(i) = A_abs.cwiseAbs().maxCoeff();

            // Convert row vectors to column vectors
            VectorXd Q_R_col = Q_R.row(i).transpose();
            VectorXd Q_L_col = Q_L.row(i).transpose();

            RowVector4d f_D;
            f_D << (f1L + f1R), (f2L + f2R), (f3L + f3R), (f4L + f4R);
            VectorXd F_temp = 0.5 * f_D.transpose() - 0.5 * A_abs * (Q_R_col - Q_L_col);
            F.row(i) = F_temp.transpose();
        }

    }
}

#endif  // FLUXCOMP_H
