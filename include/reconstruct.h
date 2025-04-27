#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "io/meshread.h"
#include "io/initialize.h"

using namespace Eigen;

void reconstruct(const MeshData &mesh, 
                 const MatrixXd &Q,
                 MatrixXd &Q_L,
                 MatrixXd &Q_R,
                 MatrixXd &dQx,
                 MatrixXd &dQy,
                 const Vector4d &Q_in,
                 const Flow &flow,
                 const Solver &solver,
                 MatrixXd &Qx1_temp,
                 MatrixXd &Qx2_temp,
                 MatrixXd &Qy1_temp,
                 MatrixXd &Qy2_temp,
                 MatrixXd &Q_max,
                 MatrixXd &Q_min,
                 VectorXd &phi)
{
    Qx1_temp.setZero();
    Qx2_temp.setZero();
    Qy1_temp.setZero();
    Qy2_temp.setZero();
    phi.setOnes();

    if (solver.order == 1.0) { // first order reconstruction
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;

            Q_L.row(i) = Q.row(c1);

            if (c2 >= 0) {
                Q_R.row(i) = Q.row(c2);
            }

            // No-slip wall boundary condition (c2 == -1)
            if (c2 == -1) {
                double rhoL = Q_L(i, 0);
                double uL = Q_L(i, 1) / rhoL;
                double vL = Q_L(i, 2) / rhoL;
                double pL = (Q_L(i, 3) - 0.5 * rhoL * (uL * uL + vL * vL)) * (flow.gamma - 1.0);

                double nx = mesh.n_f(i, 0);
                double ny = mesh.n_f(i, 1);
                double vn = uL * nx + vL * ny;
                double uR = uL - 2.0 * vn * nx;
                double vR = vL - 2.0 * vn * ny;

                Q_R(i, 0) = rhoL;
                Q_R(i, 1) = rhoL * uR;
                Q_R(i, 2) = rhoL * vR;
                Q_R(i, 3) = pL / (flow.gamma - 1.0) + 0.5 * rhoL * (uR * uR + vR * vR);
            }
            
            // Free stream boundary condition (c2 == -2)
            if (c2 == -2) {
                Q_R.row(i) = Q_in.transpose();
            }
        }
    } else { // second order reconstruction
    
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;
            const Vector2d rc1 = mesh.r_c.row(c1);
            const Vector2d rf = mesh.r_f.row(i);
            const Vector2d nf = mesh.n_f.row(i);
    
            RowVector4d dQ;
            RowVector4d Qg;
            double dx, dy;
    
            if (c2 >= 0) {
                const Vector2d rc2 = mesh.r_c.row(c2);
                dQ = Q.row(c2) - Q.row(c1);
                dx = rc2(0) - rc1(0);
                dy = rc2(1) - rc1(1);
    
                // Contribution to both cells
                auto update = [&](int c, double dx, double dy) {
                    Qx1_temp.row(c) += dQ * dx * mesh.Iyy(c);
                    Qx2_temp.row(c) += dQ * dy * mesh.Ixy(c);
                    Qy1_temp.row(c) += dQ * dy * mesh.Ixx(c);
                    Qy2_temp.row(c) += dQ * dx * mesh.Ixy(c);
                };
                update(c1, dx, dy);
                update(c2, dx, dy);
            } 
            else if (c2 == -2) { // Free stream condition
                dQ = Q_in.transpose() - Q.row(c1);
                dx = -2.0 * (rc1(0) - rf(0)) * nf(0);
                dy = -2.0 * (rc1(1) - rf(1)) * nf(1);
                Qx1_temp.row(c1) += dQ * dx * mesh.Iyy(c1);
                Qx2_temp.row(c1) += dQ * dy * mesh.Ixy(c1);
                Qy1_temp.row(c1) += dQ * dy * mesh.Ixx(c1);
                Qy2_temp.row(c1) += dQ * dx * mesh.Ixy(c1);
            } 
            else if (c2 == -1) { // Wall boundary
                double rhoi = Q(c1, 0);
                double ui = Q(c1, 1) / rhoi;
                double vi = Q(c1, 2) / rhoi;
                double pi = (Q(c1, 3) - 0.5 * rhoi * (ui * ui + vi * vi)) * (flow.gamma - 1.0);
    
                double vn = ui * nf(0) + vi * nf(1);
                double ug = ui - 2.0 * vn * nf(0);
                double vg = vi - 2.0 * vn * nf(1);
    
                Qg << rhoi, rhoi * ug, rhoi * vg, pi / (flow.gamma - 1.0) + 0.5 * rhoi * (ug * ug + vg * vg);
                dQ = Qg - Q.row(c1);
                dx = -2.0 * (rc1(0) - rf(0)) * nf(0);
                dy = -2.0 * (rc1(1) - rf(1)) * nf(1);
    
                Qx1_temp.row(c1) += dQ * dx * mesh.Iyy(c1);
                Qx2_temp.row(c1) += dQ * dy * mesh.Ixy(c1);
                Qy1_temp.row(c1) += dQ * dy * mesh.Ixx(c1);
                Qy2_temp.row(c1) += dQ * dx * mesh.Ixy(c1);
            }
            // Storing Min Max Q values for limiters
            if (flow.use_lim == 0) continue;
            else {
                for (int j = 0; j < 4; ++j) {
                if (c2 >= 0) {
                        Q_max(c1, j) = std::max(Q(c1, j), Q(c2, j));
                        Q_min(c1, j) = std::min(Q(c1, j), Q(c2, j));
                        Q_max(c2, j) = Q_max(c1, j);
                        Q_min(c2, j) = Q_min(c1, j);
                    }
                    else if (c2 == -1) { // Wall BC
                        Q_max(c1, j) = std::max(Q(c1, j), Qg(j));
                        Q_min(c1, j) = std::min(Q(c1, j), Qg(j));
                    }
                    else if (c2 == -2) { // Free stream BC
                        Q_max(c1, j) = std::max(Q(c1, j), Q_in(j));
                        Q_min(c1, j) = std::min(Q(c1, j), Q_in(j));
                    }
                }
            }
        }
    
        // Final gradient computation
        dQx = Qx1_temp - Qx2_temp;
        dQy = Qy1_temp - Qy2_temp;
    
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;
            const Vector2d rc1 = mesh.r_c.row(c1);
            const Vector2d rf = mesh.r_f.row(i);
    
            double dfx = rf(0) - rc1(0);
            double dfy = rf(1) - rc1(1);
            Q_L.row(i) = Q.row(c1) + dQx.row(c1) * dfx + dQy.row(c1) * dfy;
    
            if (c2 >= 0) {
                const Vector2d rc2 = mesh.r_c.row(c2);
                dfx = rf(0) - rc2(0);
                dfy = rf(1) - rc2(1);
                Q_R.row(i) = Q.row(c2) + dQx.row(c2) * dfx + dQy.row(c2) * dfy;
            } 
            else if (c2 == -1) { // Wall BC final state
                double rhoL = Q_L(i, 0);
                double uL = Q_L(i, 1) / rhoL;
                double vL = Q_L(i, 2) / rhoL;
                double pL = (Q_L(i, 3) - 0.5 * rhoL * (uL * uL + vL * vL)) * (flow.gamma - 1.0);
    
                const Vector2d nf = mesh.n_f.row(i);
                double vn = uL * nf(0) + vL * nf(1);
                double uR = uL - 2.0 * vn * nf(0);
                double vR = vL - 2.0 * vn * nf(1);
    
                Q_R(i, 0) = rhoL;
                Q_R(i, 1) = rhoL * uR;
                Q_R(i, 2) = rhoL * vR;
                Q_R(i, 3) = pL / (flow.gamma - 1.0) + 0.5 * rhoL * (uR * uR + vR * vR);
            } 
            else if (c2 == -2) {
                Q_R.row(i) = Q_in.transpose();
            }
        }
        
        // Barth limmiter testing 
        if (flow.use_lim == 1) {
            for (int i = 0; i < mesh.n_faces; ++i) {
                int c1 = mesh.f2c(i, 0) - 1;
                int c2 = mesh.f2c(i, 1) - 1;
                double phi_f1;
                double phi_f2;
                if (c2 < 0) continue;
                for (int j = 0; j < 4; ++j) {
                    // cell 1
                    if (Q_L(i, j) > Q_max(c1,j)) {
                        phi_f1 = (Q_max(c1,j) - Q(c1, j)) / (Q_L(i,j) - Q(c1,j));
                    }
                    else if (Q_L(i, j) < Q_min(c1,j)) {
                        phi_f1 = (Q_min(c1,j) - Q(c1, j)) / (Q_L(i,j) - Q(c1,j));
                    }
                    else {
                        phi_f1 = 1.0;
                    }
                    // cell 2
                    if (Q_R(i, j) > Q_max(c2,j)) {
                        phi_f2 = (Q_max(c2,j) - Q(c2, j)) / (Q_R(i,j) - Q(c2,j));
                    }
                    else if (Q_R(i, j) < Q_min(c2,j)) {
                        phi_f2 = (Q_min(c2,j) - Q(c2, j)) / (Q_R(i,j) - Q(c2,j));
                    }
                    else {
                        phi_f2 = 1.0;
                    }
                    if (phi_f1 < phi(c1)) phi(c1) = phi_f1;
                    if (phi_f2 < phi(c2)) phi(c2) = phi_f2;
                }
            }

            // update Q_L and Q_R with the limiter (note: potentially could put this in the loop above)
            for (int i = 0; i < mesh.n_faces; ++i) {
                int c1 = mesh.f2c(i, 0) - 1;
                int c2 = mesh.f2c(i, 1) - 1;
                const Vector2d rf = mesh.r_f.row(i);

                // Left face
                const Vector2d rc1 = mesh.r_c.row(c1);
                double dfx = rf(0) - rc1(0);
                double dfy = rf(1) - rc1(1);
                Q_L.row(i) = Q.row(c1) + phi(c1) * (dQx.row(c1) * dfx + dQy.row(c1) * dfy);

                if (c2 < 0) continue;
                // Right face
                const Vector2d rc2 = mesh.r_c.row(c2);
                dfx = rf(0) - rc2(0);
                dfy = rf(1) - rc2(1);
                Q_R.row(i) = Q.row(c2) + phi(c2) * (dQx.row(c2) * dfx + dQy.row(c2) * dfy);
            }
        }
    }
}


// Utility function to write a matrix to a text file.
void writeMatrixToFile(const Eigen::MatrixXd& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for writing!" << std::endl;
        return;
    }

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j < mat.cols() - 1)
                file << " ";
        }
        file << "\n";
    }

    file.close();
}

#endif  // RECONSTRUCT_H
