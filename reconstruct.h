#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include <Eigen>
#include <fstream>
#include <iostream>

using namespace Eigen;

// Function to reconstruct left and right states at each face
// f2c: Face-to-cell connectivity (n_faces x 2, one-indexed)
// Q: Cell state matrix (n_cells x 4)
// Q_L: Output left state at faces (n_faces x 4)
// Q_R: Output right state at faces (n_faces x 4)
// dQ_L, dQ_R: Output gradients (face corrections)
// n_f: Face normal matrix (n_faces x 2)
// Q_in: Prescribed state for boundary (Vector4d)
// gamma: Specific heat ratio
void reconstruct(const MatrixXi &f2c,
                 const MatrixXd &Q,
                 const MatrixXd &r_f,
                 const MatrixXd &r_c,
                 const VectorXd &Ixx,
                 const VectorXd &Iyy,
                 const VectorXd &Ixy,
                 const VectorXd &delta,
                 MatrixXd &Q_L,
                 MatrixXd &Q_R,
                 MatrixXd &dQ_L,
                 MatrixXd &dQ_R,
                 const MatrixXd &n_f,
                 const Vector4d &Q_in,
                 double gamma,
                 double order)
{
    int n_faces = f2c.rows();
    int n_cells = r_c.rows();

    if (order == 1.0) { // first order reconstruction
        for (int i = 0; i < n_faces; ++i) {
            int c1 = f2c(i, 0) - 1;
            int c2 = f2c(i, 1) - 1;

            Q_L.row(i) = Q.row(c1);

            if (c2 >= 0) {
                Q_R.row(i) = Q.row(c2);
            }

            // No-slip wall boundary condition (c2 == -1)
            if (c2 == -1) {
                double rhoL = Q_L(i, 0);
                double uL = Q_L(i, 1) / rhoL;
                double vL = Q_L(i, 2) / rhoL;
                double pL = (Q_L(i, 3) - 0.5 * rhoL * (uL * uL + vL * vL)) * (gamma - 1.0);

                double nx = n_f(i, 0);
                double ny = n_f(i, 1);
                double vn = uL * nx + vL * ny;
                double uR = uL - 2.0 * vn * nx;
                double vR = vL - 2.0 * vn * ny;

                Q_R(i, 0) = rhoL;
                Q_R(i, 1) = rhoL * uR;
                Q_R(i, 2) = rhoL * vR;
                Q_R(i, 3) = pL / (gamma - 1.0) + 0.5 * rhoL * (uR * uR + vR * vR);
            }
            
            // Free stream boundary condition (c2 == -2)
            if (c2 == -2) {
                Q_R.row(i) = Q_in.transpose();
            }
        }
    } else { // second order reconstruction
        // Temporary arrays for gradient estimation
        MatrixXd Qx1_temp = MatrixXd::Zero(n_cells, 4);
        MatrixXd Qx2_temp = MatrixXd::Zero(n_cells, 4);
        MatrixXd Qy1_temp = MatrixXd::Zero(n_cells, 4);
        MatrixXd Qy2_temp = MatrixXd::Zero(n_cells, 4);
    
        for (int i = 0; i < n_faces; ++i) {
            int c1 = f2c(i, 0) - 1;
            int c2 = f2c(i, 1) - 1;
            const Vector2d rc1 = r_c.row(c1);
            const Vector2d rf = r_f.row(i);
            const Vector2d nf = n_f.row(i);
    
            RowVector4d dQ;
            double dx, dy;
    
            if (c2 >= 0) {
                const Vector2d rc2 = r_c.row(c2);
                dQ = Q.row(c2) - Q.row(c1);
                dx = rc2(0) - rc1(0);
                dy = rc2(1) - rc1(1);
    
                // Contribution to both cells
                auto update = [&](int c, double dx, double dy) {
                    Qx1_temp.row(c) += dQ * dx * Iyy(c);
                    Qx2_temp.row(c) += dQ * dy * Ixy(c);
                    Qy1_temp.row(c) += dQ * dy * Ixx(c);
                    Qy2_temp.row(c) += dQ * dx * Ixy(c);
                };
                update(c1, dx, dy);
                update(c2, dx, dy);
            } 
            else if (c2 == -2) { // Free stream condition
                dQ = Q_in.transpose() - Q.row(c1);
                dx = -2.0 * (rc1(0) - rf(0)) * nf(0);
                dy = -2.0 * (rc1(1) - rf(1)) * nf(1);
                Qx1_temp.row(c1) += dQ * dx * Iyy(c1);
                Qx2_temp.row(c1) += dQ * dy * Ixy(c1);
                Qy1_temp.row(c1) += dQ * dy * Ixx(c1);
                Qy2_temp.row(c1) += dQ * dx * Ixy(c1);
            } 
            else if (c2 == -1) { // Wall boundary
                double rho = Q(c1, 0);
                double u = Q(c1, 1) / rho;
                double v = Q(c1, 2) / rho;
                double p = (Q(c1, 3) - 0.5 * rho * (u * u + v * v)) * (gamma - 1.0);
    
                double vn = u * nf(0) + v * nf(1);
                double ug = u - 2.0 * vn * nf(0);
                double vg = v - 2.0 * vn * nf(1);
    
                RowVector4d Qg;
                Qg << rho, rho * ug, rho * vg, p / (gamma - 1.0) + 0.5 * rho * (ug * ug + vg * vg);
                dQ = Qg - Q.row(c1);
                dx = -2.0 * (rc1(0) - rf(0)) * nf(0);
                dy = -2.0 * (rc1(1) - rf(1)) * nf(1);
    
                Qx1_temp.row(c1) += dQ * dx * Iyy(c1);
                Qx2_temp.row(c1) += dQ * dy * Ixy(c1);
                Qy1_temp.row(c1) += dQ * dy * Ixx(c1);
                Qy2_temp.row(c1) += dQ * dx * Ixy(c1);
            }
        }
    
        // Final gradient computation
        MatrixXd Qx = Qx1_temp - Qx2_temp;
        MatrixXd Qy = Qy1_temp - Qy2_temp;
    
        for (int i = 0; i < n_faces; ++i) {
            int c1 = f2c(i, 0) - 1;
            int c2 = f2c(i, 1) - 1;
            const Vector2d rc1 = r_c.row(c1);
            const Vector2d rf = r_f.row(i);
    
            double dfx = rf(0) - rc1(0);
            double dfy = rf(1) - rc1(1);
            dQ_L.row(i) = Qx.row(c1) * dfx + Qy.row(c1) * dfy;
            Q_L.row(i) = Q.row(c1) + dQ_L.row(i);
    
            if (c2 >= 0) {
                const Vector2d rc2 = r_c.row(c2);
                dfx = rf(0) - rc2(0);
                dfy = rf(1) - rc2(1);
                dQ_R.row(i) = Qx.row(c2) * dfx + Qy.row(c2) * dfy;
                Q_R.row(i) = Q.row(c2) + dQ_R.row(i);
            } 
            else if (c2 == -1) { // Wall BC final state
                double rhoL = Q_L(i, 0);
                double uL = Q_L(i, 1) / rhoL;
                double vL = Q_L(i, 2) / rhoL;
                double pL = (Q_L(i, 3) - 0.5 * rhoL * (uL * uL + vL * vL)) * (gamma - 1.0);
    
                const Vector2d nf = n_f.row(i);
                double vn = uL * nf(0) + vL * nf(1);
                double uR = uL - 2.0 * vn * nf(0);
                double vR = vL - 2.0 * vn * nf(1);
    
                Q_R(i, 0) = rhoL;
                Q_R(i, 1) = rhoL * uR;
                Q_R(i, 2) = rhoL * vR;
                Q_R(i, 3) = pL / (gamma - 1.0) + 0.5 * rhoL * (uR * uR + vR * vR);
            } 
            else if (c2 == -2) {
                Q_R.row(i) = Q_in.transpose();
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
