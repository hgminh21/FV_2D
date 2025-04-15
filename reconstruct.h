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
        // Temporary arrays for gradient estimation.
        MatrixXd Qx1_temp = MatrixXd::Zero(n_cells, 4);
        MatrixXd Qx2_temp = MatrixXd::Zero(n_cells, 4);
        MatrixXd Qy1_temp = MatrixXd::Zero(n_cells, 4);
        MatrixXd Qy2_temp = MatrixXd::Zero(n_cells, 4);

        for (int i = 0; i < n_faces; ++i) {
            int c1 = f2c(i, 0) - 1;
            int c2 = f2c(i, 1) - 1;

            // Interior face contributions (c2 >= 0)
            if (c2 >= 0) {
                RowVector4d dQ = Q.row(c2) - Q.row(c1);
                double dx = r_c(c2, 0) - r_c(c1, 0);
                double dy = r_c(c2, 1) - r_c(c1, 1);

                Qx1_temp.row(c1) += dQ * dx * Iyy(c1) / delta(c1);
                Qx2_temp.row(c1) += dQ * dy * Ixy(c1) / delta(c1);
                Qy1_temp.row(c1) += dQ * dy * Ixx(c1) / delta(c1);
                Qy2_temp.row(c1) += dQ * dx * Ixy(c1) / delta(c1);

                // For the other cell, subtract the contribution.
                Qx1_temp.row(c2) += dQ * dx * Iyy(c2) / delta(c2);
                Qx2_temp.row(c2) += dQ * dy * Ixy(c2) / delta(c2);
                Qy1_temp.row(c2) += dQ * dy * Ixx(c2) / delta(c2);
                Qy2_temp.row(c2) += dQ * dx * Ixy(c2) / delta(c2);
            }
            // Free stream condition for second order (c2 == -2)
            if (c2 == -2) {
                RowVector4d dQ = Q_in.transpose() - Q.row(c1);
                double dx = 2.0 * (r_f(i, 0) - r_c(c1, 0));
                double dy = 2.0 * (r_f(i, 1) - r_c(c1, 1));
                Qx1_temp.row(c1) += dQ * dx * Iyy(c1) / delta(c1);
                Qx2_temp.row(c1) += dQ * dy * Ixy(c1) / delta(c1);
                Qy1_temp.row(c1) += dQ * dy * Ixx(c1) / delta(c1);
                Qy2_temp.row(c1) += dQ * dx * Ixy(c1) / delta(c1);
            }
            // No-slip wall condition (c2 == -1)
            if (c2 == -1) {
                double rhoi = Q(c1, 0);
                double ui = Q(c1, 1) / rhoi;
                double vi = Q(c1, 2) / rhoi;
                // Use cell value Q(c1,3) for pressure
                double pi = (Q(c1, 3) - 0.5 * rhoi * (ui * ui + vi * vi)) * (gamma - 1.0);

                double nx = n_f(i, 0);
                double ny = n_f(i, 1);
                double vn = ui * nx + vi * ny;
                double ug = ui - 2.0 * vn * nx;
                double vg = vi - 2.0 * vn * ny;

                // Construct the reflected state Qg properly.
                RowVector4d Qg;
                Qg(0) = rhoi;
                Qg(1) = rhoi * ug;
                Qg(2) = rhoi * vg;
                Qg(3) = pi / (gamma - 1.0) + 0.5 * rhoi * (ug * ug + vg * vg);

                RowVector4d dQ = Qg - Q.row(c1);
                double dx = 2.0 * (r_f(i, 0) - r_c(c1, 0));
                double dy = 2.0 * (r_f(i, 1) - r_c(c1, 1));
                Qx1_temp.row(c1) += dQ * dx * Iyy(c1) / delta(c1);
                Qx2_temp.row(c1) += dQ * dy * Ixy(c1) / delta(c1);
                Qy1_temp.row(c1) += dQ * dy * Ixx(c1) / delta(c1);
                Qy2_temp.row(c1) += dQ * dx * Ixy(c1) / delta(c1);
            }
        }

        // Compute gradients from temporary arrays.
        MatrixXd Qx = Qx1_temp - Qx2_temp;
        MatrixXd Qy = Qy1_temp - Qy2_temp;

        // Use the computed gradients to reconstruct states and gradient corrections.
        for (int i = 0; i < n_faces; ++i) {
            int c1 = f2c(i, 0) - 1;
            int c2 = f2c(i, 1) - 1;

            double dfx = r_f(i, 0) - r_c(c1, 0);
            double dfy = r_f(i, 1) - r_c(c1, 1);
            dQ_L.row(i) = Qx.row(c1) * dfx + Qy.row(c1) * dfy;
            Q_L.row(i) = Q.row(c1) + dQ_L.row(i);

            if (c2 >= 0) {
                dfx = r_f(i, 0) - r_c(c2, 0);
                dfy = r_f(i, 1) - r_c(c2, 1);
                dQ_R.row(i) = Qx.row(c2) * dfx + Qy.row(c2) * dfy;
                Q_R.row(i) = Q.row(c2) + dQ_R.row(i);
            }

            // No-slip wall condition (c2 == -1) for the final state
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
            
            // Free stream condition (c2 == -2)
            if (c2 == -2) {
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
