#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include <Eigen>

using namespace Eigen;

// Function to reconstruct left and right states at each face
// f2c: Face-to-cell connectivity (n_faces x 2, one-indexed)
// Q: Cell state matrix (n_cells x 4)
// Q_L: Output left state at faces (n_faces x 4)
// Q_R: Output right state at faces (n_faces x 4)
// n_f: Face normal matrix (n_faces x 2)
// Q_in: Prescribed state for boundary (Vector4d)
// gamma: Specific heat ratio
void reconstruct(const MatrixXi &f2c,
                 const MatrixXd &Q,
                 MatrixXd &Q_L,
                 MatrixXd &Q_R,
                 const MatrixXd &n_f,
                 const Vector4d &Q_in,
                 double gamma)
{
    int n_faces = f2c.rows();
    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c(i, 0) - 1;  // convert to zero-index
        int c2 = f2c(i, 1) - 1;  // convert to zero-index

        // Set left state from cell c1
        Q_L.row(i) = Q.row(c1);

        // For interior faces, set right state directly
        if (c2 >= 0) {
            Q_R.row(i) = Q.row(c2);
        }

        // Reflective boundary condition (c2 == -1)
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

            Q_R(i, 0) = rhoL;                           // density remains the same
            Q_R(i, 1) = rhoL * uR;
            Q_R(i, 2) = rhoL * vR;
            Q_R(i, 3) = pL / (gamma - 1.0) + 0.5 * rhoL * (uR * uR + vR * vR);
        }

        // Prescribed boundary condition (c2 == -2)
        if (c2 == -2) {
            Q_R.row(i) = Q_in.transpose();
        }
    }
}

#endif  // RECONSTRUCT_H
