#ifndef FLUXCOMP_H
#define FLUXCOMP_H

#include <Eigen>
#include <cmath>
#include <algorithm>

using namespace Eigen;
using std::max;
using std::abs;

// Function to compute the numerical fluxes at faces using the Lax-Friedrichs scheme.
// Q_L: Left state at faces (n_faces x 4)
// Q_R: Right state at faces (n_faces x 4)
// n_f: Face normal matrix (n_faces x 2)
// gamma: Specific heat ratio
// Returns a matrix F (n_faces x 4) containing the computed fluxes.
void compute_fluxes(const MatrixXd &Q_L,
                    const MatrixXd &Q_R,
                    const MatrixXd &n_f,
                    double gamma,
                    MatrixXd &F,
                    VectorXd &s_max_all) // NEW OUTPUT

{
    int n_faces = Q_L.rows();
    MatrixXd F_local = MatrixXd::Zero(n_faces, 4);
    
    for (int i = 0; i < n_faces; ++i) {
        double nx = n_f(i, 0);
        double ny = n_f(i, 1);

        // Left state quantities
        double rhoL = Q_L(i, 0);
        double uL = Q_L(i, 1) / rhoL;
        double vL = Q_L(i, 2) / rhoL;
        double EL = Q_L(i, 3);
        double pL = (EL - 0.5 * rhoL * (uL * uL + vL * vL)) * (gamma - 1);

        // Right state quantities
        double rhoR = Q_R(i, 0);
        double uR = Q_R(i, 1) / rhoR;
        double vR = Q_R(i, 2) / rhoR;
        double ER = Q_R(i, 3);
        double pR = (ER - 0.5 * rhoR * (uR * uR + vR * vR)) * (gamma - 1);

        double vnL = uL * nx + vL * ny;
        double vnR = uR * nx + vR * ny;
        double cL = sqrt(gamma * pL / rhoL);
        double cR = sqrt(gamma * pR / rhoR);

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

        // Lax-Friedrichs numerical flux
        double s_max = max(abs(vnL) + cL, abs(vnR) + cR);
        s_max_all(i) = s_max;

        F_local(i, 0) = 0.5 * (f1L + f1R) - 0.5 * s_max * (Q_R(i, 0) - Q_L(i, 0));
        F_local(i, 1) = 0.5 * (f2L + f2R) - 0.5 * s_max * (Q_R(i, 1) - Q_L(i, 1));
        F_local(i, 2) = 0.5 * (f3L + f3R) - 0.5 * s_max * (Q_R(i, 2) - Q_L(i, 2));
        F_local(i, 3) = 0.5 * (f4L + f4R) - 0.5 * s_max * (Q_R(i, 3) - Q_L(i, 3));
    }
    F = F_local;
}

#endif  // FLUXCOMP_H
