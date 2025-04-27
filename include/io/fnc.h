#ifndef FNC_H
#define FNC_H

#include <Eigen/Dense>
#include "io/meshread.h"
#include "io/initialize.h"

using namespace Eigen;

void forceNcoef_cal(const MeshData &mesh,
                    const Flow &flow,
                    const MatrixXd Q,
                    const Vector4d Q_in,
                    const MatrixXd dVdn,
                    double &CL,
                    double &CD,
                    VectorXd &CP,
                    VectorXd &TauX,
                    VectorXd &TauY)
{
    // Free-stream dynamic pressure
    double rhoi = Q_in(0);
    double ui = Q_in(1) / rhoi;
    double vi = Q_in(2) / rhoi;
    double pi = (Q_in(3) - 0.5 * rhoi * (ui * ui + vi * vi)) * (flow.gamma - 1.0);
    double qbar = 0.5 * rhoi * (ui * ui + vi * vi);

    // Initialize parameters
    int j = 0;
    CL = 0.0; CD = 0.0;

    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;
        int c2 = mesh.f2c(i, 1) - 1;
        if (c2 >= 0) continue;
        if (c2 == -2) continue;

        const Vector2d nf  = mesh.n_f.row(i);

        double rho = Q(c1, 0);
        double u = Q(c1, 1) / rho;
        double v = Q(c1, 2) / rho;
        double p = (Q(c1, 3) - 0.5 * rho * (u * u + v * v)) * (flow.gamma - 1.0);

        CL += p * nf(1) * mesh.A(i) / qbar;
        CD += p * nf(0) * mesh.A(i) / qbar;

        CP(j) = (p - pi) / qbar;

        if (flow.type == 2) {
            TauX(j) = flow.mu * dVdn(j, 0);
            TauY(j) = flow.mu * dVdn(j, 1);
        }
        j += 1;
    }
}

#endif  // FNC_H 