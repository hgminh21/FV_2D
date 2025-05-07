#ifndef FNC_H
#define FNC_H

#include <Eigen/Dense>
#include <cmath>
#include "io/meshread.h"
#include "io/initialize.h"

using namespace Eigen;
using namespace std;

void forceNcoef_cal(const MeshData &mesh,
                    const Flow &flow,
                    const MatrixXd Q,
                    const Vector4d Q_in,
                    const VectorXd dVdn,
                    double &CL,
                    double &CD,
                    double &Fx,
                    double &Fy,
                    VectorXd &CP,
                    VectorXd &TauW,
                    VectorXd &Cf)
{
    // Free-stream dynamic pressure
    double rhoi = Q_in(0);
    double ui = Q_in(1) / rhoi;
    double vi = Q_in(2) / rhoi;
    double Aoa = atan2(vi, ui);
    double pi = (Q_in(3) - 0.5 * rhoi * (ui * ui + vi * vi)) * (flow.gamma - 1.0);
    double qbar = 0.5 * rhoi * (ui * ui + vi * vi);

    // Initialize parameters
    int j = 0;
    Fx = 0.0; Fy = 0.0;

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

        Fx += p * nf(0) * mesh.A(i);
        Fy += p * nf(1) * mesh.A(i);

        CP(j) = (p - pi) / qbar;

        if (flow.type == 2) {
            TauW(j) = flow.mu * dVdn(j);
            Cf(j) = TauW(j) / qbar;
            Fx += TauW(j) * nf(0) * mesh.A(i);
            Fy += TauW(j) * nf(1) * mesh.A(i);
        }
        j += 1;
    }
    CL = (-Fx * sin(Aoa) + Fy * cos(Aoa)) / qbar;
    CD = (Fx * cos(Aoa) + Fy * sin(Aoa)) / qbar;
}

#endif  // FNC_H 