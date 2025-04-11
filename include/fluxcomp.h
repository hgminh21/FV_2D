#ifndef FLUXCOMP_H
#define FLUXCOMP_H

#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

void FluxComp(const vector<vector<double>>& Mesh,
              const vector<vector<double>>& Q_L,
              const vector<vector<double>>& Q_R,
              vector<vector<double>>& F) {

    int n_faces = Mesh[0][1];
    double gamma = 1.4;

    vector<vector<double>> n_f(Mesh.begin() + 1 + 3 * n_faces, Mesh.begin() + 1 + 4 * n_faces);

    F.resize(n_faces, vector<double>(4, 0.0));

    for (int i = 0; i < n_faces; ++i) {
        double nx = n_f[i][0];
        double ny = n_f[i][1];

        double rhoL = Q_L[i][0];
        double uL = Q_L[i][1] / rhoL;
        double vL = Q_L[i][2] / rhoL;
        double EL = Q_L[i][3];
        double pL = (EL - 0.5 * rhoL * (uL * uL + vL * vL)) * (gamma - 1);

        double rhoR = Q_R[i][0];
        double uR = Q_R[i][1] / rhoR;
        double vR = Q_R[i][2] / rhoR;
        double ER = Q_R[i][3];
        double pR = (ER - 0.5 * rhoR * (uR * uR + vR * vR)) * (gamma - 1);
        
        if (rhoL <= 0 || rhoR <= 0) {
            cerr << "Warning: Non-positive density at face " << i << endl;
            continue;
        }

        if (pL <= 0 || pR <= 0) {
            cerr << "Warning: Non-positive pressure at face " << i << endl;
            continue;
        }

        double vnL = uL * nx + vL * ny;
        double vnR = uR * nx + vR * ny;
        double cL = sqrt(gamma * pL / rhoL);
        double cR = sqrt(gamma * pR / rhoR);

        double f1L = rhoL * vnL;
        double f2L = rhoL * uL * vnL + pL * nx;
        double f3L = rhoL * vL * vnL + pL * ny;
        double f4L = vnL * (EL + pL);

        double f1R = rhoR * vnR;
        double f2R = rhoR * uR * vnR + pR * nx;
        double f3R = rhoR * vR * vnR + pR * ny;
        double f4R = vnR * (ER + pR);

        // double vn = 0.5 * (abs(vnL) + abs(vnR));
        // double c = 0.5 * (cL + cR);
        
        // F[i][0] = 0.5 * (f1L + f1R) - 0.5 * (vn + c) * (Q_R[i][0] - Q_L[i][0]);
        // F[i][1] = 0.5 * (f2L + f2R) - 0.5 * (vn + c) * (Q_R[i][1] - Q_L[i][1]);
        // F[i][2] = 0.5 * (f3L + f3R) - 0.5 * (vn + c) * (Q_R[i][2] - Q_L[i][2]);
        // F[i][3] = 0.5 * (f4L + f4R) - 0.5 * (vn + c) * (Q_R[i][3] - Q_L[i][3]);

        // Lax-Friedrichs
        double s_max = max(abs(vnL) + cL, abs(vnR) + cR);
        
        F[i][0] = 0.5 * (f1L + f1R) - 0.5 * s_max * (Q_R[i][0] - Q_L[i][0]);
        F[i][1] = 0.5 * (f2L + f2R) - 0.5 * s_max * (Q_R[i][1] - Q_L[i][1]);
        F[i][2] = 0.5 * (f3L + f3R) - 0.5 * s_max * (Q_R[i][2] - Q_L[i][2]);
        F[i][3] = 0.5 * (f4L + f4R) - 0.5 * s_max * (Q_R[i][3] - Q_L[i][3]);
    }
}

#endif // FLUXCOMP_H
