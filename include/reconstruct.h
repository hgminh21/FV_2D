#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H

#include <vector>
#include <cmath>

using namespace std;

void Reconstruct(
    const vector<vector<double>>& Mesh,
    const vector<vector<double>>& Q,
    const vector<double>& Q_in,
    vector<vector<double>>& Q_L,
    vector<vector<double>>& Q_R
) {
    int n_faces = Mesh[0][1];
    int n_cells = Mesh[0][2];

    vector<vector<double>> f2c(Mesh.begin() + 1 + 2 * n_faces, Mesh.begin() + 1 + 3 * n_faces);
    vector<vector<double>> n_f(Mesh.begin() + 1 + 3 * n_faces, Mesh.begin() + 1 + 4 * n_faces);

    double gamma = 1.4;

    Q_L.resize(n_faces, vector<double>(4, 0.0));
    Q_R.resize(n_faces, vector<double>(4, 0.0));

    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c[i][0] - 1;
        int c2 = f2c[i][1] - 1;

        Q_L[i] = Q[c1];

        if (c2 >= 0) {
            Q_R[i] = Q[c2];
        }

        if (c2 == -1) {
            double rhoL = Q_L[i][0];
            double uL = Q_L[i][1] / rhoL;
            double vL = Q_L[i][2] / rhoL;
            double pL = (Q_L[i][3] - 0.5 * rhoL * (uL * uL + vL * vL)) * (gamma - 1.0);

            double nx = n_f[i][0];
            double ny = n_f[i][1];

            double vn = uL * nx + vL * ny;
            double uR = uL - 2.0 * vn * nx;
            double vR = vL - 2.0 * vn * ny;

            Q_R[i][0] = rhoL; // left and right density to be equal
            Q_R[i][1] = rhoL * uR;
            Q_R[i][2] = rhoL * vR;
            Q_R[i][3] = pL / (gamma - 1.0) + 0.5 * rhoL * (uR * uR + vR * vR); // left and right pressure to be equal
        }

        if (c2 == -2) {
            Q_R[i] = Q_in;
        }
    }
}


#endif // RECONSTRUCT_H
