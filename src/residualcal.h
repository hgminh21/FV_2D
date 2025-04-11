#ifndef RESIDUALCAL_H
#define RESIDUALCAL_H

#include <vector>
#include "fluxcomp.h"
#include "reconstruct.h"

using namespace std;

void ResidualCal(
    const vector<vector<double>>& Mesh,   // Mesh data
    const vector<vector<double>>& Q,      // Solution vector Q
    const vector<double>& Q_in,           // Initial condition vector Q_in
    vector<vector<double>>& Res           // Residual vector (output)
) {
    int n_nodes = Mesh[0][0];
    int n_faces = Mesh[0][1];
    int n_cells = Mesh[0][2];

    // Compute necessary intermediate data from Mesh
    vector<vector<double>> f2c(Mesh.begin() + 1 + 2 * n_faces, Mesh.begin() + 1 + 3 * n_faces);  // Face to cell mapping
    vector<vector<double>> A(Mesh.begin() + 1 + 4 * n_faces + n_cells + n_nodes, Mesh.begin() + 1 + 5 * n_faces + n_cells + n_nodes);  // Area
    vector<vector<double>> V(Mesh.begin() + 1 + 5 * n_faces + n_cells + n_nodes, Mesh.begin() + 1 + 5 * n_faces + 2 * n_cells + n_nodes);  // Volume

    // Solution Reconstruction
    vector<vector<double>> Q_L, Q_R;
    Reconstruct(Mesh, Q, Q_in, Q_L, Q_R);

    // Flux computation
    vector<vector<double>> F;
    FluxComp(Mesh, Q_L, Q_R, F);

    // Initialize the residual vector
    Res.resize(n_cells, vector<double>(4, 0.0));

    // Residual computation
    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c[i][0] - 1;
        int c2 = f2c[i][1] - 1;
        
        // Update the residual for c1
        Res[c1][0] = Res[c1][0] - (1.0 / V[c1][0]) * F[i][0] * A[i][0];
        Res[c1][1] = Res[c1][1] - (1.0 / V[c1][0]) * F[i][1] * A[i][0];
        Res[c1][2] = Res[c1][2] - (1.0 / V[c1][0]) * F[i][2] * A[i][0];
        Res[c1][3] = Res[c1][3] - (1.0 / V[c1][0]) * F[i][3] * A[i][0];

        // Update the residual for c2 if it's valid
        if (c2 >= 0) {
            Res[c2][0] = Res[c2][0] + (1.0 / V[c2][0]) * F[i][0] * A[i][0];
            Res[c2][1] = Res[c2][1] + (1.0 / V[c2][0]) * F[i][1] * A[i][0];
            Res[c2][2] = Res[c2][2] + (1.0 / V[c2][0]) * F[i][2] * A[i][0];
            Res[c2][3] = Res[c2][3] + (1.0 / V[c2][0]) * F[i][3] * A[i][0];
        }
    }
}

#endif
