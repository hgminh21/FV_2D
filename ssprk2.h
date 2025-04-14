#ifndef SSPRK2_H
#define SSPRK2_H

#include <vector>
#include "mesh_read.h"
#include "residualcal.h"

using namespace std;

void SSPRK2(
    // const vector<vector<double>>& Mesh,   // Mesh data
    const int& n_faces,
    const int& n_cells,
    const vector<vector<double>>& f2c,
    const vector<vector<double>>& n_f,
    const vector<vector<double>>& A,
    const vector<vector<double>>& V,
    const vector<vector<double>>& Q,      // Solution vector Q
    const vector<double>& Q_in,           // Initial condition vector Q_in
    vector<vector<double>>& Q_out
) {
        double dt = 0.0005;
        double nc = n_faces;

        vector<vector<double>> Q_0 = Q;
        
        vector<vector<double>> Res;
        ResidualCal(n_faces, n_cells, f2c, n_f, A, V, Q_0, Q_in, Res);
        double Res1, Res2;

        vector<vector<double>> Q_1(n_cells, vector<double>(4, 0.0));
        for (int i = 0; i < n_cells; ++i) {
            Q_1[i][0] =  Q_0[i][0] + dt * Res[i][0];
            Q_1[i][1] =  Q_0[i][1] + dt * Res[i][1];
            Q_1[i][2] =  Q_0[i][2] + dt * Res[i][2];
            Q_1[i][3] =  Q_0[i][3] + dt * Res[i][3];
            Res1 =  Res1 + Res[i][0] * Res[i][0] / nc;
        }
    
        ResidualCal(n_faces, n_cells, f2c, n_f, A, V, Q_1, Q_in, Res);

        for (int i = 0; i < n_cells; ++i) {
            Q_out[i][0] =  0.5 * (Q_0[i][0] + Q_1[i][0] + dt * Res[i][0]);
            Q_out[i][1] =  0.5 * (Q_0[i][1] + Q_1[i][1] + dt * Res[i][1]);
            Q_out[i][2] =  0.5 * (Q_0[i][2] + Q_1[i][2] + dt * Res[i][2]);
            Q_out[i][3] =  0.5 * (Q_0[i][3] + Q_1[i][3] + dt * Res[i][3]);
            Res2 =  Res2 + Res[i][0] * Res[i][0] / nc;
        }

        Res1 = sqrt(Res1);
        Res2 = sqrt(Res2);
        cout << "Res1 = " << Res1 << endl;
        cout << "Res2 = " << Res2 << endl;
    }
    
#endif