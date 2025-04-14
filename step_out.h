#ifndef STEP_OUT_H
#define STEP_OUT_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

// #include <CGNS>

using namespace std;

void StepOutput(
    int& n_nodes,
    int& n_faces,
    int& n_cells,
    const vector<vector<double>>& f2n,
    const vector<vector<double>>& f2c,
    const vector<vector<double>>& n_f,
    const vector<vector<double>>& r_nodes,
    const vector<vector<double>>& Q_out,
    vector<vector<double>>& Q_step,
    vector<double>& rho,
    vector<double>& u,
    vector<double>& v,
    vector<double>& p
) {
    double gamma = 1.4;
    // Post-processing
    Q_step.resize(n_nodes, vector<double>(4, 0.0));
    vector<vector<double>> n_shared(n_nodes, vector<double>(1, 0.0));
    rho.resize(n_nodes), u.resize(n_nodes), v.resize(n_nodes), p.resize(n_nodes);
    
    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c[i][0] - 1;
        int c2 = f2c[i][1] - 1;
        int n1 = f2n[i][0] - 1;
        int n2 = f2n[i][1] - 1;
        if (c2 >= 0) {
            n_shared[n1][0] = n_shared[n1][0] + 1;
            n_shared[n2][0] = n_shared[n2][0] + 1;
            Q_step[n1][0] = Q_step[n1][0] + (Q_out[c1][0] + Q_out[c2][0]) / 2.0;
            Q_step[n2][0] = Q_step[n2][0] + (Q_out[c1][0] + Q_out[c2][0]) / 2.0;
            Q_step[n1][1] = Q_step[n1][1] + (Q_out[c1][1] + Q_out[c2][1]) / 2.0;
            Q_step[n2][1] = Q_step[n2][1] + (Q_out[c1][1] + Q_out[c2][1]) / 2.0;
            Q_step[n1][2] = Q_step[n1][2] + (Q_out[c1][2] + Q_out[c2][2]) / 2.0;
            Q_step[n2][2] = Q_step[n2][2] + (Q_out[c1][2] + Q_out[c2][2]) / 2.0;
            Q_step[n1][3] = Q_step[n1][3] + (Q_out[c1][3] + Q_out[c2][3]) / 2.0;
            Q_step[n2][3] = Q_step[n2][3] + (Q_out[c1][3] + Q_out[c2][3]) / 2.0;
        }
        else {
            n_shared[n1][0] = n_shared[n1][0] + 1;
            n_shared[n2][0] = n_shared[n2][0] + 1;
            Q_step[n1][0] = Q_step[n1][0] + Q_out[c1][0];
            Q_step[n2][0] = Q_step[n2][0] + Q_out[c1][0];
            Q_step[n1][1] = Q_step[n1][1] + Q_out[c1][1];
            Q_step[n2][1] = Q_step[n2][1] + Q_out[c1][1];
            Q_step[n1][2] = Q_step[n1][2] + Q_out[c1][2];
            Q_step[n2][2] = Q_step[n2][2] + Q_out[c1][2];
            Q_step[n1][3] = Q_step[n1][3] + Q_out[c1][3];
            Q_step[n2][3] = Q_step[n2][3] + Q_out[c1][3];
        }
    }

    for (int i = 0; i < n_nodes; ++i) {
        Q_step[i][0] = Q_step[i][0] / n_shared[i][0];
        Q_step[i][1] = Q_step[i][1] / n_shared[i][0];
        Q_step[i][2] = Q_step[i][2] / n_shared[i][0];
        Q_step[i][3] = Q_step[i][3] / n_shared[i][0];
        rho[i] = Q_step[i][0];
        u[i] = Q_step[i][1] / rho[i];
        v[i] = Q_step[i][2] / rho[i];
        double E = Q_step[i][3];
        p[i] = (E - 0.5 * rho[i] * (u[i] * u[i] + v[i] * v[i])) / (gamma - 1);
    }
        // Write solution to "sol.dat"
        ofstream fout("sol.dat");
        fout << fixed << setprecision(6);
        fout << "x y rho u v p\n"; // Optional header
    
        for (int i = 0; i < n_nodes; ++i) {
            fout << r_nodes[i][0] << " "
                 << r_nodes[i][1] << " "
                 << rho[i] << " "
                 << u[i] << " "
                 << v[i] << " "
                 << p[i] << "\n";
        }
    
        fout.close();
    
}

#endif