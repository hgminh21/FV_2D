#include <iostream>
#include <vector>

#include "mesh_read.h"
#include "ssprk2.h"
#include "step_out.h"

using namespace std;

int main() {
    vector<vector<double>> Mesh;

    // Read the mesh file
    Mesh_read("grid_file.in", Mesh);
    cout << "Finished reading mesh file." << endl;

    // // Open file in write mode
    // std::ofstream outFile("matrix.txt");
    // // Check if file opened successfully
    // if (!outFile) {
    //     std::cerr << "Failed to open file.\n";
    //     return 1;
    // }
    // // Write matrix to file
    // for (const auto& row : Mesh) {
    //     for (const auto& val : row) {
    //         outFile << val << " ";
    //     }
    //     outFile << "\n"; // new line after each row
    // }
    // outFile.close();
    // std::cout << "Matrix written to matrix.txt successfully.\n";

    int n_nodes = Mesh[0][0];
    int n_faces = Mesh[0][1];
    int n_cells = Mesh[0][2];

    // Input
    double rho = 1.4;
    double p = 1;
    double u = 0.85;
    double v = 0;
    double gamma = 1.4;
    int max_step = 1;
    
    // Initialize
    double E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    vector<double> Q_in={rho, rho*u, rho*v, E};
    vector<vector<double>> Q_out(n_cells, Q_in);

    // SSP-RK2
    int step = 1;
    while (step <= max_step) {
        cout << "Current step: " << step << endl;
        vector<vector<double>> Q = Q_out;
        SSPRK2(Mesh, Q, Q_in, Q_out);
        step = step + 1;
    }

    cout << "Exited loop. Final step: " << step - 1 << endl;
    vector<vector<double>> Q_step;
    vector<double> rho_o, u_o, v_o, p_o;
    StepOutput(Mesh, Q_out, Q_step, rho_o, u_o, v_o, p_o);
    
    return 0;
}
