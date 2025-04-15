#include <iostream>
#include <Eigen>
#include "meshread.h"
#include "ssprk2.h"

using namespace std;
using namespace Eigen;

int main() {
    // Read mesh from file
    MeshData mesh = readMesh("grid_file.in");
    cout << "Finished read mesh data." << endl;
    // outputMeshData(mesh, "meshdata.txt"); 

    int n_cells = mesh.V.rows();
    
    // Physical constants
    double rho = 1.4;
    double p = 1.0;
    double u = 0.85;
    double v = 0.0;
    double gamma = 1.4;

    // Initial conserved variables
    double E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    Vector4d Q_init;
    Q_init << rho, rho * u, rho * v, E;

    // Initialize Q for all cells
    MatrixXd Q = MatrixXd::Zero(n_cells, 4);
    for (int i = 0; i < n_cells; ++i) {
        Q.row(i) = Q_init.transpose();
    }

    // Time integration parameters
    double CFL = 1;
    int n_steps = 1;
    int order = 2;

    // Run time integration
    ssprk2(mesh, Q, Q_init, gamma, CFL, n_steps, order);

    return 0;
}
