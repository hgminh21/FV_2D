#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>  // For std::exit

#include "initialize.h"
#include "meshread.h"
#include "ssprk2.h"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    // Check if the user has provided an input file as an argument
    if (argc < 2) {
        cerr << "Error: Please specify the input file path as a command-line argument." << endl;
        return 1; // Exit with an error code
    }

    // Get the input file path from the command-line arguments
    const char* input_file = argv[1];

    Vector4d Q_init;
    MatrixXd Q;
    MeshData mesh;
    Flow flow;
    Solver solver;

    // Read mesh from the file specified in the terminal argument
    initialize(input_file, mesh, flow, solver, Q_init, Q);

    // Run time integration
    ssprk2(mesh, solver, flow, Q, Q_init);

    return 0;
}
