#include <iostream>
#include <Eigen>

#include "initialize.h"
#include "meshread.h"
#include "ssprk2.h"

using namespace std;
using namespace Eigen;

int main() {
    Vector4d Q_init;
    MatrixXd Q;
    MeshData mesh;
    Flow flow;
    Solver solver;

    // Read mesh from file
    initialize("flow.in", mesh, flow, solver, Q_init, Q);

    // Run time integration
    ssprk2(mesh, solver, flow, Q, Q_init);

    return 0;
}
