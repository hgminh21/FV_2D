#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>  // For std::exit
#include <petscsys.h>  // Include for PetscInitialize/Finalize
#include <chrono> // For timing 

#include "io/initialize.h"
#include "io/meshread.h"
#include "ssprk2.h"
#include "implicit/implicit.h"

using namespace std;
using namespace Eigen;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {

    cout << "============================= UNIVERSITY OF KANSAS =============================" << endl;
    cout << "=========================== FINITE VOLUME CFD SOLVER ===========================" << endl;
    cout << "=============================== By Hoang Minh To ===============================" << endl;

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
    Reconstruct recon;
    Time time;
    Flux flux;
    
    auto t0 = Clock::now();   // Start

    // Initialize the mesh and flow data
    cout << "Initializing..." << endl;
    initialize(input_file, mesh, flow, solver, recon, flux, time, Q_init, Q);
    // Check if PETSc has been initialized; if not, initialize it
    PetscBool isMPIInitialized;
    PetscInitialized(&isMPIInitialized);
    if (!isMPIInitialized) {
        PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, "Usage: ...");
        if (ierr) {
            std::cerr << "PETSc initialization failed!" << std::endl;
            return 1;  // Exit with error
        }
    }

    cout << "Finished initializing." << endl;
    auto t1 = Clock::now();   // After initialization
    std::cout << "Time elapsed for initialize = " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    cout << "================================================================================" << endl;
    cout << "Simulation started." << endl;

    // Run time integration
    if (time.method == "implicit") {
        // Perform tasks for the implicit method
        implicit_scheme(mesh, solver, flow, recon, flux, time, Q, Q_init);
    } else {
        // Handle other methods or default case
        ssprk2(mesh, solver, flow, recon, flux, time, Q, Q_init);
    }

    cout << "Simulation completed." << endl;
    auto t2 = Clock::now();   // After initialization
    std::cout << "Time elapsed simulation = " << std::chrono::duration<double>(t2 - t1).count() << " s\n";
    cout << "================================================================================" << endl;

    // Finalize PETSc
    PetscFinalize();

    return 0;
}
