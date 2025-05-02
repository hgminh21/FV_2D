#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>  // For std::exit
#include <petscsys.h>  // Include for PetscInitialize/Finalize
#include <chrono> // For timing 

#include "io/initialize.h"
#include "io/meshread.h"
#include "explicit/ssprk3.h"
#include "explicit/ssprk2.h"
#include "implicit/implicit.h"
#include <omp.h> 

using namespace std;
using namespace Eigen;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {

    cout << "============================= UNIVERSITY OF KANSAS =============================" << endl;
    cout << "=========================== FINITE VOLUME CFD SOLVER ===========================" << endl;
    cout << "=============================== By Hoang Minh To ===============================" << endl;

    // Check if the user has provided an input file as an argument
    // 1) Default to all hardware threads
    int nt = omp_get_max_threads();
    std::string infile;
    
    // 2) Simple flag parser: -t N for threads, then the input file
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t" && i+1 < argc) {
            nt = std::atoi(argv[++i]);          // consume the thread count
        }
        else if (infile.empty()) {
            infile = arg;                       // first non-flag = infile
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    // 3) Must have an input file
    if (infile.empty()) {
        std::cerr << "Usage: " << argv[0] << " [-t N] <input>\n";
        return 1;
    }

    // 4) Tell OpenMP how many threads to use
    omp_set_num_threads(nt);
    std::cout << "Running with " << nt << " OpenMP threads\n";

    // REMOVE -t FROM argv SO THE SOLVER NEVER SEES IT
    int w = 1;  // write‐index: keep argv[0]
    for (int r = 1; r < argc; ++r) {
        std::string s = argv[r];
        if (s == "-t" && r+1 < argc) {
            // skip both "-t" and its numeric argument
            ++r;
        }
        else {
            argv[w++] = argv[r];
        }
    }
    argc = w;
    argv[w] = nullptr;  // just in case any parser walks argv[] to a nullptr

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
    initialize(infile, mesh, flow, solver, recon, flux, time, Q_init, Q);
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
        if (time.rk_steps == 2) {
            ssprk2(mesh, solver, flow, recon, flux, time, Q, Q_init);
        } else if (time.rk_steps == 3) {
            // Perform tasks for the explicit method
            ssprk3(mesh, solver, flow, recon, flux, time, Q, Q_init);
        }
    }

    cout << "Simulation completed." << endl;
    auto t2 = Clock::now();   // After initialization
    std::cout << "Time elapsed simulation = " << std::chrono::duration<double>(t2 - t1).count() << " s\n";
    cout << "================================================================================" << endl;

    // Finalize PETSc
    PetscFinalize();

    return 0;
}
