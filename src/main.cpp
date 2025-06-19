#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <cstdlib>  // For std::exit
#include <chrono> // For timing 

#include "io/initialize.h"
#include "io/meshread.h"
// #include "explicit/ssprk3.h"
#include "explicit/ssprk2.h"
// #include "implicit/implicit.h"
#include <omp.h> 

using namespace std;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {

    // Check if the user has provided an input file as an argument
    // 1) Default to all hardware threads
    // int nt = omp_get_max_threads();
    int nt  = 1;
    string infile;
    
    // 2) Simple flag parser: -t N for threads, then the input file
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-t" && i + 1 < argc) {
            int nt2 = atoi(argv[++i]);
            if (nt2 <= nt) {
                nt = nt2;
            } else {
                cerr << "Warning: -t N must be <= " << nt << "\n";
                return 1;
            }
        }
        else if (infile.empty()) {
            infile = arg;  // first non-flag = infile
        }
        else {
            cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    // 3) Must have an input file
    if (infile.empty()) {
        cerr << "Usage: " << argv[0] << " [-t N] <input>\n";
        return 1;
    }

    // // 4) Tell OpenMP how many threads to use
    // omp_set_num_threads(nt);

    // // REMOVE -t FROM argv SO THE SOLVER NEVER SEES IT
    // int w = 1;  // write‐index: keep argv[0]
    // for (int r = 1; r < argc; ++r) {
    //     string s = argv[r];
    //     if (s == "-t" && r+1 < argc) {
    //         // skip both "-t" and its numeric argument
    //         ++r;
    //     }
    //     else {
    //         argv[w++] = argv[r];
    //     }
    // }
    // argc = w;
    // argv[w] = nullptr;  // just in case any parser walks argv[] to a nullptr

    string build_date = __DATE__; 
    // Fixed box width (match your overall box width)
    const int box_width = 81;  // Width of the box (between the borders)
    string left_content = "║ Threads: " + to_string(nt) + "   Build Date: " + build_date;
    int left_content_length = left_content.length() - 1;
    int remaining_space = box_width - left_content_length;

    cout << "==============================The University of Kansas============================" << endl;
    std::cout << std::endl;
    cout << "╔════════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║ .----------------.  .----------------.  .----------------.  .----------------. ║" << endl;
    cout << "║| .--------------. || .--------------. || .--------------. || .--------------. |║" << endl;
    cout << "║| |  _________   | || | ____   ____  | || |    _____     | || |  ________    | |║" << endl;
    cout << "║| | |_   ___  |  | || ||_  _| |_  _| | || |   / ___ `.   | || | |_   ___ `.  | |║" << endl;
    cout << "║| |   | |_  \\_|  | || |  \\ \\   / /   | || |  |_/___) |   | || |   | |   `. \\ | |║" << endl;
    cout << "║| |   |  _|      | || |   \\ \\ / /    | || |   .'____.'   | || |   | |    | | | |║" << endl;
    cout << "║| |  _| |_       | || |    \\ ' /     | || |  / /____     | || |  _| |___.' / | |║" << endl;
    cout << "║| | |_____|      | || |     \\_/      | || |  |_______|   | || | |________.'  | |║" << endl;
    cout << "║| |              | || |              | || |              | || |              | |║" << endl;
    cout << "║| '--------------' || '--------------' || '--------------' || '--------------' |║" << endl;
    cout << "║ '----------------'  '----------------'  '----------------'  '----------------' ║" << endl;
    cout << "║                             Finite Volume CFD Solver                           ║" << endl;
    cout << "║                                by Hoang Minh To                                ║" << endl;
    cout << "║                     Computational Fluid Dynamics Laboratory                    ║" << endl;
    cout << "║                         Aerospace Engineering Department                       ║" << endl;
    cout << "╠════════════════════════════════════════════════════════════════════════════════╣" << endl;
    cout << left_content;
    cout << setw(remaining_space) << " " << " ║" << endl;
    cout << "╚════════════════════════════════════════════════════════════════════════════════╝" << endl;
    std::cout << std::endl;

    vector<double> Q_init;
    vector<double> Q;
    MeshData mesh;
    Flow flow;
    Solver solver;
    Reconstruct recon;
    Time time;
    Flux flux;
    
    // Initialize the mesh and flow data
    cout << "==================================Initializing====================================" << endl;
    auto t0 = Clock::now();   // Start
    initialize(infile, mesh, flow, solver, recon, flux, time, Q_init, Q);

    cout << "Finished initializing." << endl;
    auto t1 = Clock::now();   // After initialization
    cout << "Time elapsed for initialize = " << chrono::duration<double>(t1 - t0).count() << " s\n";
    std::cout << std::endl;
    cout << "================================Simulation started================================" << endl;

    // // Run time integration
    // if (time.method == "implicit") {
    //     // Perform tasks for the implicit method
    //     // implicit_scheme(mesh, solver, flow, recon, flux, time, Q, Q_init);
    //     std::cerr << "Implicit method is not yet implemented." << std::endl;
    // } else {
    //     // Handle other methods or default case
    //     if (time.rk_steps == 2) {
            ssprk2(mesh, solver, flow, recon, flux, time, Q, Q_init);
    //     } else if (time.rk_steps == 3) {
    //         // Perform tasks for the explicit method
    //         ssprk3(mesh, solver, flow, recon, flux, time, Q, Q_init);
    //     }
    // }

    cout << "Simulation completed." << endl;
    auto t2 = Clock::now();   // After initialization
    cout << "Time elapsed simulation = " << chrono::duration<double>(t2 - t1).count() << " s\n";
    cout << "==============================Simulation Successful===============================" << endl;

    return 0;
}
