#ifndef SSPRK2_H
#define SSPRK2_H

#include <iostream>
#include <filesystem>

#include "io/meshread.h"
#include "io/initialize.h"
// #include "reconstruct/reconLinear.h"
// #include "reconstruct/reconGG.h"
#include "reconstruct/reconLS.h"
#include "flux/fluxcomp.h"
// #include "flux/visfluxcomp.h"
#include "rescomp.h"
// #include "io/writeout.h"
// #include "io/fnc.h"
#include "varini.h"


void ssprk2(const MeshData &mesh, const Solver &solver, const Flow &flow, const Reconstruct &recon, const Flux &flux, const Time &time, std::vector<double> &Q, const std::vector<double> &Q_in) {
    // // Allocate containers for left and right face states, fluxes, and residual.
    // Initialize the variables
    reconVars rv = init_reconVars(mesh);
    reconScraps rs = init_reconScraps(mesh);
    fluxVars fv = init_fluxVars(mesh);
    fluxScraps fs = init_fluxScraps(mesh);
    resVars resv = init_resVars(mesh);
    ioVars iv = init_ioVars(mesh);

    // std::filesystem::create_directory("sol");

    // // Open file for writing (in append mode)
    // // std::ofstream outfile("sol/res_log.dat", std::ios::app); // not sure why this doesnt work
    // std::ofstream outfile("sol/res_log.dat");
    // // Check if file is opened successfully
    // if (!outfile.is_open()) {
    //     std::cerr << "Error opening file!" << std::endl;
    // }
    // outfile << "variables=iter, res1, res2, res3, res4, Fx, Fy, CL, CD" << std::endl;

    // Time-stepping loop using SSP RK2 method.
    for (int step = 0; step <= solver.n_step; ++step) {
        // Stage 1: Compute the residual using the current state Q.
        // Solution reconstruction
        // if (recon.method == "linear") {
            // reconstruct_linear(mesh, Q, Q_in, flow, solver, rv);
        // }
        // else if (recon.method == "gauss-green") {
        //     reconstruct_gaussgreen(mesh, Q, Q_in, flow, solver, recon, rv, rs);
        // }
        // else if (recon.method == "least-square") {
            reconstruct_leastsquare(mesh, Q, Q_in, flow, solver, recon, rv, rs);
        // }
        // Compute fluxes
        // if (flow.type == 1) {
            compute_fluxes(mesh, flow, flux, rv, fv);
        // }
        // else if (flow.type == 2) {
            // compute_fluxes_vis(mesh, flow, flux, rv, rs, fv, fs);
        // }
        // Compute the residual
        compute_residual(mesh, solver, time, fv, resv);

        // Compute the intermediate state: Q_stage = Q^n + dt * L(Q^n)
        if (time.use_cfl == 1) {    // Use CFL condition
            if (time.local_dt == 0) { // Global time step
                double dt_glob = *std::min_element(resv.dt_local.begin(), resv.dt_local.end());
                for (int i = 0; i < mesh.n_cells; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        iv.Q1[4*i+j] = Q[4*i+j] + dt_glob * resv.Res[4*i+j];
                    }
                }
            }
            else {  // Local time step
                for (int i = 0; i < mesh.n_cells; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        iv.Q1[4*i+j] = Q[4*i+j] + resv.dt_local[i] * resv.Res[4*i+j];
                    }
                }
            }
        }
        else {  // Use fixed time step
            for (int i = 0; i < mesh.n_cells; ++i) {
                for (int j = 0; j < 4; ++j) {
                    iv.Q1[4*i+j] = Q[4*i+j] + time.dt * resv.Res[4*i+j];
                }
            }
        }

        // Stage 2: Recompute the residual at the intermediate state.
        // Solution reconstruction
        // if (recon.method == "linear") {
        //     reconstruct_linear(mesh, iv.Q1, Q_in, flow, solver, rv);
        // }
        // else if (recon.method == "gauss-green") {
        //     reconstruct_gaussgreen(mesh, iv.Q1, Q_in, flow, solver, recon, rv, rs);
        // }
        // else if (recon.method == "least-square") {
            reconstruct_leastsquare(mesh, iv.Q1, Q_in, flow, solver, recon, rv, rs);
        // }
        // Compute fluxes
        // if (flow.type == 1) {
            compute_fluxes(mesh, flow, flux, rv, fv);
        // }
        // else if (flow.type == 2) {
        //     compute_fluxes_vis(mesh, flow, flux, rv, rs, fv, fs);
        // }
        // Compute the residual
        compute_residual(mesh, solver, time, fv, resv);

        // Final update: Q^(n+1) = 0.5 * (Q^n + Q_stage + dt * L(Q_stage))
        if (time.use_cfl == 1) {    // Use CFL condition
            if (time.local_dt == 0) { // Global time step
                double dt_glob = *std::min_element(resv.dt_local.begin(), resv.dt_local.end());
                for (int i = 0; i < mesh.n_cells; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        Q[4*i+j] = 0.5 * Q[4*i+j] + 0.5 * (iv.Q1[4*i+j] + dt_glob * resv.Res[4*i+j]);
                    }
                }
            }
            else {  // Local time step
                for (int i = 0; i < mesh.n_cells; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        Q[4*i+j] = 0.5 * Q[4*i+j] + 0.5 * (iv.Q1[4*i+j] + resv.dt_local[i] * resv.Res[4*i+j]);
                    }
                }
            }
        }
        else {
            for (int i = 0; i < mesh.n_cells; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Q[4*i+j] = 0.5 * Q[4*i+j] + 0.5 * (iv.Q1[4*i+j] + time.dt * resv.Res[4*i+j]);
                }
            }
        }

        //  Print progress info every few steps.
        if (step % solver.m_step == 0) {
            // double CL, CD, Fx, Fy;
            // forceNcoef_cal(mesh, flow, Q, Q_in, dVdn, CL, CD, Fx, Fy, CP, TauW, Cf);
            std::cout << "Completed step " << step << " of " << solver.n_step << std::endl;
            // // Compute L2 norms for each column of the residual
            double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;

            for (int i = 0; i < mesh.n_cells; ++i) {
                res1 += std::pow(resv.Res[4*i + 0], 2);
                res2 += std::pow(resv.Res[4*i + 1], 2);
                res3 += std::pow(resv.Res[4*i + 2], 2);
                res4 += std::pow(resv.Res[4*i + 3], 2);
            }

            res1 = std::sqrt(res1);
            res2 = std::sqrt(res2);
            res3 = std::sqrt(res3);
            res4 = std::sqrt(res4);
  
            std::cout << "Residuals: "
                    << "res1 = " << res1 << ", "
                    << "res2 = " << res2 << ", "
                    << "res3 = " << res3 << ", "
                    << "res4 = " << res4 << std::endl;

            // // Save to file in the format: step res1 res2 res3 res4
            // outfile << step << " " << res1 << " " << res2 << " " << res3 << " " << res4 << " " << Fx << " " << Fy  << " " << CL << " " << CD << std::endl;                
        }

        // if (step % solver.o_step == 0) {
        //     // Compute Q_out
        //     write_output(mesh, Q, flow, solver, recon, Q_in, dQx, dQy, Q_out);
            
        //     // Solutions file
        //     Eigen::MatrixXd output(mesh.r_node.rows(), 6);
        //     output << mesh.r_node, Q_out;  // r_node (x, y) | Q_out (rho, rho*u, ...)

        //     std::string filename = "sol/Q_output_" + std::to_string(step) + ".dat";
        //     std::ofstream out(filename);
        //     if (out) {
        //         out << "TITLE = \"Solution output\"\n";
        //         out << "VARIABLES = \"X\" \"Y\" \"rho\" \"u\" \"v\" \"p\" \n";
        //         out << "ZONE T = \"0\" \n";
        //         out << "SOLUTIONTIME = "+ std::to_string(step) + "\n";
        //         out << "NODES = "+std::to_string(mesh.n_nodes) + ", ";
        //         out << "ELEMENTS = "+std::to_string(mesh.c2n_tri.rows()) + ", ";
        //         out << "ZONETYPE=FETriangle \n";
        //         out << "DATAPACKING=POINT \n" ;
        //         out << "DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE) \n";
        //         out << output << "\n";
        //         out << mesh.c2n_tri << "\n";
        //         out.close();
        //         std::cout << "Solutions written to " << filename << std::endl;
        //     } else {
        //         std::cerr << "Error writing output file!" << std::endl;
        //     }

        //     // Surface file
        //     std::string filename2 = "sol/surf_output_" + std::to_string(step) + ".dat";
        //     std::ofstream out2(filename2);
        //     Eigen::MatrixXd output2(mesh.n_fwalls, 5);
        //     output2 << mesh.r_w, CP, TauW, Cf;
        //     out2 << "variables=X, Y, CP, TauW, Cf \n";
        //     out2 << output2;
        //     out2.close();                   
        // } 
    }
}

#endif  // SSPRK2_H
