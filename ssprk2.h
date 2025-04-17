#ifndef SSPRK2_H
#define SSPRK2_H

#include <Eigen>
#include <iostream>
#include <filesystem>

// Include the other header files required for reconstruction, flux computation, and residual computation.
#include "meshread.h"
#include "initialize.h"
#include "reconstruct.h"
#include "fluxcomp.h"
#include "visfluxcomp.h"
#include "rescomp.h"
#include "writeout.h"

void ssprk2(const MeshData &mesh, const Solver &solver, const Flow &flow, Eigen::MatrixXd &Q, const Eigen::Vector4d &Q_in) {
    // Allocate containers for left and right face states, fluxes, and residual.
    Eigen::MatrixXd Q_L = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd Q_R = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd dQx = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd dQy = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Q_out = Eigen::MatrixXd::Zero(mesh.n_nodes, 4);
    Eigen::VectorXd s_max_all(mesh.n_faces);
    Eigen::MatrixXd F(mesh.n_faces, 4);
    Eigen::MatrixXd Res(mesh.n_cells, 4);
    Eigen::VectorXd dt_local(mesh.n_cells);
    
    std::filesystem::create_directory("sol");

    // Open file for writing (in append mode)
    // std::ofstream outfile("sol/res_log.dat", std::ios::app);
    std::ofstream outfile("sol/res_log.dat");
    // Check if file is opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
    }
    outfile << "TITLE = \"Residual log\"\n";
    outfile << "VARIABLES = \"Iteration\" \"Density\" \"X-momentum\" \"Y-momentum\" \"Total energy\" \n";
    outfile << "DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE) \n";

    // Time-stepping loop using SSP RK2 method.
    for (int step = 0; step <= solver.n_step; ++step) {
        // Stage 1: Compute the residual using the current state Q.
        reconstruct(mesh, Q, Q_L, Q_R, dQx, dQy, Q_in, flow, solver);
        if (flow.type == 1) {
            compute_fluxes(mesh, Q_L, Q_R, flow, F, s_max_all);
        }
        else if (flow.type == 2) {
            compute_fluxes_vis(mesh, Q_L, Q_R, dQx, dQy, flow, F, s_max_all);
        }
        compute_residual(mesh, F, s_max_all, solver, Res, dt_local);

        // Compute the intermediate state: Q_stage = Q^n + dt * L(Q^n)
        Eigen::MatrixXd Q1 = Q + dt_local.asDiagonal() * Res;

        // Stage 2: Recompute the residual at the intermediate state.
        reconstruct(mesh, Q1, Q_L, Q_R, dQx, dQy, Q_in, flow, solver);
        if (flow.type == 1) {
            compute_fluxes(mesh, Q_L, Q_R, flow, F, s_max_all);
        }
        else if (flow.type == 2) {
            compute_fluxes_vis(mesh, Q_L, Q_R, dQx, dQy, flow, F, s_max_all);
        }
        compute_residual(mesh, F, s_max_all, solver, Res, dt_local);

        // Final update: Q^(n+1) = 0.5 * (Q^n + Q_stage + dt * L(Q_stage))
        Q = 0.5 * Q + 0.5 * (Q1 + dt_local.asDiagonal() * Res);

        // (Optional) Print progress info every few steps.
            if (step % solver.m_step == 0) {
                std::cout << "Completed step " << step << " of " << solver.n_step << std::endl;
                // Compute L2 norms for each column of the residual
                double res1 = Res.col(0).norm();
                double res2 = Res.col(1).norm();
                double res3 = Res.col(2).norm();
                double res4 = Res.col(3).norm();   
                std::cout << "Residuals: "
                        << "res1 = " << res1 << ", "
                        << "res2 = " << res2 << ", "
                        << "res3 = " << res3 << ", "
                        << "res4 = " << res4 << std::endl;

                // Save to file in the format: step res1 res2 res3 res4
                outfile << step << " " << res1 << " " << res2 << " " << res3 << " " << res4 << std::endl;
            }

            if (step % solver.o_step == 0) {
                // Compute Q_out
                write_output(mesh, Q, flow, solver, Q_out);
                // Concatenate r_node and Q_out side-by-side (column-wise)
                Eigen::MatrixXd output(mesh.r_node.rows(), 6);
                output << mesh.r_node, Q_out;  // r_node (x, y) | Q_out (rho, rho*u, ...)

                std::string filename = "sol/Q_output_" + std::to_string(step) + ".dat";
                std::ofstream out(filename);
                if (out) {
                    out << "TITLE = \"Solution output\"\n";
                    out << "VARIABLES = \"X\" \"Y\" \"rho\" \"u\" \"v\" \"p\" \n";
                    out << "ZONE T = \"0\" \n";
                    out << "SOLUTIONTIME = "+ std::to_string(step) + "\n";
                    out << "DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE) \n";
                    out << output << "\n";
                    out.close();
                    std::cout << "Solutions written to " << filename << std::endl;
                } else {
                    std::cerr << "Error writing output file!" << std::endl;
                }
            } 

    }
}

#endif  // SSPRK2_H
