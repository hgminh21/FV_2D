#ifndef SSPRK2_H
#define SSPRK2_H

#include <Eigen/Dense>
#include <iostream>
#include <filesystem>

#include "io/meshread.h"
#include "io/initialize.h"
#include "reconstruct.h"
#include "flux/fluxcomp.h"
#include "flux/visfluxcomp.h"
#include "rescomp.h"
#include "io/writeout.h"
#include "io/fnc.h"

void ssprk2(const MeshData &mesh, const Solver &solver, const Flow &flow, Time &time, Eigen::MatrixXd &Q, const Eigen::Vector4d &Q_in) {
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
    Eigen::MatrixXd Q1 = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    
    Eigen::MatrixXd Qx1_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Qx2_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Qy1_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Qy2_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);

    Eigen::MatrixXd F_viscous = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd Q_f = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd dQ_fx = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd dQ_fy = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    
    Eigen::MatrixXd dVdn = Eigen::MatrixXd::Zero(mesh.n_fwalls, 2);
    Eigen::VectorXd CP = Eigen::VectorXd::Zero(mesh.n_fwalls);
    Eigen::VectorXd TauX = Eigen::VectorXd::Zero(mesh.n_fwalls);
    Eigen::VectorXd TauY = Eigen::VectorXd::Zero(mesh.n_fwalls);

    std::filesystem::create_directory("sol");

    // Open file for writing (in append mode)
    // std::ofstream outfile("sol/res_log.dat", std::ios::app);
    std::ofstream outfile("sol/res_log.dat");
    // Check if file is opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
    }
    outfile << "variables=iter, res1, res2, res3, res4, CL, CD" << std::endl;

    // Time-stepping loop using SSP RK2 method.
    for (int step = 0; step <= solver.n_step; ++step) {
        // Stage 1: Compute the residual using the current state Q.
        reconstruct(mesh, Q, Q_L, Q_R, dQx, dQy, Q_in, flow, solver, Qx1_temp, Qx2_temp, Qy1_temp, Qy2_temp);
        if (flow.type == 1) {
            compute_fluxes(mesh, Q_L, Q_R, flow, F, s_max_all);
        }
        else if (flow.type == 2) {
            compute_fluxes_vis(mesh, Q_L, Q_R, dQx, dQy, flow, F, s_max_all, F_viscous, Q_f, dQ_fx, dQ_fy, dVdn);
        }
        compute_residual(mesh, F, s_max_all, solver, time, Res, dt_local);

        // Compute the intermediate state: Q_stage = Q^n + dt * L(Q^n)
        if (time.use_cfl == 1) {    // Use CFL condition
            if (time.local_dt == 0) { // Global time step
                double dt_glob = dt_local.minCoeff();
                Q1 = Q + dt_glob * Res;
            }
            else {  // Local time step
                Q1 = Q + dt_local.asDiagonal() * Res;
            }
        }
        else {  // Use fixed time step
            Q1 = Q + time.dt * Res;
        }

        // Stage 2: Recompute the residual at the intermediate state.
        reconstruct(mesh, Q1, Q_L, Q_R, dQx, dQy, Q_in, flow, solver, Qx1_temp, Qx2_temp, Qy1_temp, Qy2_temp);
        if (flow.type == 1) {
            compute_fluxes(mesh, Q_L, Q_R, flow, F, s_max_all);
        }
        else if (flow.type == 2) {
            compute_fluxes_vis(mesh, Q_L, Q_R, dQx, dQy, flow, F, s_max_all, F_viscous, Q_f, dQ_fx, dQ_fy, dVdn);
        }
        compute_residual(mesh, F, s_max_all, solver, time, Res, dt_local);

        // Final update: Q^(n+1) = 0.5 * (Q^n + Q_stage + dt * L(Q_stage))
        if (time.use_cfl == 1) {    // Use CFL condition
            if (time.local_dt == 0) { // Global time step
                double dt_glob = dt_local.minCoeff();
                Q = 0.5 * Q + 0.5 * (Q1 + dt_glob * Res);
            }
            else {  // Local time step
                Q = 0.5 * Q + 0.5 * (Q1 + dt_local.asDiagonal() * Res);
            }
        }
        else {
            Q = 0.5 * Q + 0.5 * (Q1 + time.dt * Res);
        }

        //  Print progress info every few steps.
            if (step % solver.m_step == 0) {
                double CL, CD;
                forceNcoef_cal(mesh, flow, Q, Q_in, dVdn, CL, CD, CP, TauX, TauY);
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
                outfile << step << " " << res1 << " " << res2 << " " << res3 << " " << res4 << " " << CL << " " << CD << std::endl;                
            }

            if (step % solver.o_step == 0) {
                // Compute Q_out
                write_output(mesh, Q, flow, solver, Q_in, dQx, dQy, Q_out);
                
                // Solutions file
                Eigen::MatrixXd output(mesh.r_node.rows(), 6);
                output << mesh.r_node, Q_out;  // r_node (x, y) | Q_out (rho, rho*u, ...)

                std::string filename = "sol/Q_output_" + std::to_string(step) + ".dat";
                std::ofstream out(filename);
                if (out) {
                    out << "TITLE = \"Solution output\"\n";
                    out << "VARIABLES = \"X\" \"Y\" \"rho\" \"u\" \"v\" \"p\" \n";
                    out << "ZONE T = \"0\" \n";
                    out << "SOLUTIONTIME = "+ std::to_string(step) + "\n";
                    out << "NODES = "+std::to_string(mesh.n_nodes) + ", ";
                    out << "ELEMENTS = "+std::to_string(mesh.c2n_tri.rows()) + ", ";
                    out << "ZONETYPE=FETriangle \n";
                    out << "DATAPACKING=POINT \n" ;
                    out << "DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE) \n";
                    out << output << "\n";
                    out << mesh.c2n_tri << "\n";
                    out.close();
                    std::cout << "Solutions written to " << filename << std::endl;
                } else {
                    std::cerr << "Error writing output file!" << std::endl;
                }

                // Surface file
                std::string filename2 = "sol/surf_output_" + std::to_string(step) + ".dat";
                std::ofstream out2(filename2);
                Eigen::MatrixXd output2(mesh.n_fwalls, 5);
                output2 << mesh.r_w, CP, TauX, TauY;
                out2 << "variables=X, Y, CP, TauX, TauY \n";
                out2 << output2;
                out2.close();                   
            } 

    }
}

#endif  // SSPRK2_H
