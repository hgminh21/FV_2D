#ifndef IMPLICIT_H
#define IMPLICIT_H

#include <Eigen/Dense>
#include <iostream>
#include <filesystem>

#include "io/meshread.h"
#include "io/initialize.h"
#include "reconstruct/reconLinear.h"
#include "reconstruct/reconGG.h"
#include "reconstruct/reconLS.h"
#include "flux/fluxcomp.h"
#include "flux/visfluxcomp.h"
#include "rescomp.h"
#include "implicit/resgrad.h"
#include "io/writeout.h"


#include "implicit/solverBiCG.h"
#include "implicit/solverConGrad.h"
#include "implicit/solverILU.h"
#include "implicit/solverLDLT.h"
#include "implicit/solverLU.h"

void implicit_scheme(const MeshData &mesh, const Solver &solver, const Flow &flow, const Reconstruct &recon, const Flux &flux, const Time &time, Eigen::MatrixXd &Q, const Eigen::Vector4d &Q_in) 
{
    // Allocate containers for left and right face states, fluxes, and residual.
    Eigen::MatrixXd Q_L = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd Q_R = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd dQx = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd dQy = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Q_out = Eigen::MatrixXd::Zero(mesh.n_nodes, 4);
    Eigen::MatrixXd dQt = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Q_max = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Q_min = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd phi = Eigen::MatrixXd::Ones(mesh.n_cells, 4);

    Eigen::VectorXd s_max_all(mesh.n_faces);
    Eigen::MatrixXd F(mesh.n_faces, 4);
    Eigen::MatrixXd Res(mesh.n_cells, 4);
    Eigen::VectorXd dt_local(mesh.n_cells);
    
    Eigen::MatrixXd Qx1_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Qx2_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Qy1_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Qy2_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);

    Eigen::MatrixXd Resx1_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Resx2_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Resy1_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd Resy2_temp = Eigen::MatrixXd::Zero(mesh.n_cells, 4);

    Eigen::MatrixXd F_viscous = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd Q_f = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd dQ_fx = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::MatrixXd dQ_fy = Eigen::MatrixXd::Zero(mesh.n_faces, 4);
    Eigen::VectorXd dVdn = Eigen::VectorXd::Zero(mesh.n_fwalls, 2);

    Eigen::MatrixXd dResx = Eigen::MatrixXd::Zero(mesh.n_cells, 4);
    Eigen::MatrixXd dResy = Eigen::MatrixXd::Zero(mesh.n_cells, 4);

    Eigen::MatrixXd A_im1 = Eigen::MatrixXd::Zero(mesh.n_cells, mesh.n_cells);
    Eigen::MatrixXd A_im2 = Eigen::MatrixXd::Zero(mesh.n_cells, mesh.n_cells);
    Eigen::MatrixXd A_im3 = Eigen::MatrixXd::Zero(mesh.n_cells, mesh.n_cells);
    Eigen::MatrixXd A_im4 = Eigen::MatrixXd::Zero(mesh.n_cells, mesh.n_cells);
    
    std::filesystem::create_directory("sol");

    // Open file for writing (in append mode)
    // std::ofstream outfile("sol/res_log.dat", std::ios::app);
    std::ofstream outfile("sol/res_log.dat");
    // Check if file is opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
    }

    // Time-stepping loop using implicit method
    for (int step = 0; step <= solver.n_step; ++step) {
        // Stage 1: Compute the residual using the current state Q.
        if (recon.method == "linear") {
            reconstruct_linear(mesh, Q, Q_L, Q_R, Q_in, flow, solver);
        }
        else if (recon.method == "gauss-green") {
            reconstruct_gaussgreen(mesh, Q, Q_L, Q_R, dQx, dQy, Q_in, flow, solver, recon, Qx1_temp, Qy1_temp, Q_max, Q_min, phi);
        }
        else if (recon.method == "least-square") {
            reconstruct_leastsquare(mesh, Q, Q_L, Q_R, dQx, dQy, Q_in, flow, solver, recon, Qx1_temp, Qx2_temp, Qy1_temp, Qy2_temp, Q_max, Q_min, phi);
        }
        if (flow.type == 1) {
            compute_fluxes(mesh, Q_L, Q_R, flow, flux, F, s_max_all);
        }
        else if (flow.type == 2) {
            compute_fluxes_vis(mesh, Q_L, Q_R, dQx, dQy, flow, flux, F, s_max_all, F_viscous, Q_f, dQ_fx, dQ_fy, dVdn);
        }
        compute_residual(mesh, F, s_max_all, solver, time, Res, dt_local);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(mesh.n_cells, mesh.n_cells);
        if (time.use_cfl == 1) {    // Use CFL condition
            if (time.local_dt == 0) { // Global time step
                double dt_glob = dt_local.minCoeff();
                I = 1.0 / dt_glob * I;
            }
            else {  // Local time step
                I = dt_local.cwiseInverse().asDiagonal();
            }
        }
        else {
            I = 1.0 / time.dt * I;
        }

        if (step == 0) {
            Q = Q + time.dt * Res;
        }
        else {
            res_reconstruct(mesh, Res, F, dResx, dResy, dQx, dQy, A_im1, A_im2, A_im3, A_im4, Resx1_temp, Resx2_temp, Resy1_temp, Resy2_temp, I);
            
            // solver_BiCG(A_im1, A_im2, A_im3, A_im4, Res, dQt);
            // solver_ConGrad(mesh, A_im1, A_im2, A_im3, A_im4, Res, dQt);
            solver_ILU(A_im1, A_im2, A_im3, A_im4, Res, dQt);
            // dQt = solver_LDLT(A_im1, A_im2, A_im3, A_im4, Res);
            // dQt = solver_LU(A_im1, A_im2, A_im3, A_im4, Res);

            Q = dQt + Q;
        }
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
                std::cout << "dQt norm: "
                            << dQt.col(0).norm() << ", "
                            << dQt.col(1).norm() << ", "
                            << dQt.col(2).norm() << ", "
                            << dQt.col(3).norm() << std::endl;
            }

            if (step % solver.o_step == 0) {
                // Compute Q_out
                write_output(mesh, Q, flow, solver, recon, Q_in, dQx, dQy, Q_out);
                
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
            } 

    }

}

#endif 