#ifndef SSPRK2_H
#define SSPRK2_H

#include <Eigen>
#include <iostream>
#include <filesystem>

// Include the other header files required for reconstruction, flux computation, and residual computation.
#include "reconstruct.h"
#include "fluxcomp.h"
#include "rescomp.h"
#include "meshread.h"
#include "writeout.h"

// The SSP RK2 function performs the time-stepping loop for updating Q.
// Parameters:
//   mesh    : MeshData structure containing geometric and connectivity data.
//   Q       : Cell state matrix (n_cells x 4). This matrix is updated in place.
//   Q_in    : Prescribed state for boundary conditions (Vector4d).
//   gamma   : Specific heat ratio.
//   dt      : Time step size.
//   n_steps : Number of time steps to perform.
void ssprk2(const MeshData &mesh, Eigen::MatrixXd &Q, const Eigen::Vector4d &Q_in, double gamma, double CFL, int n_steps, int order) {
    // Allocate containers for left and right face states, fluxes, and residual.
    Eigen::MatrixXd Q_L = Eigen::MatrixXd::Zero(mesh.f2c.rows(), 4);
    Eigen::MatrixXd Q_R = Eigen::MatrixXd::Zero(mesh.f2c.rows(), 4);
    Eigen::MatrixXd dQ_L = Eigen::MatrixXd::Zero(mesh.f2c.rows(), 4);
    Eigen::MatrixXd dQ_R = Eigen::MatrixXd::Zero(mesh.f2c.rows(), 4);
    Eigen::MatrixXd Q_out = Eigen::MatrixXd::Zero(mesh.r_node.rows(), 4);
    Eigen::VectorXd s_max_all(mesh.f2c.rows());
    Eigen::MatrixXd F(mesh.f2c.rows(), 4);
    Eigen::MatrixXd Res(mesh.V.rows(), 4);
    Eigen::VectorXd dt_local(mesh.V.rows());
    
    std::filesystem::create_directory("sol");

    // Time-stepping loop using SSP RK2 method.
    for (int step = 0; step < n_steps; ++step) {
        // Stage 1: Compute the residual using the current state Q.
        reconstruct(mesh.f2c, Q, mesh.r_f, mesh.r_c, mesh.Ixx, mesh.Iyy, mesh.Ixy, mesh.delta, Q_L, Q_R, dQ_L, dQ_R, mesh.n_f, Q_in, gamma, order);
        compute_fluxes(Q_L, Q_R, mesh.n_f, gamma, F, s_max_all);
        compute_residual(mesh.f2c, mesh.A, mesh.V, F, s_max_all, CFL, Res, dt_local);
        writeMatrixToFile(Q_L, "Q_L.txt");
        writeMatrixToFile(Q_R, "Q_R.txt");

        // Compute the intermediate state: Q_stage = Q^n + dt * L(Q^n)
        Eigen::MatrixXd Q1 = Q + dt_local.asDiagonal() * Res;

        // Stage 2: Recompute the residual at the intermediate state.
        reconstruct(mesh.f2c, Q1, mesh.r_f, mesh.r_c, mesh.Ixx, mesh.Iyy, mesh.Ixy, mesh.delta, Q_L, Q_R, dQ_L, dQ_R, mesh.n_f, Q_in, gamma, order);
        compute_fluxes(Q_L, Q_R, mesh.n_f, gamma, F, s_max_all);
        compute_residual(mesh.f2c, mesh.A, mesh.V, F, s_max_all, CFL, Res, dt_local);
        writeMatrixToFile(Q_L, "Q_L2.txt");
        writeMatrixToFile(Q_R, "Q_R2.txt");

        // Final update: Q^(n+1) = 0.5 * (Q^n + Q_stage + dt * L(Q_stage))
        Q = 0.5 * Q + 0.5 * (Q1 + dt_local.asDiagonal() * Res);

        // (Optional) Print progress info every few steps.
            if (step % 20 == 0) {
                std::cout << "Completed step " << step << " of " << n_steps << std::endl;
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
            }

            // if (step % 100 == 0) {
            //     // Compute Q_out
            //     write_output(mesh, Q, gamma, Q_out);
            //     // Concatenate r_node and Q_out side-by-side (column-wise)
            //     Eigen::MatrixXd output(mesh.r_node.rows(), 6);
            //     output << mesh.r_node, Q_out;  // r_node (x, y) | Q_out (rho, rho*u, ...)

            //     std::string filename = "sol/Q_output_" + std::to_string(step) + ".dat";
            //     std::ofstream out(filename);
            //     if (out) {
            //         out << "# x y rho rho*u rho*v rho*E\n";
            //         out << output << "\n";
            //         out.close();
            //         std::cout << "Q written to " << filename << std::endl;
            //     } else {
            //         std::cerr << "Error writing output file!" << std::endl;
            //     }
            // } 

    }
}

#endif // SSPRK2_H
