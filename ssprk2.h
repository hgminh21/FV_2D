#ifndef SSPRK2_H
#define SSPRK2_H

#include <Eigen>
#include <iostream>

// Include the other header files required for reconstruction, flux computation, and residual computation.
#include "reconstruct.h"
#include "fluxcomp.h"
#include "rescomp.h"
#include "meshread.h"

// The SSP RK2 function performs the time-stepping loop for updating Q.
// Parameters:
//   mesh    : MeshData structure containing geometric and connectivity data.
//   Q       : Cell state matrix (n_cells x 4). This matrix is updated in place.
//   Q_in    : Prescribed state for boundary conditions (Vector4d).
//   gamma   : Specific heat ratio.
//   dt      : Time step size.
//   n_steps : Number of time steps to perform.
void ssprk2(const MeshData &mesh, Eigen::MatrixXd &Q, const Eigen::Vector4d &Q_in, double gamma, double CFL, int n_steps) {
    // Allocate containers for left and right face states, fluxes, and residual.
    Eigen::MatrixXd Q_L = Eigen::MatrixXd::Zero(mesh.f2c.rows(), 4);
    Eigen::MatrixXd Q_R = Eigen::MatrixXd::Zero(mesh.f2c.rows(), 4);
    Eigen::VectorXd s_max_all(mesh.f2c.rows());
    Eigen::MatrixXd F(mesh.f2c.rows(), 4);
    Eigen::MatrixXd Res(mesh.V.rows(), 4);
    Eigen::VectorXd dt_local(mesh.V.rows());
    
    // Time-stepping loop using SSP RK2 method.
    for (int step = 0; step < n_steps; ++step) {
        // Stage 1: Compute the residual using the current state Q.
        reconstruct(mesh.f2c, Q, Q_L, Q_R, mesh.n_f, Q_in, gamma);
        compute_fluxes(Q_L, Q_R, mesh.n_f, gamma, F, s_max_all);
        compute_residual(mesh.f2c, mesh.A, mesh.V, F, s_max_all, CFL, Res, dt_local);
        
        // Compute the intermediate state: Q_stage = Q^n + dt * L(Q^n)
        Eigen::MatrixXd Q1 = Q + dt_local.asDiagonal() * Res;
        
        // Stage 2: Recompute the residual at the intermediate state.
        reconstruct(mesh.f2c, Q1, Q_L, Q_R, mesh.n_f, Q_in, gamma);
        compute_fluxes(Q_L, Q_R, mesh.n_f, gamma, F, s_max_all);
        compute_residual(mesh.f2c, mesh.A, mesh.V, F, s_max_all, CFL, Res, dt_local);
        
        // Final update: Q^(n+1) = 0.5 * (Q^n + Q_stage + dt * L(Q_stage))
        Q = 0.5 * Q + 0.5 * (Q1 + dt_local.asDiagonal() * Res);

        // (Optional) Print progress info every few steps.
            if (step % 10 == 0) {
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
    }
}

#endif // SSPRK2_H
