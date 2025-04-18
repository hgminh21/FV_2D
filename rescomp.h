#ifndef RESCOMP_H
#define RESCOMP_H

#include <Eigen>
#include "meshread.h"
#include "initialize.h"

using namespace Eigen;

void compute_residual(const MeshData &mesh,
                    const MatrixXd &F,
                    const VectorXd &s_max_all,
                    const Solver &solver,
                    MatrixXd &Res,
                    VectorXd &dt_local)
{
    Res.setZero();
    dt_local.setZero();
    // MatrixXd Res_local = MatrixXd::Zero(mesh.n_cells, 4);
    // VectorXd dt_sum = VectorXd::Zero(mesh.n_cells);
    
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;  // convert to zero-index
        int c2 = mesh.f2c(i, 1) - 1;  // convert to zero-index
        
        Res.row(c1) -= (F.row(i) * mesh.A(i)) / mesh.V(c1);
        dt_local(c1) += s_max_all(i) * mesh.A(i) / mesh.V(c1);
        // Res_local.row(c1) -= (F.row(i) * mesh.A(i)) / mesh.V(c1);
        // dt_sum(c1) += s_max_all(i) * mesh.A(i) / mesh.V(c1);

        if (c2 >= 0) {
            Res.row(c2) += (F.row(i) * mesh.A(i)) / mesh.V(c2);
            dt_local(c2) += s_max_all(i) * mesh.A(i) / mesh.V(c2);
            // Res_local.row(c2) += (F.row(i) * mesh.A(i)) / mesh.V(c2);
            // dt_sum(c2) += s_max_all(i) * mesh.A(i) / mesh.V(c2);
        }
    }
    // Res = Res_local;
    dt_local = solver.CFL * dt_local.cwiseInverse();
}

#endif  // RESCOMP_H
