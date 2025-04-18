#ifndef RESCOMP_H
#define RESCOMP_H

#include <Eigen/Dense>
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
    
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;  // convert to zero-index
        int c2 = mesh.f2c(i, 1) - 1;  // convert to zero-index
        
        Res.row(c1) -= (F.row(i) * mesh.A(i)) / mesh.V(c1);
        dt_local(c1) += s_max_all(i) * mesh.A(i) / mesh.V(c1);

        if (c2 >= 0) {
            Res.row(c2) += (F.row(i) * mesh.A(i)) / mesh.V(c2);
            dt_local(c2) += s_max_all(i) * mesh.A(i) / mesh.V(c2);
        }
    }
    dt_local = solver.CFL * dt_local.cwiseInverse();
}

#endif  // RESCOMP_H
