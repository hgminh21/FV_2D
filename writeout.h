#ifndef WRITEOUT_H
#define WRITEOUT_H

#include <Eigen>
#include "meshread.h"
#include "initialize.h"

using namespace Eigen;

void write_output(const MeshData &mesh,
                  const MatrixXd &Q,
                  const Flow &flow,
                  const Solver &solver,
                  MatrixXd &Q_out)
{
    Q_out = MatrixXd::Zero(mesh.n_nodes, 4);
    VectorXd n_shared = VectorXd::Zero(mesh.n_nodes);
    if (solver.order == 1) {
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;
            int n1 = mesh.f2n(i, 0) - 1;
            int n2 = mesh.f2n(i, 1) - 1;
    
            n_shared(n1) += 1;
            n_shared(n2) += 1;
    
            if (c2 >= 0) {
                Q_out.row(n1) += 0.5 * (Q.row(c1) + Q.row(c2));
                Q_out.row(n2) += 0.5 * (Q.row(c1) + Q.row(c2));
            } else {
                Q_out.row(n1) += Q.row(c1);
                Q_out.row(n2) += Q.row(c1);  // use only c1 when c2 is invalid
            }
        }
    }
    else if (solver.order == 2) {
        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;
            int n1 = mesh.f2n(i, 0) - 1;
            int n2 = mesh.f2n(i, 1) - 1;
    
            n_shared(n1) += 1;
            n_shared(n2) += 1;
    
            if (c2 >= 0) {
                Q_out.row(n1) += 0.5 * (Q.row(c1) + Q.row(c2));
                Q_out.row(n2) += 0.5 * (Q.row(c1) + Q.row(c2));
            } else {
                Q_out.row(n1) += Q.row(c1);
                Q_out.row(n2) += Q.row(c1);  // use only c1 when c2 is invalid
            }
        }
    }
    // Average by number of shared contributions
    Q_out = n_shared.cwiseInverse().asDiagonal() * Q_out;

    // Now compute primitive variables from conservative
    VectorXd rho = Q_out.col(0);
    VectorXd u = Q_out.col(1).array() / rho.array();
    VectorXd v = Q_out.col(2).array() / rho.array();
    VectorXd E = Q_out.col(3);

    VectorXd kinetic = 0.5 * rho.array() * (u.array().square() + v.array().square());
    VectorXd p = (E.array() - kinetic.array()) * (flow.gamma - 1.0);

    // Overwrite Q_out with primitive variables: rho, u, v, p
    Q_out.col(0) = rho;
    Q_out.col(1) = u;
    Q_out.col(2) = v;
    Q_out.col(3) = p;
}

#endif // WRITEOUT_H