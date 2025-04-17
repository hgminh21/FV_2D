#ifndef WRITEOUT_H
#define WRITEOUT_H

#include <Eigen>
#include "meshread.h"

using namespace Eigen;

void write_output(const MeshData &mesh,
                  const MatrixXd &Q,
                  double gamma,
                  MatrixXd &Q_out)
{
    int n_nodes = mesh.n_nodes;
    int n_faces = mesh.n_faces;

    const MatrixXi &f2n = mesh.f2n;
    const MatrixXi &f2c = mesh.f2c;

    Q_out = MatrixXd::Zero(n_nodes, 4);
    VectorXd n_shared = VectorXd::Zero(n_nodes);

    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c(i, 0) - 1;
        int c2 = f2c(i, 1) - 1;
        int n1 = f2n(i, 0) - 1;
        int n2 = f2n(i, 1) - 1;

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

    // Average by number of shared contributions
    Q_out = n_shared.cwiseInverse().asDiagonal() * Q_out;

    // Now compute primitive variables from conservative
    VectorXd rho = Q_out.col(0);
    VectorXd u = Q_out.col(1).array() / rho.array();
    VectorXd v = Q_out.col(2).array() / rho.array();
    VectorXd E = Q_out.col(3);

    VectorXd kinetic = 0.5 * rho.array() * (u.array().square() + v.array().square());
    VectorXd p = (E.array() - kinetic.array()) * (gamma - 1.0);

    // Overwrite Q_out with primitive variables: rho, u, v, p
    Q_out.col(0) = rho;
    Q_out.col(1) = u;
    Q_out.col(2) = v;
    Q_out.col(3) = p;
}

#endif // WRITEOUT_H