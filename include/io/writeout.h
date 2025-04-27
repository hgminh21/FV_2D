#ifndef WRITEOUT_H
#define WRITEOUT_H

#include <Eigen/Dense>
#include "io/meshread.h"
#include "io/initialize.h"

using namespace Eigen;

void write_output(const MeshData &mesh,
                  const MatrixXd &Q,
                  const Flow &flow,
                  const Solver &solver,
                  const Vector4d &Q_in,
                  const MatrixXd &dQx,
                  const MatrixXd &dQy,
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
    else { // second order reconstruction
        // // Temporary arrays for gradient estimation
        // MatrixXd Qx1_temp = MatrixXd::Zero(mesh.n_cells, 4);
        // MatrixXd Qx2_temp = MatrixXd::Zero(mesh.n_cells, 4);
        // MatrixXd Qy1_temp = MatrixXd::Zero(mesh.n_cells, 4);
        // MatrixXd Qy2_temp = MatrixXd::Zero(mesh.n_cells, 4);
    
        // for (int i = 0; i < mesh.n_faces; ++i) {
        //     int c1 = mesh.f2c(i, 0) - 1;
        //     int c2 = mesh.f2c(i, 1) - 1;
        //     const Vector2d rc1 = mesh.r_c.row(c1);
        //     const Vector2d rf = mesh.r_f.row(i);
        //     const Vector2d nf = mesh.n_f.row(i);
    
        //     RowVector4d dQ;
        //     double dx, dy;
    
        //     if (c2 >= 0) {
        //         const Vector2d rc2 = mesh.r_c.row(c2);
        //         dQ = Q.row(c2) - Q.row(c1);
        //         dx = rc2(0) - rc1(0);
        //         dy = rc2(1) - rc1(1);
    
        //         // Contribution to both cells
        //         auto update = [&](int c, double dx, double dy) {
        //             Qx1_temp.row(c) += dQ * dx * mesh.Iyy(c);
        //             Qx2_temp.row(c) += dQ * dy * mesh.Ixy(c);
        //             Qy1_temp.row(c) += dQ * dy * mesh.Ixx(c);
        //             Qy2_temp.row(c) += dQ * dx * mesh.Ixy(c);
        //         };
        //         update(c1, dx, dy);
        //         update(c2, dx, dy);
        //     } 
        //     else if (c2 == -2) { // Free stream condition
        //         dQ = Q_in.transpose() - Q.row(c1);
        //         dx = -2.0 * (rc1(0) - rf(0)) * nf(0);
        //         dy = -2.0 * (rc1(1) - rf(1)) * nf(1);
        //         Qx1_temp.row(c1) += dQ * dx * mesh.Iyy(c1);
        //         Qx2_temp.row(c1) += dQ * dy * mesh.Ixy(c1);
        //         Qy1_temp.row(c1) += dQ * dy * mesh.Ixx(c1);
        //         Qy2_temp.row(c1) += dQ * dx * mesh.Ixy(c1);
        //     } 
        //     else if (c2 == -1) { // Wall boundary
        //         double rhoi = Q(c1, 0);
        //         double ui = Q(c1, 1) / rhoi;
        //         double vi = Q(c1, 2) / rhoi;
        //         double pi = (Q(c1, 3) - 0.5 * rhoi * (ui * ui + vi * vi)) * (flow.gamma - 1.0);
    
        //         double vn = ui * nf(0) + vi * nf(1);
        //         double ug = ui - 2.0 * vn * nf(0);
        //         double vg = vi - 2.0 * vn * nf(1);
    
        //         RowVector4d Qg;
        //         Qg << rhoi, rhoi * ug, rhoi * vg, pi / (flow.gamma - 1.0) + 0.5 * rhoi * (ug * ug + vg * vg);
        //         dQ = Qg - Q.row(c1);
        //         dx = -2.0 * (rc1(0) - rf(0)) * nf(0);
        //         dy = -2.0 * (rc1(1) - rf(1)) * nf(1);
    
        //         Qx1_temp.row(c1) += dQ * dx * mesh.Iyy(c1);
        //         Qx2_temp.row(c1) += dQ * dy * mesh.Ixy(c1);
        //         Qy1_temp.row(c1) += dQ * dy * mesh.Ixx(c1);
        //         Qy2_temp.row(c1) += dQ * dx * mesh.Ixy(c1);
        //     }
        // }
    
        // // Final gradient computation
        // MatrixXd dQx = Qx1_temp - Qx2_temp;
        // MatrixXd dQy = Qy1_temp - Qy2_temp;
        double dfx1, dfx2, dfy1, dfy2;

        for (int i = 0; i < mesh.n_faces; ++i) {
            int c1 = mesh.f2c(i, 0) - 1;
            int c2 = mesh.f2c(i, 1) - 1;
            int n1 = mesh.f2n(i, 0) - 1;
            int n2 = mesh.f2n(i, 1) - 1;
    
            n_shared(n1) += 1;
            n_shared(n2) += 1;

            const Vector2d rc1 = mesh.r_c.row(c1);
            const Vector2d rn1 = mesh.r_node.row(n1);
            const Vector2d rn2 = mesh.r_node.row(n2);

            if (c2 >= 0) {
                const Vector2d rc2 = mesh.r_c.row(c2);

                dfx1 = rn1(0) - rc1(0);
                dfy1 = rn1(1) - rc1(1);
                dfx2 = rn1(0) - rc2(0);
                dfy2 = rn1(1) - rc2(1);
                Q_out.row(n1) += 0.5 * (Q.row(c1) + dQx.row(c1) * dfx1 + dQy.row(c1) * dfy1 
                                      + Q.row(c2) + dQx.row(c2) * dfx2 + dQy.row(c2) * dfy2);

                dfx1 = rn2(0) - rc1(0);
                dfy1 = rn2(1) - rc1(1);
                dfx2 = rn2(0) - rc2(0);
                dfy2 = rn2(1) - rc2(1);
                Q_out.row(n2) += 0.5 * (Q.row(c1) + dQx.row(c1) * dfx1 + dQy.row(c1) * dfy1 
                                      + Q.row(c2) + dQx.row(c2) * dfx2 + dQy.row(c2) * dfy2);
            } 
            else {
                dfx1 = rn1(0) - rc1(0);
                dfy1 = rn1(1) - rc1(1);
                Q_out.row(n1) += (Q.row(c1) + dQx.row(c1) * dfx1 + dQy.row(c1) * dfy1);

                dfx1 = rn2(0) - rc1(0);
                dfy1 = rn2(1) - rc1(1);
                Q_out.row(n2) += (Q.row(c1) + dQx.row(c1) * dfx1 + dQy.row(c1) * dfy1);
            }
            // else if (c2 == -1) { // Wall BC final state    
            //     const Vector2d nf = mesh.n_f.row(i);
            //     dfx1 = rn1(0) - rc1(0);
            //     dfy1 = rn1(1) - rc1(1);
            //     RowVector4d Q_L1 = Q.row(c1) + dQx.row(c1) * dfx1 + dQy.row(c1) * dfy1;

            //     double rho1 = Q_L1(0);
            //     double u1 = Q_L1(1) / rho1;
            //     double v1 = Q_L1(2) / rho1;
            //     double p1 = (Q_L1(3) - 0.5 * rho1 * (u1 * u1 + v1 * v1)) * (flow.gamma - 1.0);
    
            //     double vn1 = u1 * nf(0) + v1 * nf(1);
            //     double ug1 = u1 - 2.0 * vn1 * nf(0);
            //     double vg1 = v1 - 2.0 * vn1 * nf(1);
    
            //     RowVector4d Q_R1;
            //     Q_R1 << rho1, rho1 * ug1, rho1 * vg1, p1 / (flow.gamma - 1.0) + 0.5 * rho1 * (ug1 * ug1 + vg1 * vg1);
            //     Q_out.row(n1) += 0.5 * (Q_L1 + Q_R1);

            //     dfx2 = rn2(0) - rc1(0);
            //     dfy2 = rn2(1) - rc1(1);
            //     RowVector4d Q_L2 = Q.row(c1) + dQx.row(c1) * dfx2 + dQy.row(c1) * dfy2;

            //     double rho2 = Q_L2(0);
            //     double u2 = Q_L2(1) / rho2;
            //     double v2 = Q_L2(2) / rho2;
            //     double p2 = (Q_L2(3) - 0.5 * rho2 * (u2 * u2 + v2 * v2)) * (flow.gamma - 1.0);
    
            //     double vn2 = u2 * nf(0) + v2 * nf(1);
            //     double ug2 = u2 - 2.0 * vn2 * nf(0);
            //     double vg2 = v2 - 2.0 * vn2 * nf(1);
    
            //     RowVector4d Q_R2;
            //     Q_R2 << rho2, rho2 * ug2, rho2 * vg2, p2 / (flow.gamma - 1.0) + 0.5 * rho2 * (ug2 * ug2 + vg2 * vg2);
            //     Q_out.row(n2) += 0.5 * (Q_L2 + Q_R2);
            // } 
            // else if (c2 == -2) {
            //     dfx1 = rn1(0) - rc1(0);
            //     dfy1 = rn1(1) - rc1(1);
            //     Q_out.row(n1) += 0.5 * (Q.row(c1) + dQx.row(c1) * dfx1 + dQy.row(c1) * dfy1 + Q_in.transpose());

            //     dfx2 = rn2(0) - rc1(0);
            //     dfy2 = rn2(1) - rc1(1);
            //     Q_out.row(n2) += 0.5 * (Q.row(c1) + dQx.row(c1) * dfx2 + dQy.row(c1) * dfy2 + Q_in.transpose());
            // }
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