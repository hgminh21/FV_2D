#ifndef RESGRAD_H
#define RESGRAD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <array>
#include <iostream>

#include "meshread.h"
#include "initialize.h"

inline void res_reconstruct(const MeshData &mesh, 
                            const Eigen::MatrixXd &Res,
                            const Eigen::MatrixXd &F,
                            Eigen::MatrixXd &dResx,
                            Eigen::MatrixXd &dResy,
                            const Eigen::MatrixXd &dQx,
                            const Eigen::MatrixXd &dQy,
                            Eigen::MatrixXd &A_im1,
                            Eigen::MatrixXd &A_im2,
                            Eigen::MatrixXd &A_im3,
                            Eigen::MatrixXd &A_im4,
                            Eigen::MatrixXd &Resx1_temp,
                            Eigen::MatrixXd &Resx2_temp,
                            Eigen::MatrixXd &Resy1_temp,
                            Eigen::MatrixXd &Resy2_temp,
                            const Eigen::MatrixXd &I) 
{
    // Zero out matrices and temporaries
    A_im1.setZero();
    A_im2.setZero();
    A_im3.setZero();
    A_im4.setZero();
    Resx1_temp.setZero();
    Resx2_temp.setZero();
    Resy1_temp.setZero();
    Resy2_temp.setZero();

    const double eps = 1e-12;

    // Accumulate residual gradients
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;
        int c2 = mesh.f2c(i, 1) - 1;

        const Eigen::Vector2d rc1 = mesh.r_c.row(c1);
        const Eigen::Vector2d rf  = mesh.r_f.row(i);
        const Eigen::Vector2d nf  = mesh.n_f.row(i);

        Eigen::RowVector4d dRes;
        double dx, dy;

        if (c2 >= 0) {
            const Eigen::Vector2d rc2 = mesh.r_c.row(c2);
            dRes = Res.row(c2) - Res.row(c1);
            dx   = rc2.x() - rc1.x();
            dy   = rc2.y() - rc1.y();

            auto update = [&](int c, double dx_, double dy_) {
                Resx1_temp.row(c).noalias() += dRes * dx_ * mesh.Iyy(c);
                Resx2_temp.row(c).noalias() += dRes * dy_ * mesh.Ixy(c);
                Resy1_temp.row(c).noalias() += dRes * dy_ * mesh.Ixx(c);
                Resy2_temp.row(c).noalias() += dRes * dx_ * mesh.Ixy(c);
            };
            update(c1, dx, dy);
            update(c2, dx, dy);
        } 
        else if (c2 == -2) {
            // Free-stream boundary
            dRes = -Res.row(c1);
            dx   = -2.0 * (rc1.x() - rf.x()) * nf.x();
            dy   = -2.0 * (rc1.y() - rf.y()) * nf.y();
            Resx1_temp.row(c1).noalias() += dRes * dx * mesh.Iyy(c1);
            Resx2_temp.row(c1).noalias() += dRes * dy * mesh.Ixy(c1);
            Resy1_temp.row(c1).noalias() += dRes * dy * mesh.Ixx(c1);
            Resy2_temp.row(c1).noalias() += dRes * dx * mesh.Ixy(c1);
        } 
        else if (c2 == -1) {
            // Wall boundary
            Eigen::RowVector4d Resg = (F.row(i) * mesh.A(i)) / mesh.V(c1);
            dRes = Resg - Res.row(c1);
            dx   = -2.0 * (rc1.x() - rf.x()) * nf.x();
            dy   = -2.0 * (rc1.y() - rf.y()) * nf.y();
            Resx1_temp.row(c1).noalias() += dRes * dx * mesh.Iyy(c1);
            Resx2_temp.row(c1).noalias() += dRes * dy * mesh.Ixy(c1);
            Resy1_temp.row(c1).noalias() += dRes * dy * mesh.Ixx(c1);
            Resy2_temp.row(c1).noalias() += dRes * dx * mesh.Ixy(c1);
        }
    }

    // Compute final gradients
    dResx = Resx1_temp - Resx2_temp;
    dResy = Resy1_temp - Resy2_temp;

    // Build implicit Jacobian contributions per face
    std::array<Eigen::MatrixXd*,4> Aims = {{&A_im1, &A_im2, &A_im3, &A_im4}};
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;
        int c2 = mesh.f2c(i, 1) - 1;
        if (c2 < 0) continue;

        for (int j = 0; j < 4; ++j) {
            auto& Aim = *Aims[j];
            double dQx1 = dQx(c1,j), dQx2 = dQx(c2,j);
            double dQy1 = dQy(c1,j), dQy2 = dQy(c2,j);
            double dRx1 = dResx(c1,j), dRx2 = dResx(c2,j);
            double dRy1 = dResy(c1,j), dRy2 = dResy(c2,j);

            bool smallX1 = std::abs(dQx1) < eps;
            bool smallY1 = std::abs(dQy1) < eps;
            bool smallX2 = std::abs(dQx2) < eps;
            bool smallY2 = std::abs(dQy2) < eps;

            // Initialize to zero
            Aim(c1, c1) = 0.0;
            Aim(c1, c2) = 0.0;

            // First gradient (cell 1 side)
            if (!smallX1 && !smallY1) {
                Aim(c1, c1) += 0.5 * (dRx1 / dQx1 + dRy1 / dQy1);
            } else {
                if (!smallX1) Aim(c1, c1) += dRx1 / dQx1;
                if (!smallY1) Aim(c1, c1) += dRy1 / dQy1;
            }

            // Second gradient (cell 2 side)
            if (!smallX2 && !smallY2) {
                Aim(c1, c2) += 0.5 * (dRx2 / dQx2 + dRy2 / dQy2);
            } else {
                if (!smallX2) Aim(c1, c2) += dRx2 / dQx2;
                if (!smallY2) Aim(c1, c2) += dRy2 / dQy2;
            }

            // Symmetric mirror
            Aim(c2, c2) = Aim(c1, c2);
            Aim(c2, c1) = Aim(c1, c1);
        }
    }

    // Form final implicit matrices: I - A_im*
    A_im1 = I - A_im1;
    A_im2 = I - A_im2;
    A_im3 = I - A_im3;
    A_im4 = I - A_im4;

    // Optional: check for NaNs/Infs
    if (!A_im1.array().isFinite().all()) std::cerr << "Non-finite in A_im1\n";
    if (!A_im2.array().isFinite().all()) std::cerr << "Non-finite in A_im2\n";
    if (!A_im3.array().isFinite().all()) std::cerr << "Non-finite in A_im3\n";
    if (!A_im4.array().isFinite().all()) std::cerr << "Non-finite in A_im4\n";
}

#endif // RESGRAD_H
