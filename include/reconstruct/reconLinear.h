#ifndef RECONLINEAR_H
#define RECONLINEAR_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "io/meshread.h"
#include "io/initialize.h"

using namespace Eigen;

void reconstruct_linear(const MeshData &mesh,
                        const MatrixXd &Q,
                        MatrixXd &Q_L,
                        MatrixXd &Q_R,
                        const Vector4d &Q_in,
                        const Flow &flow,
                        const Solver &solver)
{

    // First-order reconstruction
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i,0) - 1;
        int c2 = mesh.f2c(i,1) - 1;
        // cache cell states
        const RowVector4d Q1 = Q.row(c1);
        RowVector4d Q2 = RowVector4d::Zero();
        Q_L.row(i) = Q1;
        if (c2 >= 0) {
            Q2 = Q.row(c2);
            Q_R.row(i) = Q2;
        }
        // wall BC
        if (c2 == -1) {
            double rhoL = Q1(0);
            double uL = Q1(1)/rhoL, vL = Q1(2)/rhoL;
            double pL = (Q1(3) - 0.5*rhoL*(uL*uL+vL*vL))*(flow.gamma-1.0);
            double nx = mesh.n_f(i,0), ny = mesh.n_f(i,1);
            double vn = uL*nx + vL*ny;
            double uR = uL - 2.0*vn*nx;
            double vR = vL - 2.0*vn*ny;
            Q_R(i,0) = rhoL;
            Q_R(i,1) = rhoL*uR;
            Q_R(i,2) = rhoL*vR;
            Q_R(i,3) = pL/(flow.gamma - 1.0) + 0.5*rhoL*(uR*uR+vR*vR);
        }
        // free-stream BC
        else if (c2 == -2) {
            Q_R.row(i) = Q_in.transpose();
        }
    }
}

#endif // RECONLINEAR_H