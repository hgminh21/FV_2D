#ifndef RECONLINEAR_H
#define RECONLINEAR_H

#include <fstream>
#include <iostream>

#include "io/meshread.h"
#include "io/initialize.h"
#include "explicit/varini.h"


void reconstruct_linear(const MeshData &mesh,
                        const std::vector<double> &Q,
                        const std::vector<double> &Q_in,
                        const Flow &flow,
                        const Solver &solver,
                        reconVars &rv)
{

    // First-order reconstruction
    for (int i = 0; i < mesh.n_faces; ++i) {
        // cell indices
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;
        
        // cache cell states
        double rho1 = Q[4*c1];
        double rhou1 = Q[4*c1+1];
        double rhov1 = Q[4*c1+2];
        double E1 = Q[4*c1+3];

        // set left state
        rv.Q_L[4*i] = rho1;
        rv.Q_L[4*i+1] = rhou1;
        rv.Q_L[4*i+2] = rhov1;
        rv.Q_L[4*i+3] = E1;

        // set right state
        if (c2 >= 0) {
            rv.Q_R[4*i] = Q[4*c2];
            rv.Q_R[4*i+1] = Q[4*c2+1];
            rv.Q_R[4*i+2] = Q[4*c2+2];
            rv.Q_R[4*i+3] = Q[4*c2+3];
        }
        // wall BC
        else if (c2 == -1) {
            double rhoL = rho1;
            double uL = rhou1/rhoL, vL = rhov1/rhoL;
            double pL = (E1 - 0.5*rhoL*(uL*uL+vL*vL))*(flow.gamma-1.0);
            double nx = mesh.n_f[2*i], ny = mesh.n_f[2*i+1];
            double vn = uL*nx + vL*ny;
            double uR = uL - 2.0*vn*nx;
            double vR = vL - 2.0*vn*ny;
            rv.Q_R[4*i] = rhoL;
            rv.Q_R[4*i+1] = rhoL*uR;
            rv.Q_R[4*i+2] = rhoL*vR;
            rv.Q_R[4*i+3] = pL/(flow.gamma - 1.0) + 0.5*rhoL*(uR*uR+vR*vR);
        }
        // free-stream BC
        else if (c2 == -2) {
            rv.Q_R[4*i] = Q_in[0];
            rv.Q_R[4*i+1] = Q_in[1];
            rv.Q_R[4*i+2] = Q_in[2];
            rv.Q_R[4*i+3] = Q_in[3];
        }
    }
}

#endif // RECONLINEAR_H