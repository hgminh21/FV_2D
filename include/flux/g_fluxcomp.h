#ifndef G_FLUXCOMP_H
#define G_FLUXCOMP_H

#include <cmath>
#include "io/initialize.h"
#include "explicit/g_varini.h"
#include "explicit/meshcopy.h"

class ComputeFluxes {
public:
    DeviceMeshData dMesh;
    const DeviceFlow* dflow;
    const DeviceReconVars* d_rv;
    DeviceFluxVars d_fv;
    const int* n_faces;

    deviceFunction void operator()(const unsigned int i) const {
        if (i >= *n_faces) {
            // std::cout << "this bih out of bound" << std::endl;
            return;
        }
        double nx = dMesh.d_n_f[2*i];
        double ny = dMesh.d_n_f[2*i+1];
        double gamma = *(dflow->d_gamma);
        // Left state
        double rhoL = d_rv->Q_L[4*i];
        double uL = d_rv->Q_L[4*i+1] / rhoL;
        double vL = d_rv->Q_L[4*i+2] / rhoL;
        double EL = d_rv->Q_L[4*i+3];
        double pL = ((EL - 0.5 * rhoL * (uL*uL + vL*vL)) * (gamma-1.0));

        // Right state
        double rhoR = d_rv->Q_R[4*i];
        double uR = d_rv->Q_R[4*i+1] / rhoR;
        double vR = d_rv->Q_R[4*i+2] / rhoR;
        double ER = d_rv->Q_R[4*i+3];
        double pR = (ER - 0.5 * rhoR * (uR*uR + vR*vR)) * (gamma-1.0);

        double vnL = uL*nx + vL*ny;
        double vnR = uR*nx + vR*ny;
        double cL = sqrt(gamma * pL / rhoL);
        double cR = sqrt(gamma * pR / rhoR);

        double fL[4] = {rhoL*vnL, rhoL*uL*vnL + pL*nx, rhoL*vL*vnL + pL*ny, vnL*(EL + pL)};
        double fR[4] = {rhoR*vnR, rhoR*uR*vnR + pR*nx, rhoR*vR*vnR + pR*ny, vnR*(ER + pR)};

        double s_max;
            s_max = 0.5 * (fabs(vnL)+fabs(vnR)) + 0.5*(cL+cR);
        d_fv.s_max_all[i] = s_max;
        for (int j=0; j<4; ++j)
            d_fv.F[4*i+j] = 0.5*(fL[j]+fR[j]) - 0.5*s_max*(d_rv->Q_R[4*i+j] - d_rv->Q_L[4*i+j]);
    }
};

#endif // G_FLUXCOMP_H
