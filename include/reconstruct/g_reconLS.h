#ifndef G_RECONLS_H
#define G_RECONLS_H  

#include "explicit/g_varini.h"
#include "explicit/meshcopy.h"
#include "io/initialize.h"

// Compute per-cell gradients using neighbor info
class ComputeCellGradients {
public:
    DeviceMeshData dMesh;
    const double* d_Q;
    const double* d_Q_in;
    const DeviceFlow* d_flow;
    DeviceReconScraps d_rs;
    const int* n_cells;

    deviceFunction void operator()(const unsigned int c) const {
        if (c >= *n_cells) {
            // std::cout << "this bih out of bound" << std::endl;
            return; // out of bounds
        }
        
        int start = dMesh.d_c2f_offset[c];
        int end   = dMesh.d_c2f_offset[c+1];

        double gradX[4] = {0.0, 0.0, 0.0, 0.0};
        double gradY[4] = {0.0, 0.0, 0.0, 0.0};
        for (int f = start; f < end; ++ f) {
            int i = dMesh.d_c2f_flat[f];
            int c1 = dMesh.d_f2c[2*i]-1;
            int c2 = dMesh.d_f2c[2*i+1]-1;
            
            double x1 = dMesh.d_r_c[2*c1], y1 = dMesh.d_r_c[2*c1+1];
            double xf = dMesh.d_r_f[2*i], yf = dMesh.d_r_f[2*i+1];
            double nx = dMesh.d_n_f[2*i], ny = dMesh.d_n_f[2*i+1];
            double Q1[4], Q2[4];
            double dx, dy;
            for(int j=0;j<4;j++) Q1[j] = d_Q[4*c1 + j];
            if (c2 >= 0) {
                for(int j=0;j<4;j++) Q2[j] = d_Q[4*c2 + j];
            }
            double dQ[4] = {0.0, 0.0, 0.0, 0.0};
            if (c2 >= 0) {
                dx = dMesh.d_r_c[2*c2] - x1;
                dy = dMesh.d_r_c[2*c2+1] - y1;
                // compute gradient
                for (int j = 0; j < 4; ++j) {
                    dQ[j] = Q2[j] - Q1[j];
                }
            } else if (c2 == -2) { // free-stream
                dx = -2.0 * (x1 - xf) * nx;
                dy = -2.0 * (y1 - yf) * ny;
                for (int j = 0; j < 4; ++j) {
                    dQ[j] = d_Q_in[j] - Q1[j];
                }
            } else { // wall BC
                double rhoL = Q1[0];
                double uL = Q1[1] / rhoL;
                double vL = Q1[2] / rhoL;
                double pL = (Q1[3] - 0.5 * rhoL * (uL * uL + vL * vL)) * (d_flow->d_gamma[0] - 1.0);
                double vn = uL * nx + vL * ny;
                double ug = uL - 2.0 * vn * nx;
                double vg = vL - 2.0 * vn * ny;
                double Qg[4];
                Qg[0] = rhoL;
                Qg[1] = rhoL * ug;
                Qg[2] = rhoL * vg;
                Qg[3] = pL / (d_flow->d_gamma[0] - 1.0) + 0.5 * rhoL * (ug * ug + vg * vg);
                for (int j = 0; j < 4; ++j) {
                    dQ[j] = Qg[j] - Q1[j];
                }
                dx = -2.0 * (x1 - xf) * nx;
                dy = -2.0 * (y1 - yf) * ny;
            }
            // update gradients
            for(int j=0;j<4;j++){
                gradX[j] += dQ[j] * (dx * dMesh.d_Ixy[c] - dy * dMesh.d_Ixx[c]);
                gradY[j] += dQ[j] * (dy * dMesh.d_Iyy[c] - dx * dMesh.d_Ixy[c]);
            }
        }

        // Apply geometric factors
        for(int j=0;j<4;j++){
            d_rs.dQx[4*c+j] = gradX[j];
            d_rs.dQy[4*c+j] = gradY[j];
        }
    }
};

// Compute left/right states at faces
class ComputeFaceStates {
public:
    DeviceMeshData dMesh;
    const double* d_Q;
    const double* d_Q_in;
    const DeviceFlow* d_flow;
    DeviceReconScraps d_rs;
    DeviceReconVars d_rv;
    const int* n_faces;

    deviceFunction void operator()(const unsigned int i) const {
        if (i >= *n_faces) {
            // std::cout << "this bih out of bound" << std::endl;
            return; // out of bounds
        }
        int c1 = dMesh.d_f2c[2*i]-1;
        int c2 = dMesh.d_f2c[2*i+1]-1;

        double x1 = dMesh.d_r_c[2*c1], y1 = dMesh.d_r_c[2*c1+1];
        double xf = dMesh.d_r_f[2*i], yf = dMesh.d_r_f[2*i+1];
        double dfx1 = xf - x1, dfy1 = yf - y1;
        double gamma = *(d_flow->d_gamma);

        // left state
        for(int j=0;j<4;j++)
            d_rv.Q_L[4*i+j] = d_Q[4*c1+j] + d_rs.dQx[4*c1+j]*dfx1 + d_rs.dQy[4*c1+j]*dfy1;

        // right state
        if(c2>=0){
            double x2 = dMesh.d_r_c[2*c2], y2 = dMesh.d_r_c[2*c2+1];
            double dfx2 = xf - x2, dfy2 = yf - y2;
            for(int j=0;j<4;j++)
                d_rv.Q_R[4*i+j] = d_Q[4*c2+j] + d_rs.dQx[4*c2+j]*dfx2 + d_rs.dQy[4*c2+j]*dfy2;
        } else if(c2==-1){ // reflective wall
            double rhoL = d_rv.Q_L[4*i];
            double uL = d_rv.Q_L[4*i+1]/rhoL;
            double vL = d_rv.Q_L[4*i+2]/rhoL;
            double pL = (d_rv.Q_L[4*i+3] - 0.5*rhoL*(uL*uL+vL*vL))*(gamma-1.0);
            double nx=dMesh.d_n_f[2*i], ny=dMesh.d_n_f[2*i+1];
            double vn = uL*nx + vL*ny;
            double uR = uL - 2.0*vn*nx;
            double vR = vL - 2.0*vn*ny;
            d_rv.Q_R[4*i]   = rhoL;
            d_rv.Q_R[4*i+1] = rhoL*uR;
            d_rv.Q_R[4*i+2] = rhoL*vR;
            d_rv.Q_R[4*i+3] = pL/(gamma-1.0)+0.5*rhoL*(uR*uR+vR*vR);
        } else { // inflow/outflow
            for(int j=0;j<4;j++)
                d_rv.Q_R[4*i+j] = d_Q_in[j];
        }
    }
};

#endif // G_RECONLS_H