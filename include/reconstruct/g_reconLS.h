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

    deviceFunction void operator()(const unsigned int i) const {
        int n_neighbors = dMesh.d_neighbor_count[i];
        int offset = dMesh.d_neighbor_offset[i];

        double x_c = dMesh.d_r_c[2*i];
        double y_c = dMesh.d_r_c[2*i+1];

        double gradX[4] = {0.0, 0.0, 0.0, 0.0};
        double gradY[4] = {0.0, 0.0, 0.0, 0.0};

        // Loop over neighbors
        for(int n=0; n<n_neighbors; ++n){
            int nb = dMesh.d_neighbor_flat[offset + n];
            double x_nb, y_nb;
            double Q_nb[4];
            if(nb >= 0){  // interior neighbor
                x_nb = dMesh.d_r_c[2*nb];
                y_nb = dMesh.d_r_c[2*nb+1];
                for(int j=0;j<4;j++) Q_nb[j] = d_Q[4*nb + j];
            } else {      // boundary condition
                int bc_type = nb;
                x_nb = dMesh.d_r_f[-(bc_type+1)*2];  // approximate BC point (face)
                y_nb = dMesh.d_r_f[-(bc_type+1)*2+1];
                for(int j=0;j<4;j++) Q_nb[j] = d_Q_in[j];
            }

            double dx = x_nb - x_c;
            double dy = y_nb - y_c;
            for(int j=0;j<4;j++){
                gradX[j] += (Q_nb[j] - d_Q[4*i + j]) * dx;
                gradY[j] += (Q_nb[j] - d_Q[4*i + j]) * dy;
            }
        }

        // Apply geometric factors
        for(int j=0;j<4;j++){
            d_rs.dQx[4*i+j] = gradX[j]*dMesh.d_Iyy[i] + gradY[j]*dMesh.d_Ixy[i];
            d_rs.dQy[4*i+j] = gradY[j]*dMesh.d_Ixx[i] + gradX[j]*dMesh.d_Ixy[i];
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

    deviceFunction void operator()(const unsigned int i) const {
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