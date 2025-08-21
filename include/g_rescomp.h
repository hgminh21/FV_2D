#ifndef G_RESCOMP_H
#define G_RESCOMP_H

#include "io/initialize.h"
#include "explicit/g_varini.h"
#include "explicit/meshcopy.h"

class ComputeResidualCellBased {
public:
    DeviceMeshData dMesh;
    const DeviceFluxVars* d_fv;
    DeviceResVars d_resv;
    const int* d_c2f_flat;
    const int* d_c2f_offset;
    double CFL;
    int use_cfl;

    deviceFunction void operator()(const unsigned int c) const {
        double res[4] = {0.0,0.0,0.0,0.0};
        double dt_loc = 0.0;

        int start = d_c2f_offset[c];
        int end   = d_c2f_offset[c+1];

        for (int idx = start; idx < end; ++idx) {
            int i = d_c2f_flat[idx]; // face index
            int c1 = dMesh.d_f2c[2*i]-1;
            int c2 = dMesh.d_f2c[2*i+1]-1;
            double Ai = dMesh.d_A[i];

            bool isLeft = (c1 == c);
            for (int j=0;j<4;j++)
                res[j] += (isLeft ? -1.0 : 1.0) * d_fv->F[4*i + j] * Ai / dMesh.d_V[c];

            if (use_cfl)
                dt_loc += d_fv->s_max_all[i] * Ai / dMesh.d_V[c];
        }

        for (int j=0;j<4;j++) d_resv.Res[4*c + j] = res[j];
        if (use_cfl) d_resv.dt_local[c] = CFL / dt_loc;
    }
};

#endif // G_RESCOMP_H