#ifndef G_DEVICEVARINI_H
#define G_DEVICEVARINI_H

#include "io/meshread.h"

#ifdef USECUDA
    #include "backend/cudaDevice.cuh"
    typedef CUDAdevice device;
#elif defined USESYCL
    #include "backend/syclDevice.hpp"
    typedef SYCLdevice device; 
#elif defined USEHIP
    #include "backend/hipDevice.hpp"
    typedef HIPdevice device;      
#else
     #include "backend/ompDevice.hpp"
     typedef OMPdevice device;
#endif

struct DeviceReconVars {
    double *Q_L = nullptr;
    double *Q_R = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        d.Malloc((void**)&Q_L, mesh.n_faces * 4 * sizeof(double));
        d.Malloc((void**)&Q_R, mesh.n_faces * 4 * sizeof(double));
    }

    void free(device& d) {
        d.Free(Q_L);
        d.Free(Q_R);
    }
};

struct DeviceReconScraps {
    // double *Qx1_temp = nullptr, *Qx2_temp = nullptr;
    // double *Qy1_temp = nullptr, *Qy2_temp = nullptr;
    double *dQx = nullptr, *dQy = nullptr;
    double *Q_max = nullptr, *Q_min = nullptr;
    double *phi = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        auto sizeC4 = mesh.n_cells * 4 * sizeof(double);
        d.Malloc((void**)&dQx, sizeC4);
        d.Malloc((void**)&dQy, sizeC4);
        d.Malloc((void**)&Q_max, sizeC4);
        d.Malloc((void**)&Q_min, sizeC4);
        d.Malloc((void**)&phi, sizeC4);
        // d.Malloc((void**)&Qx1_temp, sizeC4);
        // d.Malloc((void**)&Qx2_temp, sizeC4);
        // d.Malloc((void**)&Qy1_temp, sizeC4);
        // d.Malloc((void**)&Qy2_temp, sizeC4);
    }

    void free(device& d) {
        d.Free(dQx); d.Free(dQy);
        d.Free(Q_max); d.Free(Q_min); d.Free(phi);
        // d.Free(Qx1_temp); d.Free(Qx2_temp);
        // d.Free(Qy1_temp); d.Free(Qy2_temp);
    }
};

struct DeviceFluxVars {
    double *F = nullptr;
    double *s_max_all = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        d.Malloc((void**)&F, mesh.n_faces * 4 * sizeof(double));
        d.Malloc((void**)&s_max_all, mesh.n_faces * sizeof(double));
    }

    void free(device& d) {
        d.Free(F);
        d.Free(s_max_all);
    }
};

struct DeviceFluxScraps {
    double *Q_f = nullptr, *dQ_fx = nullptr, *dQ_fy = nullptr;
    double *F_viscous = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        auto sizeF4 = mesh.n_faces * 4 * sizeof(double);
        d.Malloc((void**)&Q_f, sizeF4);
        d.Malloc((void**)&dQ_fx, sizeF4);
        d.Malloc((void**)&dQ_fy, sizeF4);
        d.Malloc((void**)&F_viscous, sizeF4);
    }

    void free(device& d) {
        d.Free(Q_f); d.Free(dQ_fx); d.Free(dQ_fy); d.Free(F_viscous);
    }
};

struct DeviceResVars {
    double *Res = nullptr;
    double *dt_local = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        d.Malloc((void**)&Res, mesh.n_cells * 4 * sizeof(double));
        d.Malloc((void**)&dt_local, mesh.n_cells * sizeof(double));
    }

    void free(device& d) {
        d.Free(Res);
        d.Free(dt_local);
    }
};

struct DeviceIoVars {
    double *Q_out = nullptr;
    double *Q1 = nullptr;
    double *dVdn = nullptr;
    double *CP = nullptr;
    double *TauW = nullptr;
    double *Cf = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        d.Malloc((void**)&Q_out, mesh.n_nodes * 4 * sizeof(double));
        d.Malloc((void**)&Q1, mesh.n_cells * 4 * sizeof(double));
        d.Malloc((void**)&dVdn, mesh.n_fwalls * sizeof(double));
        d.Malloc((void**)&CP, mesh.n_fwalls * sizeof(double));
        d.Malloc((void**)&TauW, mesh.n_fwalls * sizeof(double));
        d.Malloc((void**)&Cf, mesh.n_fwalls * sizeof(double));
    }

    void free(device& d) {
        d.Free(Q_out); d.Free(Q1);
        d.Free(dVdn); d.Free(CP); d.Free(TauW); d.Free(Cf);
    }
};
#endif // DEVICEVARINI_H
