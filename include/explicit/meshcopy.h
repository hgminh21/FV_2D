#ifndef MESHCOPY_H
#define MESHCOPY_H
#include "io/meshread.h"
#include "io/initialize.h"

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

struct DeviceMeshData {
    // Device pointers
    double* d_r_node    = nullptr;
    double* d_r_f       = nullptr;
    double* d_n_f       = nullptr;
    double* d_A         = nullptr;
    double* d_r_c       = nullptr;
    double* d_V         = nullptr;
    double* d_Ixx       = nullptr;
    double* d_Iyy       = nullptr;
    double* d_Ixy       = nullptr;
    double* d_delta     = nullptr;

    int*    d_f2n       = nullptr;
    int*    d_f2c       = nullptr;
    int*    d_c2n_tri   = nullptr;
    int*    d_neighbor_flat = nullptr;
    int*    d_neighbor_count = nullptr;
    int*    d_neighbor_offset = nullptr;

    int*    d_c2f_flat = nullptr;    // flattened list of faces per cell
    int*    d_c2f_offset = nullptr;  // prefix sum for starting index

    // Allocate all device memory
    void allocate(device& d, const MeshData& mesh) {
        d.Malloc((void**)&d_f2n, mesh.n_faces * 2 * sizeof(int));
        d.Malloc((void**)&d_f2c, mesh.n_faces * 2 * sizeof(int));
        d.Malloc((void**)&d_r_node, mesh.n_nodes * 2 * sizeof(double));
        d.Malloc((void**)&d_r_f, mesh.n_faces * 2 * sizeof(double));
        d.Malloc((void**)&d_n_f, mesh.n_faces * 2 * sizeof(double));
        d.Malloc((void**)&d_A, mesh.n_faces * sizeof(double));
        d.Malloc((void**)&d_r_c, mesh.n_cells * 2 * sizeof(double));
        d.Malloc((void**)&d_V, mesh.n_cells * sizeof(double));
        d.Malloc((void**)&d_Ixx, mesh.n_cells * sizeof(double));
        d.Malloc((void**)&d_Iyy, mesh.n_cells * sizeof(double));
        d.Malloc((void**)&d_Ixy, mesh.n_cells * sizeof(double));
        d.Malloc((void**)&d_delta, mesh.n_cells * sizeof(double));
        d.Malloc((void**)&d_c2n_tri, mesh.c2n_tri.size() * sizeof(int));
        // i have to do this since im not sure the size of the neighbor arrays
        d.Malloc((void**)&d_neighbor_flat, mesh.neighbor_flat.size() * sizeof(int));
        d.Malloc((void**)&d_neighbor_count, mesh.neighbor_count.size() * sizeof(int));
        d.Malloc((void**)&d_neighbor_offset, mesh.neighbor_offset.size() * sizeof(int));

        d.Malloc((void**)&d_c2f_flat, mesh.c2f_flat.size() * sizeof(int));
        d.Malloc((void**)&d_c2f_offset, mesh.c2f_offset.size() * sizeof(int));
    }

    // Copy all host data to device
    void copyToDevice(device& d, const MeshData& mesh) {
        d.MemcpyHostToDevice(d_r_node, mesh.r_node.data(), mesh.n_nodes * 2 * sizeof(double));
        d.MemcpyHostToDevice(d_f2n, mesh.f2n.data(), mesh.n_faces * 2 * sizeof(int));
        d.MemcpyHostToDevice(d_f2c, mesh.f2c.data(), mesh.n_faces * 2 * sizeof(int));
        d.MemcpyHostToDevice(d_r_f, mesh.r_f.data(), mesh.n_faces * 2 * sizeof(double));
        d.MemcpyHostToDevice(d_n_f, mesh.n_f.data(), mesh.n_faces * 2 * sizeof(double));
        d.MemcpyHostToDevice(d_A, mesh.A.data(), mesh.n_faces * sizeof(double));
        d.MemcpyHostToDevice(d_r_c, mesh.r_c.data(), mesh.n_cells * 2 * sizeof(double));
        d.MemcpyHostToDevice(d_V, mesh.V.data(), mesh.n_cells * sizeof(double));
        d.MemcpyHostToDevice(d_Ixx, mesh.Ixx.data(), mesh.n_cells * sizeof(double));
        d.MemcpyHostToDevice(d_Iyy, mesh.Iyy.data(), mesh.n_cells * sizeof(double));
        d.MemcpyHostToDevice(d_Ixy, mesh.Ixy.data(), mesh.n_cells * sizeof(double));
        d.MemcpyHostToDevice(d_delta, mesh.delta.data(), mesh.n_cells * sizeof(double));
        d.MemcpyHostToDevice(d_c2n_tri, mesh.c2n_tri.data(), mesh.c2n_tri.size() * sizeof(int));
        d.MemcpyHostToDevice(d_neighbor_flat, mesh.neighbor_flat.data(), mesh.neighbor_flat.size() * sizeof(int));
        d.MemcpyHostToDevice(d_neighbor_count, mesh.neighbor_count.data(), mesh.neighbor_count.size() * sizeof(int));
        d.MemcpyHostToDevice(d_neighbor_offset, mesh.neighbor_offset.data(), mesh.neighbor_offset.size() * sizeof(int));
        d.MemcpyHostToDevice(d_c2f_flat, mesh.c2f_flat.data(), mesh.c2f_flat.size() * sizeof(int));
        d.MemcpyHostToDevice(d_c2f_offset, mesh.c2f_offset.data(), mesh.c2f_offset.size() * sizeof(int));
    }

    // Free all device memory
    void free(device& d) {
        d.Free(d_f2n);
        d.Free(d_f2c);
        d.Free(d_r_node);
        d.Free(d_r_f);
        d.Free(d_n_f);
        d.Free(d_A);
        d.Free(d_r_c);
        d.Free(d_V);
        d.Free(d_Ixx);
        d.Free(d_Iyy);
        d.Free(d_Ixy);
        d.Free(d_delta);
    }
};

struct DeviceFlow {
    double* d_rho = nullptr;
    double* d_p = nullptr;
    double* d_u = nullptr;
    double* d_v = nullptr;
    double* d_gamma = nullptr;
    double* d_Pr = nullptr;
    double* d_R = nullptr;
    double* d_mu = nullptr;
    double* d_Cp = nullptr;
    double* d_T = nullptr;
    double* d_k = nullptr;
    int* d_type = nullptr;

    void allocate(device& d) {
        d.Malloc((void**)&d_rho, sizeof(double));
        d.Malloc((void**)&d_p, sizeof(double));
        d.Malloc((void**)&d_u, sizeof(double));
        d.Malloc((void**)&d_v, sizeof(double));
        d.Malloc((void**)&d_gamma, sizeof(double));
        d.Malloc((void**)&d_Pr, sizeof(double));
        d.Malloc((void**)&d_R, sizeof(double));
        d.Malloc((void**)&d_mu, sizeof(double));
        d.Malloc((void**)&d_Cp, sizeof(double));
        d.Malloc((void**)&d_T, sizeof(double));
        d.Malloc((void**)&d_k, sizeof(double));
        d.Malloc((void**)&d_type, sizeof(int));
    }

    void copyToDevice(device& d, const Flow& flow) {
        d.MemcpyHostToDevice(d_rho, &flow.rho, sizeof(double));
        d.MemcpyHostToDevice(d_p, &flow.p, sizeof(double));
        d.MemcpyHostToDevice(d_u, &flow.u, sizeof(double));
        d.MemcpyHostToDevice(d_v, &flow.v, sizeof(double));
        d.MemcpyHostToDevice(d_gamma, &flow.gamma, sizeof(double));
        d.MemcpyHostToDevice(d_Pr, &flow.Pr, sizeof(double));
        d.MemcpyHostToDevice(d_R, &flow.R, sizeof(double));
        d.MemcpyHostToDevice(d_mu, &flow.mu, sizeof(double));
        d.MemcpyHostToDevice(d_Cp, &flow.Cp, sizeof(double));
        d.MemcpyHostToDevice(d_T, &flow.T, sizeof(double));
        d.MemcpyHostToDevice(d_k, &flow.k, sizeof(double));
        d.MemcpyHostToDevice(d_type, &flow.type, sizeof(int));
    }

    void free(device& d) {
        d.Free(d_rho);
        d.Free(d_p);
        d.Free(d_u);
        d.Free(d_v);
        d.Free(d_gamma);
        d.Free(d_Pr);
        d.Free(d_R);
        d.Free(d_mu);
        d.Free(d_Cp);
        d.Free(d_T);
        d.Free(d_k);
        d.Free(d_type);
    }
};

struct DeviceIniVars {
    double* d_Q_init = nullptr;
    double* d_Q = nullptr;

    void allocate(device& d, const MeshData& mesh) {
        d.Malloc((void**)&d_Q_init, mesh.n_cells * 4 * sizeof(double));
        d.Malloc((void**)&d_Q, mesh.n_cells * 4 * sizeof(double));
    }

    void copyToDevice(device& d, const std::vector<double>& Q_init, const std::vector<double>& Q) {
        d.MemcpyHostToDevice(d_Q_init, Q_init.data(), Q_init.size() * sizeof(double));
        d.MemcpyHostToDevice(d_Q, Q.data(), Q.size() * sizeof(double));
    }

    void free(device& d) {
        d.Free(d_Q_init);
        d.Free(d_Q);
    }
};
#endif // MESHCOPY_H