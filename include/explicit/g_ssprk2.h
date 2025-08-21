#ifndef G_SSPRK2_H
#define G_SSPRK2_H

#include <iostream>
#include <filesystem>

#include "io/meshread.h"
#include "io/initialize.h"
#include "reconstruct/g_reconLS.h"
#include "flux/g_fluxcomp.h"
#include "g_rescomp.h"
#include "g_varini.h"
#include "explicit/meshcopy.h"

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

// Cross-platform helper to get optimal threads per block / work-group size
unsigned int getOptimalThreadsPerBlock(device& d) {
#ifdef USECUDA
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    return props.warpSize * 4; // safe multiple of warp (e.g., 128)
#elif defined USESYCL
    auto dev = d.queue.get_device();
    return dev.get_info<sycl::info::device::max_work_group_size>() / 2; 
#elif defined USEOMP
    return omp_get_max_threads();
#else
    return 64; // fallback
#endif
}

class ComputeQStage {
public:
    const DeviceResVars* d_resv; // device residuals
    const double* d_Q;           // current Q on device
    double* d_Q1;                // output Q_stage
    double dt;                   // fixed time step
    int use_cfl;                 // CPU scalar, copied into device
    int local_dt;                // CPU scalar
    double dt_global;            // CPU scalar
    unsigned int n_cells;

    deviceFunction void operator()(const unsigned int c) const {
        double delta_t = dt; // default

        if (use_cfl == 1) {
            delta_t = (local_dt == 0) ? dt_global : d_resv->dt_local[c];
        }

        for (int j = 0; j < 4; ++j) {
            d_Q1[4*c + j] = d_Q[4*c + j] + delta_t * d_resv->Res[4*c + j];
        }
    }
};

// Add this helper functor next to ComputeQStage
class ComputeFinalUpdate {
public:
    const DeviceResVars* d_resv;   // residual at stage-2 (L(Q_stage))
    const double* d_Qn;            // Q^n (old state)
    const double* d_Q1;            // Q_stage
    double* d_Qout;                // write Q^{n+1} here (we'll use dIni.d_Q)
    double dt;                     // fixed dt
    int use_cfl;                   // 1: CFL, 0: fixed dt
    int local_dt;                  // 0: global, 1: local
    double dt_global;              // global time step if local_dt==0
    unsigned int n_cells;

    deviceFunction void operator()(const unsigned int c) const {
        double delta_t = dt;
        if (use_cfl == 1) {
            delta_t = (local_dt == 0) ? dt_global : d_resv->dt_local[c];
        }

        // Q^{n+1} = 0.5*Q^n + 0.5*(Q_stage + dt * L(Q_stage))
        for (int j = 0; j < 4; ++j) {
            const double qn  = d_Qn [4*c + j];
            const double qs  = d_Q1 [4*c + j];
            const double Lqs = d_resv->Res[4*c + j];
            d_Qout[4*c + j] = 0.5 * qn + 0.5 * (qs + delta_t * Lqs);
        }
    }
};


void ssprk2(const MeshData &mesh, const Solver &solver, const Flow &flow,
            const Reconstruct &recon, const Flux &flux, const Time &time,
            std::vector<double> &Q, const std::vector<double> &Q_in)
{
    device d;
    unsigned int Nthreads = getOptimalThreadsPerBlock(d);
    unsigned int block1 = (mesh.n_nodes + Nthreads - 1) / Nthreads;
    unsigned int block2 = (mesh.n_faces + Nthreads - 1) / Nthreads;
    unsigned int block3 = (mesh.n_cells + Nthreads - 1) / Nthreads;

    DeviceReconVars   drv;
    DeviceReconScraps drs;
    DeviceFluxVars    dfv;
    DeviceFluxScraps  dfs;
    DeviceResVars     dresv;
    DeviceIoVars      div;
    DeviceMeshData    dMesh;
    DeviceIniVars     dIni;
    DeviceFlow        dFlow;

    drv.allocate(d, mesh);
    drs.allocate(d, mesh);
    dfv.allocate(d, mesh);
    dfs.allocate(d, mesh);
    dresv.allocate(d, mesh);
    div.allocate(d, mesh);
    dMesh.allocate(d, mesh);
    dFlow.allocate(d);
    dMesh.copyToDevice(d, mesh);
    dFlow.copyToDevice(d, flow);
    dIni.allocate(d, mesh);
    dIni.copyToDevice(d, Q_in, Q);

    // (Optional) scratch host buffers for reductions / logging
    std::vector<double> host_dt_local(mesh.n_cells);
    std::vector<double> host_res(mesh.n_cells * 4);

    for (int step = 0; step <= solver.n_step; ++step) {

        // =========================
        // Stage 1: L(Q^n)
        // =========================

        // Gradients at cells using Q^n
        {
            ComputeCellGradients computeGrad;
            computeGrad.dMesh   = dMesh;
            computeGrad.d_Q     = dIni.d_Q;        // Q^n
            computeGrad.d_Q_in  = dIni.d_Q_init;   // inflow/bc state
            computeGrad.d_flow  = &dFlow;
            computeGrad.d_rs    = drs;
            d.LaunchKernel(block3, Nthreads, computeGrad);
        }

        // Face states from Q^n and gradients
        {
            ComputeFaceStates computeFace;
            computeFace.dMesh  = dMesh;
            computeFace.d_Q    = dIni.d_Q;         // Q^n
            computeFace.d_Q_in = dIni.d_Q_init;
            computeFace.d_flow = &dFlow;
            computeFace.d_rs   = drs;
            computeFace.d_rv   = drv;
            d.LaunchKernel(block2, Nthreads, computeFace);
        }

        // Fluxes at faces
        {
            ComputeFluxes computeFlux;
            computeFlux.dMesh  = dMesh;
            computeFlux.dflow  = &dFlow;
            computeFlux.d_rv   = &drv;
            computeFlux.d_fv   = dfv;
            computeFlux.method = flux.method;
            d.LaunchKernel(block2, Nthreads, computeFlux);
        }

        // Residual (cell-based)
        {
            ComputeResidualCellBased computeRes;
            computeRes.dMesh   = dMesh;
            computeRes.d_fv    = &dfv;
            computeRes.d_resv  = dresv;
            computeRes.CFL     = time.CFL;
            computeRes.use_cfl = time.use_cfl;
            d.LaunchKernel(block3, Nthreads, computeRes);
        }

        // Compute Q_stage
        {
            ComputeQStage qstage;
            qstage.d_resv    = &dresv;
            qstage.d_Q       = dIni.d_Q;     // Q^n
            qstage.d_Q1      = div.Q1;       // write Q_stage
            qstage.dt        = time.dt;
            qstage.use_cfl   = time.use_cfl;
            qstage.local_dt  = time.local_dt;

            // set dt_global if needed (host reduction on dt_local from Stage 1)
            double dt_glob = time.dt;
            if (time.use_cfl == 1 && time.local_dt == 0) {
                d.MemcpyDeviceToHost(host_dt_local.data(), dresv.dt_local, mesh.n_cells * sizeof(double));
                double sum = 0.0;
                for (int i = 0; i < mesh.n_cells; ++i) sum += host_dt_local[i];
                // dt_local kernel filled sum of (s_max A / V); invert * CFL
                // If your ComputeResidualCellBased already did: dt_local[i] = CFL / sum_i,
                // then the min reduction is appropriate:
                dt_glob = host_dt_local[0];
                for (int i = 1; i < mesh.n_cells; ++i) dt_glob = std::min(dt_glob, host_dt_local[i]);
            }
            qstage.dt_global = dt_glob;
            qstage.n_cells   = mesh.n_cells;

            d.LaunchKernel(block3, Nthreads, qstage);
        }

        // =========================
        // Stage 2: L(Q_stage)
        // =========================

        // Recompute gradients using Q_stage
        {
            ComputeCellGradients computeGrad2;
            computeGrad2.dMesh   = dMesh;
            computeGrad2.d_Q     = div.Q1;         // Q_stage
            computeGrad2.d_Q_in  = dIni.d_Q_init;
            computeGrad2.d_flow  = &dFlow;
            computeGrad2.d_rs    = drs;
            d.LaunchKernel(block3, Nthreads, computeGrad2);
        }

        // Face states from Q_stage
        {
            ComputeFaceStates computeFace2;
            computeFace2.dMesh  = dMesh;
            computeFace2.d_Q    = div.Q1;          // Q_stage
            computeFace2.d_Q_in = dIni.d_Q_init;
            computeFace2.d_flow = &dFlow;
            computeFace2.d_rs   = drs;
            computeFace2.d_rv   = drv;
            d.LaunchKernel(block2, Nthreads, computeFace2);
        }

        // Fluxes at faces (stage 2)
        {
            ComputeFluxes computeFlux2;
            computeFlux2.dMesh  = dMesh;
            computeFlux2.dflow  = &dFlow;
            computeFlux2.d_rv   = &drv;
            computeFlux2.d_fv   = dfv;
            computeFlux2.method = flux.method;
            d.LaunchKernel(block2, Nthreads, computeFlux2);
        }

        // Residual at Q_stage
        {
            ComputeResidualCellBased computeRes2;
            computeRes2.dMesh   = dMesh;
            computeRes2.d_fv    = &dfv;
            computeRes2.d_resv  = dresv;       // overwrite with L(Q_stage)
            computeRes2.CFL     = time.CFL;
            computeRes2.use_cfl = time.use_cfl;
            d.LaunchKernel(block3, Nthreads, computeRes2);
        }

        // =========================
        // Final Update: Q^{n+1}
        // =========================
        {
            ComputeFinalUpdate finalUp;
            finalUp.d_resv    = &dresv;      // L(Q_stage)
            finalUp.d_Qn      = dIni.d_Q;    // Q^n
            finalUp.d_Q1      = div.Q1;      // Q_stage
            finalUp.d_Qout    = dIni.d_Q;    // overwrite Q with Q^{n+1}
            finalUp.dt        = time.dt;
            finalUp.use_cfl   = time.use_cfl;
            finalUp.local_dt  = time.local_dt;

            // global dt from Stage 2 if needed
            double dt_glob2 = time.dt;
            if (time.use_cfl == 1 && time.local_dt == 0) {
                d.MemcpyDeviceToHost(host_dt_local.data(), dresv.dt_local, mesh.n_cells * sizeof(double));
                dt_glob2 = host_dt_local[0];
                for (int i = 1; i < mesh.n_cells; ++i) dt_glob2 = std::min(dt_glob2, host_dt_local[i]);
            }
            finalUp.dt_global = dt_glob2;
            finalUp.n_cells   = mesh.n_cells;

            d.LaunchKernel(block3, Nthreads, finalUp);
        }

       //  Print progress info every few steps.
        if (step % solver.m_step == 0) {
            // Copy final Q and residual back to host
            d.MemcpyDeviceToHost(Q.data(), dIni.d_Q, Q.size() * sizeof(double));
            d.MemcpyDeviceToHost(host_res.data(), dresv.Res, mesh.n_cells * 4 * sizeof(double));
            std::cout << "Completed step " << step << " of " << solver.n_step << std::endl;
            // // Compute L2 norms for each column of the residual
            double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;

            for (int i = 0; i < mesh.n_cells; ++i) {
                res1 += std::pow(host_res[4*i + 0], 2);
                res2 += std::pow(host_res[4*i + 1], 2);
                res3 += std::pow(host_res[4*i + 2], 2);
                res4 += std::pow(host_res[4*i + 3], 2);
            }

            res1 = std::sqrt(res1);
            res2 = std::sqrt(res2);
            res3 = std::sqrt(res3);
            res4 = std::sqrt(res4);
  
            std::cout << "Residuals: "
                    << "res1 = " << res1 << ", "
                    << "res2 = " << res2 << ", "
                    << "res3 = " << res3 << ", "
                    << "res4 = " << res4 << std::endl;
        }


    // frees
    drv.free(d); drs.free(d); dfv.free(d); dresv.free(d); div.free(d);
    dMesh.free(d); dFlow.free(d); dIni.free(d);
}


#endif // G_SSPRK2_H