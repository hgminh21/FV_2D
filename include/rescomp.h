#ifndef RESCOMP_H
#define RESCOMP_H

#include "io/meshread.h"
#include "io/initialize.h"
#include "explicit/varini.h"

void compute_residual(const MeshData &mesh,
                    const Solver &solver,
                    const Time &time,
                    fluxVars &fv,
                    resVars &resv)
{
    std::fill(resv.Res.begin(), resv.Res.end(), 0.0);
    std::fill(resv.dt_local.begin(), resv.dt_local.end(), 0.0);

    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i + 1] - 1;

        for (int j = 0; j < 4; ++j) {
            resv.Res[4*c1 + j] -= (fv.F[4*i + j] * mesh.A[i]) / mesh.V[c1];
        }
        if (time.use_cfl == 1) {
            resv.dt_local[c1] += fv.s_max_all[i] * mesh.A[i] / mesh.V[c1];
        }

        if (c2 >= 0) {
            for (int j = 0; j < 4; ++j) {
                resv.Res[4*c2 + j] += (fv.F[4*i + j] * mesh.A[i]) / mesh.V[c2];
            }
            if (time.use_cfl == 1) {
                resv.dt_local[c2] += fv.s_max_all[i] * mesh.A[i] / mesh.V[c2];
            }
        }
    }

    if (time.use_cfl == 1) {
        for (int i = 0; i < mesh.n_cells; ++i) {
            resv.dt_local[i] = time.CFL / resv.dt_local[i];
        }
    }
}

#endif  // RESCOMP_H
