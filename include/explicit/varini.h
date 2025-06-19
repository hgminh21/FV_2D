#ifndef VARINI_H
#define VARINI_H

#include <vector>
#include "io/meshread.h"

struct reconVars {
    std::vector<double> Q_L, Q_R;
};

struct reconScraps{
    std::vector<double> Qx1_temp, Qx2_temp, Qy1_temp, Qy2_temp;
    std::vector<double> dQx, dQy, Q_max, Q_min, phi;
};

struct fluxVars {
    std::vector<double> F;
    std::vector<double> s_max_all;
};

struct fluxScraps {
    std::vector<double> Q_f, dQ_fx, dQ_fy;
    std::vector<double> F_viscous;
};

struct resVars {
    std::vector<double> Res;
    std::vector<double> dt_local;
};

struct ioVars {
    std::vector<double> Q_out;
    std::vector<double> Q1;
    std::vector<double> dVdn;
    std::vector<double> CP;
    std::vector<double> TauW;
    std::vector<double> Cf;
};

reconVars init_reconVars(const MeshData& mesh) {
    auto zm = [](int r, int c) { return std::vector<double>(r*c, 0.0); };
    auto zv = [](int s) { return std::vector<double>(s, 0.0); };

    reconVars v;
    v.Q_L        = zm(mesh.n_faces, 4);
    v.Q_R        = zm(mesh.n_faces, 4);
    return v;
}

reconScraps init_reconScraps(const MeshData& mesh) {
    auto zm = [](int r, int c) { return std::vector<double>(r*c, 0.0); };
    auto zv = [](int s) { return std::vector<double>(s, 0.0); };

    reconScraps v;
    v.dQx        = zm(mesh.n_cells, 4);
    v.dQy        = zm(mesh.n_cells, 4);
    v.Q_max      = zm(mesh.n_cells, 4);
    v.Q_min      = zm(mesh.n_cells, 4);
    v.phi        = zm(mesh.n_cells, 4);
    v.Qx1_temp   = zm(mesh.n_cells, 4);
    v.Qx2_temp   = zm(mesh.n_cells, 4);
    v.Qy1_temp   = zm(mesh.n_cells, 4);
    v.Qy2_temp   = zm(mesh.n_cells, 4);
    return v;
}

fluxVars init_fluxVars(const MeshData& mesh) {
    auto zm = [](int r, int c) { return std::vector<double>(r*c, 0.0); };
    auto zv = [](int s) { return std::vector<double>(s, 0.0); };

    fluxVars v;
    v.s_max_all  = zv(mesh.n_faces);
    v.F          = zm(mesh.n_faces, 4);
    return v;
}

fluxScraps init_fluxScraps(const MeshData& mesh) {
    auto zm = [](int r, int c) { return std::vector<double>(r*c, 0.0); };
    auto zv = [](int s) { return std::vector<double>(s, 0.0); };

    fluxScraps v;
    v.F_viscous  = zm(mesh.n_faces, 4);
    v.Q_f        = zm(mesh.n_faces, 4);
    v.dQ_fx      = zm(mesh.n_faces, 4);
    v.dQ_fy      = zm(mesh.n_faces, 4);
    return v;
}

resVars init_resVars(const MeshData& mesh) {
    auto zm = [](int r, int c) { return std::vector<double>(r*c, 0.0); };
    auto zv = [](int s) { return std::vector<double>(s, 0.0); };

    resVars v;
    v.dt_local   = zv(mesh.n_cells);
    v.Res        = zm(mesh.n_cells, 4);
    return v;
}

ioVars init_ioVars(const MeshData& mesh) {
    auto zm = [](int r, int c) { return std::vector<double>(r*c, 0.0); };
    auto zv = [](int s) { return std::vector<double>(s, 0.0); }; 

    ioVars v;
    v.Q_out      = zm(mesh.n_nodes, 4);
    v.Q1         = zm(mesh.n_cells, 4);
    v.dVdn       = zv(mesh.n_fwalls);
    v.CP         = zv(mesh.n_fwalls);
    v.TauW       = zv(mesh.n_fwalls);
    v.Cf         = zv(mesh.n_fwalls);
    return v;
}

#endif // VARINI_H