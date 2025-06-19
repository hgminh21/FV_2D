#ifndef VENKAT_H
#define VENKAT_H

#include <io/initialize.h>

std::vector<double> venkat_lim(const std::vector<double> Q_max,
                        const std::vector<double> Q_min,
                        const std::vector<double> Q,
                        const std::vector<double> Q_f,
                        std::vector<double> phi,
                        const Reconstruct &recon)
{
    // skip face if cell does not need to be limited
    if (phi[0] == 1.0 || phi[3] == 1.0) {
        for (int j = 0; j < 4; ++j) {
            phi[j] = 1.0;
        }
        return phi;
    }

    for (int j = 0; j < 4; ++j) {
        double delm = Q_f[j] - Q[j];
        
        double delp;
        if (delm > 0) delp = Q_max[j] - Q[j];
        else delp = Q_min[j] - Q[j];
        
        double r = delm/delp;

        double phif = (r * r + 2.0 * r + recon.lim_tol * recon.lim_tol) 
                        / (r * r + r + 2 + recon.lim_tol * recon.lim_tol);

        phi[j] = std::min(phi[j], 1.0);
        phi[j] = std::min(phi[j], phif);
    }

    // Triggering condition for phi relaxation
    if (phi[0] == 1.0 || phi[3] == 1.0) {
        for (int j = 0; j < 4; ++j) {
            phi[j] = 1.0;
        }
    }    
    return phi;
}

#endif  // VENKAT_H