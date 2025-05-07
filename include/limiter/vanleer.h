#ifndef VANLEER_H
#define VANLEER_H

#include <Eigen/Dense>
#include <io/initialize.h>

using namespace Eigen;

RowVectorXd vanleer_lim(const RowVectorXd Q_max,
                        const RowVectorXd Q_min,
                        const RowVectorXd Q,
                        const RowVectorXd Q_f,
                        RowVectorXd phi,
                        const Reconstruct &recon)
{
    // skip face if cell does not need to be limited
    if (phi(0) == 1.0 || phi(3) == 1.0) {
        phi = RowVectorXd::Ones(4);
        return phi;
    }

    for (int j = 0; j < 4; ++j) {
        double delm = Q_f(j) - Q(j);
        
        double delp;
        if (delm > 0) delp = Q_max(j) - Q(j);
        else delp = Q_min(j) - Q(j);
        
        double r = delm/delp;

        double phif = (r + abs(r)) / (1.0 + abs(r) + recon.lim_tol);

        phi(j) = std::min(phi(j), 1.0);
        phi(j) = std::min(phi(j), phif);
    }

    // Triggering condition for phi relaxation
    if (phi(0) == 1.0 || phi(3) == 1.0) {phi = RowVectorXd::Ones(4);}
    return phi;
}

#endif  // VANLEER_H