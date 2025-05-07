#ifndef SQUEEZE_H
#define SQUEEZE_H

#include <Eigen/Dense>
#include <io/initialize.h>

using namespace Eigen;

RowVectorXd squeeze_lim(const RowVectorXd Q_max,
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
        double phif;
        double Qmax, Qmin;
        if (Q_max(j) >= 0) {Qmax = ((1.0 + recon.lim_thres) * Q_max(j));
        } else {Qmax = ((1.0 - recon.lim_thres) * Q_max(j));}
        if (Q_min(j) >= 0) {Qmin = ((1.0 - recon.lim_thres) * Q_min(j));
        } else {Qmin = ((1.0 + recon.lim_thres) * Q_min(j));}

        if (Q_f(j) > Qmax) {
            phif = (Q_max(j) - Q(j)) / (Q_f(j) - Q(j) - recon.lim_tol * abs(Q(j)));
        }
        else if (Q_f(j) < Qmin) {
            phif = (Q_min(j) - Q(j)) / (Q_f(j) - Q(j) + recon.lim_tol * abs(Q(j)));
        }
        else {
            phif = 1.0;
        }
        phi(j) = std::min(phi(j), 1.0);
        phi(j) = std::min(phi(j), phif);
    }

    // Triggering condition for phi relaxation
    if (phi(0) == 1.0 || phi(3) == 1.0) {phi = RowVectorXd::Ones(4);}
    return phi;
}

#endif  // SQUEEZE_H