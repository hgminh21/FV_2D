#ifndef SQUEEZE_H
#define SQUEEZE_H

#include <Eigen/Dense>

using namespace Eigen;

double squeeze_lim(const RowVectorXd Q_max,
                   const RowVectorXd Q_min,
                   const RowVectorXd Q,
                   const RowVectorXd Q_f,
                   double phi)
{
    for (int j = 0; j < 4; ++j) {
        double phif;
        if (Q_f(j) > Q_max(j)) {
            phif = (Q_max(j) - Q(j)) / (Q_f(j) - Q(j));
        }
        else if (Q_f(j) < Q_min(j)) {
            phif = (Q_min(j) - Q(j)) / (Q_f(j) - Q(j));
        }
        else {
            phif = 1.0;
        }
    if (phif < phi) phi = phif;
    }

    return phi;
}

#endif  // SQUEEZE_H