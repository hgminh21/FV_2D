#ifndef VENKAT_H
#define VENKAT_H

#include <Eigen/Dense>

using namespace Eigen;

double venkat_lim(const RowVectorXd Q_max,
                  const RowVectorXd Q_min,
                  const RowVectorXd Q,
                  const RowVectorXd Q_f,
                  double phi)
{
    for (int j = 0; j < 4; ++j) {
        double delm = Q_f(j) - Q(j);
        
        double delp;
        if (delm > 0) delp = Q_max(j) - Q(j);
        else delp = Q_min(j) - Q(j);
        
        double r = delm/delp;

        double phif = (r * r + 2.0 * r) / (r * r + r + 2);

        if (phif < phi) phi = phif;
    }

    return phi;
}

#endif  // VENKAT_H