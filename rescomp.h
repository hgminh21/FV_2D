#ifndef RESCOMP_H
#define RESCOMP_H

#include <Eigen>

using namespace Eigen;

// Function to compute the residual for each cell based on the face fluxes.
// f2c: Face-to-cell connectivity (n_faces x 2, one-indexed)
// A: Vector of face areas (or lengths in 2D) (n_faces)
// V: Vector of cell volumes (or areas) (n_cells)
// F: Matrix of fluxes at faces (n_faces x 4)
// Returns a matrix Res (n_cells x 4) containing the residual for each cell.
void compute_residual(const MatrixXi &f2c,
                    const VectorXd &A,
                    const VectorXd &V,
                    const MatrixXd &F,
                    const VectorXd &s_max_all,
                    double CFL,
                    MatrixXd &Res,
                    VectorXd &dt_local)
{
    int n_faces = F.rows();
    int n_cells = V.size();
    MatrixXd Res_local = MatrixXd::Zero(n_cells, 4);
    VectorXd dt_sum = VectorXd::Zero(n_cells);
    
    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c(i, 0) - 1;  // convert to zero-index
        int c2 = f2c(i, 1) - 1;  // convert to zero-index
        
        Res_local.row(c1) -= (F.row(i) * A(i)) / V(c1);
        dt_sum(c1) += s_max_all(i) * A(i) / V(c1);
        if (c2 >= 0) {
            Res_local.row(c2) += (F.row(i) * A(i)) / V(c2);
            dt_sum(c2) += s_max_all(i) * A(i) / V(c2);
        }
    }
    Res = Res_local;
    dt_local = CFL * dt_sum.cwiseInverse();
}

#endif  // RESCOMP_H
