#ifndef SOLVERLDLT_H
#define SOLVERLDLT_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <omp.h>

Eigen::MatrixXd solver_LDLT(const Eigen::MatrixXd& A_im1,
                         const Eigen::MatrixXd& A_im2,
                         const Eigen::MatrixXd& A_im3,
                         const Eigen::MatrixXd& A_im4,
                         const Eigen::MatrixXd& Res,
                         Eigen::MatrixXd& dQt)
{
    using SpMat = Eigen::SparseMatrix<double>;
    SpMat A1 = A_im1.sparseView();
    SpMat A2 = A_im2.sparseView();
    SpMat A3 = A_im3.sparseView();
    SpMat A4 = A_im4.sparseView();

    Eigen::SimplicialLDLT<SpMat> solver1, solver2, solver3, solver4;

    solver1.compute(A1);
    solver2.compute(A2);
    solver3.compute(A3);
    solver4.compute(A4);

    // Optional: check success
    if (solver1.info() != Eigen::Success) std::cerr << "Solver1 failed to factor A1\n";
    if (solver2.info() != Eigen::Success) std::cerr << "Solver2 failed to factor A2\n";
    if (solver3.info() != Eigen::Success) std::cerr << "Solver3 failed to factor A3\n";
    if (solver4.info() != Eigen::Success) std::cerr << "Solver4 failed to factor A4\n";

    dQt.col(0) = solver1.solve(Res.col(0));
    dQt.col(1) = solver2.solve(Res.col(1));
    dQt.col(2) = solver3.solve(Res.col(2));
    dQt.col(3) = solver4.solve(Res.col(3));

    return dQt;
}

#endif // SOLVERLDLT_H