#ifndef SOLVERBICG_H
#define SOLVERBICG_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>


void solver_BiCG(const Eigen::MatrixXd &A_im1, 
                const Eigen::MatrixXd &A_im2, 
                const Eigen::MatrixXd &A_im3, 
                const Eigen::MatrixXd &A_im4,
                const Eigen::MatrixXd &Res,
                Eigen::MatrixXd &dQt) 
{
        // // Stage 2: Solve the implicit system using the residuals.
        // -- convert to sparse --
        using SpMat = Eigen::SparseMatrix<double>;
        SpMat A1 = A_im1.sparseView();
        SpMat A2 = A_im2.sparseView();
        SpMat A3 = A_im3.sparseView();
        SpMat A4 = A_im4.sparseView();

        // -- set up BiCGSTAB solver type with diagonal preconditioner --
        using BiCG = Eigen::BiCGSTAB<SpMat, Eigen::DiagonalPreconditioner<double>>;

        BiCG solver1, solver2, solver3, solver4;

        // -- compute factorization once per matrix --
        solver1.compute(A1);
        solver2.compute(A2);
        solver3.compute(A3);
        solver4.compute(A4);

        // optional: check for numerical issues
        if (solver1.info() != Eigen::Success ||
            solver2.info() != Eigen::Success ||
            solver3.info() != Eigen::Success ||
            solver4.info() != Eigen::Success)
        {
            std::cerr << "BiCGSTAB compute() failed on one of the matrices\n";
        }

        // -- solve for each RHS column --
        dQt.col(0) = solver1.solve(Res.col(0));
        dQt.col(1) = solver2.solve(Res.col(1));
        dQt.col(2) = solver3.solve(Res.col(2));
        dQt.col(3) = solver4.solve(Res.col(3));

        // optional: inspect convergence
        std::cout << "Iters: "
                << solver1.iterations() << ", "
                << solver2.iterations() << ", "
                << solver3.iterations() << ", "
                << solver4.iterations() << " | "
                << "Errors: "
                << solver1.error()      << ", "
                << solver2.error()      << ", "
                << solver3.error()      << ", "
                << solver4.error()      << "\n";

}

#endif // SOLVERBICG_H