#ifndef SOLVERILU_H
#define SOLVERILU_H

#include <petscksp.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

// Solves A * x = b using PETSc with ILU preconditioning and GMRES
inline void solve_one_system(const Eigen::MatrixXd& A_eigen,
                            const Eigen::VectorXd& b_eigen,
                            Eigen::VectorXd& x_eigen,
                            int col_index)
{
    int n = A_eigen.rows();

    Mat A;
    Vec b, x;
    PetscErrorCode ierr;

    // PETSc matrix
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetType(A, MATSEQAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetUp(A); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
    if (A_eigen(i, j) != 0.0)
    ierr = MatSetValue(A, i, j, A_eigen(i, j), INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // RHS
    ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecSetSizes(b, PETSC_DECIDE, n); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecSetFromOptions(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    for (int i = 0; i < n; ++i)
    ierr = VecSetValue(b, i, b_eigen[i], INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = VecAssemblyBegin(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecAssemblyEnd(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = VecDuplicate(b, &x); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Solver
    KSP ksp; 

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = KSPSetType(ksp, KSPCG); CHKERRABORT(PETSC_COMM_WORLD, ierr);  // Use Conjugate Gradient for SPD
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRABORT(PETSC_COMM_WORLD, ierr); // Use GMRES for non-SPD
    // int max_iters = 100000;  // Set max iterations
    // double tol = 1e-6;     // Set tolerance
    // ierr = KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iters); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    
    // PC pc;
    // ierr = KSPGetPC(ksp, &pc); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // // ierr = PCSetType(pc, PCCHOLESKY); CHKERRABORT(PETSC_COMM_WORLD, ierr); // Cholesky like ldlt
    // // ierr = PCSetType(pc, PCILU); CHKERRABORT(PETSC_COMM_WORLD, ierr); // ILU preconditioner
    // ierr = PCSetType(pc, PCJACOBI); CHKERRABORT(PETSC_COMM_WORLD, ierr);  // Jacobi preconditioner

    ierr = KSPSetFromOptions(ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Get result
    const PetscScalar* x_array;
    ierr = VecGetArrayRead(x, &x_array); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    x_eigen = Eigen::VectorXd(n);  // Allocate memory
    for (int i = 0; i < n; ++i)
        x_eigen[i] = x_array[i];
    ierr = VecRestoreArrayRead(x, &x_array); CHKERRABORT(PETSC_COMM_WORLD, ierr);    

    // Check
    Eigen::VectorXd residual = A_eigen * x_eigen - b_eigen;
    std::cout << "Residual norm [col " << col_index << "]: " << residual.norm() << std::endl;
    int iters;
    ierr = KSPGetIterationNumber(ksp, &iters); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    std::cout << "Number of iterations: " << iters << std::endl;

    // Cleanup
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
    KSPDestroy(&ksp);
}


inline void solver_ILU(const Eigen::MatrixXd& A_im1,
                       const Eigen::MatrixXd& A_im2,
                       const Eigen::MatrixXd& A_im3,
                       const Eigen::MatrixXd& A_im4,
                       const Eigen::MatrixXd& Res,
                       Eigen::MatrixXd& dQ)
{
    int n = Res.rows();
    dQ.resize(n, 4);
    Eigen::VectorXd x;

    solve_one_system(A_im1, Res.col(0), x, 0); dQ.col(0) = x;
    solve_one_system(A_im2, Res.col(1), x, 1); dQ.col(1) = x;
    solve_one_system(A_im3, Res.col(2), x, 2); dQ.col(2) = x;
    solve_one_system(A_im4, Res.col(3), x, 3); dQ.col(3) = x;
}

#endif // SOLVERILU_H
