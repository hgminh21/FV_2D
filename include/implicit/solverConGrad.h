#ifndef SOLVERCONGRAD_H
#define SOLVERCONGRAD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "io/meshread.h"


void solver_ConGrad(const MeshData &mesh,
                    const Eigen::MatrixXd &A_im1, 
                    const Eigen::MatrixXd &A_im2, 
                    const Eigen::MatrixXd &A_im3, 
                    const Eigen::MatrixXd &A_im4,
                    const Eigen::MatrixXd &Res,
                    Eigen::MatrixXd &dQt) 
{
    // Total size of block system
    int n = mesh.n_cells;
    int N = 4 * n;
    using SpMat = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;
    // 1. Create the big block-diagonal matrix
    std::vector<Triplet> triplets;
    
    // Offsets for blocks
    auto insertBlock = [&](const SpMat& A, int block_id) {
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SpMat::InnerIterator it(A, k); it; ++it) {
                int globalRow = it.row() + block_id * n;
                int globalCol = it.col() + block_id * n;
                triplets.emplace_back(globalRow, globalCol, it.value());
            }
        }
    };
    
    // Insert each matrix block
    insertBlock(A_im1.sparseView(), 0);
    insertBlock(A_im2.sparseView(), 1);
    insertBlock(A_im3.sparseView(), 2);
    insertBlock(A_im4.sparseView(), 3);
    
    // Assemble block matrix
    SpMat A_block(N, N);
    A_block.setFromTriplets(triplets.begin(), triplets.end());
    
    // 2. Stack the RHS vectors into one
    Eigen::VectorXd b_block(N);
    b_block.segment(0*n, n) = Res.col(0);
    b_block.segment(1*n, n) = Res.col(1);
    b_block.segment(2*n, n) = Res.col(2);
    b_block.segment(3*n, n) = Res.col(3);
    
    // 3. Solve with any iterative solver you prefer
    // Eigen::BiCGSTAB<SpMat, Eigen::DiagonalPreconditioner<double>> LNsolver;
    Eigen::BiCGSTAB<SpMat> LNsolver; // no preconditioner

    LNsolver.compute(A_block);
    if (LNsolver.info() != Eigen::Success) {
        std::cerr << "Block system decomposition failed\n";
    }
    
    Eigen::VectorXd x_block = LNsolver.solve(b_block);
    if (LNsolver.info() != Eigen::Success) {
        std::cerr << "Block system solve failed\n";
    }
    
    // 4. Unstack the solution into dQt
    for (int i = 0; i < 4; ++i) {
        dQt.col(i) = x_block.segment(i * n, n);
    }
    
    // 5. Optional debug
    std::cout << "Block solver: iter = " << LNsolver.iterations()
                << ", error = " << LNsolver.error() << "\n";
}

#endif // SOLVERCONGRAD_H