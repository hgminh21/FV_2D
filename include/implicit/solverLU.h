#ifndef SOLVERLU_H
#define SOLVERLU_H

#include <Eigen/Dense>
#include <omp.h>

Eigen::MatrixXd solver_LU(const Eigen::MatrixXd& A_im1,
                         const Eigen::MatrixXd& A_im2,
                         const Eigen::MatrixXd& A_im3,
                         const Eigen::MatrixXd& A_im4,
                         const Eigen::MatrixXd& Res) {
    
    int n_faces = Res.rows();
    
    // Create dQt matrix to store the results (n_faces x 4)
    Eigen::MatrixXd dQt = Eigen::MatrixXd::Zero(n_faces, 4);
    
    // Precompute LU decompositions
    auto lu1 = A_im1.partialPivLu();
    auto lu2 = A_im2.partialPivLu();
    auto lu3 = A_im3.partialPivLu();
    auto lu4 = A_im4.partialPivLu();
    
    // Solve the linear systems in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Eigen::VectorXd Res_col = Res.col(0);
            Eigen::VectorXd dQt_col = lu1.solve(Res_col);
            
            // Critical section for writing to the shared output
            #pragma omp critical
            {
                dQt.col(0) = dQt_col;
            }
        }
        
        #pragma omp section
        {
            Eigen::VectorXd Res_col = Res.col(1);
            Eigen::VectorXd dQt_col = lu2.solve(Res_col);
            
            #pragma omp critical
            {
                dQt.col(1) = dQt_col;
            }
        }
        
        #pragma omp section
        {
            Eigen::VectorXd Res_col = Res.col(2);
            Eigen::VectorXd dQt_col = lu3.solve(Res_col);
            
            #pragma omp critical
            {
                dQt.col(2) = dQt_col;
            }
        }
        
        #pragma omp section
        {
            Eigen::VectorXd Res_col = Res.col(3);
            Eigen::VectorXd dQt_col = lu4.solve(Res_col);
            
            #pragma omp critical
            {
                dQt.col(3) = dQt_col;
            }
        }
    }
    
    return dQt;
}

#endif // SOLVERLU_H