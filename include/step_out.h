#ifndef STEP_OUT_H
#define STEP_OUT_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// #include <CGNS>

using namespace std;

void StepOutput(
    const vector<vector<double>>& Mesh,
    const vector<vector<double>>& Q_out,
    vector<vector<double>>& Q_step,
    vector<double>& rho,
    vector<double>& u,
    vector<double>& v,
    vector<double>& p
) {
    int n_nodes = Mesh[0][0];
    int n_faces = Mesh[0][1];
    int n_cells = Mesh[0][2];

    double gamma = 1.4;

    vector<vector<double>> f2n(Mesh.begin() + 1 + n_faces, Mesh.begin() + 1 + 2 * n_faces); // Face to node mapping
    vector<vector<double>> f2c(Mesh.begin() + 1 + 2 * n_faces, Mesh.begin() + 1 + 3 * n_faces); // Face to cell mapping
    vector<vector<double>> n_f(Mesh.begin() + 1 + 3 * n_faces, Mesh.begin() + 1 + 4 * n_faces); // Face normal vector
    vector<vector<double>> r_nodes(Mesh.begin() + 1 + 4 * n_faces + n_cells, Mesh.begin() + 1 + 4 * n_faces + n_cells + n_nodes);  // 2D Nodes coordinate 

    // Post-processing
    Q_step.resize(n_nodes, vector<double>(4, 0.0));
    vector<vector<double>> n_shared(n_nodes, vector<double>(1, 0.0));
    rho.resize(n_nodes), u.resize(n_nodes), v.resize(n_nodes), p.resize(n_nodes);
    
    for (int i = 0; i < n_faces; ++i) {
        int c1 = f2c[i][0] - 1;
        int c2 = f2c[i][1] - 1;
        int n1 = f2n[i][0] - 1;
        int n2 = f2n[i][1] - 1;
        if (c2 >= 0) {
            n_shared[n1][0] = n_shared[n1][0] + 1;
            n_shared[n2][0] = n_shared[n2][0] + 1;
            Q_step[n1][0] = Q_step[n1][0] + (Q_out[c1][0] + Q_out[c2][0]) / 2.0;
            Q_step[n2][0] = Q_step[n2][0] + (Q_out[c1][0] + Q_out[c2][0]) / 2.0;
            Q_step[n1][1] = Q_step[n1][1] + (Q_out[c1][1] + Q_out[c2][1]) / 2.0;
            Q_step[n2][1] = Q_step[n2][1] + (Q_out[c1][1] + Q_out[c2][1]) / 2.0;
            Q_step[n1][2] = Q_step[n1][2] + (Q_out[c1][2] + Q_out[c2][2]) / 2.0;
            Q_step[n2][2] = Q_step[n2][2] + (Q_out[c1][2] + Q_out[c2][2]) / 2.0;
            Q_step[n1][3] = Q_step[n1][3] + (Q_out[c1][3] + Q_out[c2][3]) / 2.0;
            Q_step[n2][3] = Q_step[n2][3] + (Q_out[c1][3] + Q_out[c2][3]) / 2.0;
        }
        else {
            n_shared[n1][0] = n_shared[n1][0] + 1;
            n_shared[n2][0] = n_shared[n2][0] + 1;
            Q_step[n1][0] = Q_step[n1][0] + Q_out[c1][0];
            Q_step[n2][0] = Q_step[n2][0] + Q_out[c1][0];
            Q_step[n1][1] = Q_step[n1][1] + Q_out[c1][1];
            Q_step[n2][1] = Q_step[n2][1] + Q_out[c1][1];
            Q_step[n1][2] = Q_step[n1][2] + Q_out[c1][2];
            Q_step[n2][2] = Q_step[n2][2] + Q_out[c1][2];
            Q_step[n1][3] = Q_step[n1][3] + Q_out[c1][3];
            Q_step[n2][3] = Q_step[n2][3] + Q_out[c1][3];
        }
    }

    for (int i = 0; i < n_nodes; ++i) {
        Q_step[i][0] = Q_step[i][0] / n_shared[i][0];
        Q_step[i][1] = Q_step[i][1] / n_shared[i][0];
        Q_step[i][2] = Q_step[i][2] / n_shared[i][0];
        Q_step[i][3] = Q_step[i][3] / n_shared[i][0];
        rho[i] = Q_step[i][0];
        u[i] = Q_step[i][1] / rho[i];
        v[i] = Q_step[i][2] / rho[i];
        double E = Q_step[i][3];
        p[i] = (E - 0.5 * rho[i] * (u[i] * u[i] + v[i] * v[i])) / (gamma - 1);
    }

    // int dim = 2;
    // string filename = "output.cgns";

    // // Open CGNS file
    // int index_file, index_base, index_zone, index_coord;
    // cgsize_t isize[3][1] = {{n_nodes}, {n_cells}, {0}};

    // if (cg_open(filename.c_str(), CG_MODE_WRITE, &index_file)) cg_error_exit();

    // // Create base
    // if (cg_base_write(index_file, "Base", dim, dim, &index_base)) cg_error_exit();

    // // Create zone
    // if (cg_zone_write(index_file, index_base, "Zone", *isize, Unstructured, &index_zone)) cg_error_exit();

    // // Write coordinates (assuming 2D mesh)
    // vector<double> x(n_nodes), y(n_nodes);
    // for (int i = 0; i < n_nodes; ++i) {
    //     x[i] = r_nodes[i][0];
    //     y[i] = r_nodes[i][1];
    // }
    // if (cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateX", x.data(), &index_coord)) cg_error_exit();
    // if (cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateY", y.data(), &index_coord)) cg_error_exit();

    // // Write elements (assume we have triangles/quads - adapt as needed)
    // vector<cgsize_t> conn;
    // vector<int> element_nodes;

    // // Loop through each cell to collect element connectivity
    // for (int i = 0; i < n_cells; ++i) {
    //     int f1 = f2c[i][0] - 1; // Get face indices for cell i
    //     int f2 = f2c[i][1] - 1;

    //     element_nodes.clear();
        
    //     // Add nodes for face 1
    //     for (int j = 0; j < f2n[f1].size(); ++j) {
    //         element_nodes.push_back(f2n[f1][j] - 1); // Convert to 0-based index
    //     }
        
    //     // Add nodes for face 2 (if relevant)
    //     if (f2 >= 0) {
    //         for (int j = 0; j < f2n[f2].size(); ++j) {
    //             element_nodes.push_back(f2n[f2][j] - 1); // Convert to 0-based index
    //         }
    //     }
        
    //     // Write element connectivity (cell-wise, can adapt if mixed elements)
    //     for (int node : element_nodes) {
    //         conn.push_back(node + 1); // Convert back to 1-based indexing for CGNS
    //     }
    // }

    // int index_section;
    // if (cg_section_write(index_file, index_base, index_zone, "Elements", TRI_3,
    //                      1, n_cells, 0, conn.data(), &index_section)) cg_error_exit();

    // // Write solution fields at nodes
    // int index_sol;
    // if (cg_sol_write(index_file, index_base, index_zone, "FlowSolution", Vertex, &index_sol)) cg_error_exit();

    // if (cg_field_write(index_file, index_base, index_zone, index_sol, RealDouble, "Density", rho.data(), nullptr)) cg_error_exit();
    // if (cg_field_write(index_file, index_base, index_zone, index_sol, RealDouble, "VelocityX", u.data(), nullptr)) cg_error_exit();
    // if (cg_field_write(index_file, index_base, index_zone, index_sol, RealDouble, "VelocityY", v.data(), nullptr)) cg_error_exit();
    // if (cg_field_write(index_file, index_base, index_zone, index_sol, RealDouble, "Pressure", p.data(), nullptr)) cg_error_exit();

    // cg_close(index_file);
    // std::cout << "CGNS file '" << filename << "' written successfully.\n";
}

#endif