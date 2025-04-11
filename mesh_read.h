#ifndef MESH_READ_H
#define MESH_READ_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

// Function to read the input file and compute mesh data
void Mesh_read(const string& input_filename, vector<vector<double>>& Mesh) {
    ifstream in(input_filename);
    
    if (!in) {
        cerr << "Error opening the file!" << endl;
        return;
    }

    int n_nodes, n_faces, n_cells;
    in >> n_nodes >> n_faces >> n_cells; // Read mesh information

    // Initialize data sizes
    vector<vector<double>> r_node(n_nodes, vector<double>(2, 0.0));
    vector<vector<double>> f_nodes(n_faces, vector<double>(2, 0.0));
    vector<vector<double>> f_cells(n_faces, vector<double>(2, 0.0));

    // Read nodal vectors
    double d1, d2;
    for (int i = 0; i < n_nodes; ++i) {
        in >> d1 >> d2;
        r_node[i] = {d1, d2};
    }
    // Read face-point connections
    for (int i = 0; i < n_faces; ++i) {
        in >> d1 >> d2;
        f_nodes[i] = {d1, d2};
    }
    // Read face-cell connections
    for (int i = 0; i < n_faces; ++i) {
        in >> d1 >> d2;
        f_cells[i] = {d1, d2};
    }

    in.close(); 

    // Compute geometry data
    int c1, c2, n1, n2;
    double temp, tempx, tempy;
    vector<double> r1(2), r2(2), xc_n(n_cells), yc_n(n_cells);

    vector<vector<double>> r_f(n_faces, vector<double>(2, 0.0)); // Face direction vector
    vector<vector<double>> n_f(n_faces, vector<double>(2, 0.0)); // Face normal vector
    vector<vector<double>> r_c(n_cells, vector<double>(2, 0.0)); // Cell centroid vector
    vector<vector<double>> A(n_faces, vector<double>(1, 0.0)); // Interfaces area
    vector<vector<double>> V(n_cells, vector<double>(1, 0.0)); // Cell volume

    // Calculate face data, volume, and centroid for cells
    for (int i = 0; i < n_faces; ++i) {
        c1 = f_cells[i][0] - 1;
        c2 = f_cells[i][1] - 1;
        n1 = f_nodes[i][0] - 1;
        n2 = f_nodes[i][1] - 1;
        r1 = r_node[n1];
        r2 = r_node[n2];
        r_f[i][0] = 0.5 * (r1[0] + r2[0]);
        r_f[i][1] = 0.5 * (r1[1] + r2[1]);
        n_f[i][0] = r2[1] - r1[1];
        n_f[i][1] = r1[0] - r2[0];
        A[i][0] = sqrt(pow(n_f[i][1], 2) + pow(n_f[i][0], 2));
        n_f[i][0] = n_f[i][0] / A[i][0];
        n_f[i][1] = n_f[i][1] / A[i][0];     
        temp = 0.5 * (r_f[i][0] * n_f[i][0] + r_f[i][1] * n_f[i][1]) * A[i][0];
        tempx = (r_f[i][0] * n_f[i][0] + r_f[i][1] * n_f[i][1]) * r_f[i][0] * A[i][0];
        tempy = (r_f[i][0] * n_f[i][0] + r_f[i][1] * n_f[i][1]) * r_f[i][1] * A[i][0];
        V[c1][0] += temp;
        xc_n[c1] += tempx;
        yc_n[c1] += tempy;
        if (c2 >= 0) {
            V[c2][0] -= temp;
            xc_n[c2] -= tempx;
            yc_n[c2] -= tempy; 
        }
    }

    // vector<double> Ixx(n_cells), Ixy(n_cells), Iyy(n_cells);
    // Calculate cell centroids
    for (int i = 0; i < n_cells; ++i) {
        r_c[i][0] = 1.0 / 3.0 * xc_n[i] / V[i][0];
        r_c[i][1] = 1.0 / 3.0 * yc_n[i] / V[i][0];
    }

    // for (int i = 0; i < n_faces; ++i) {
    //     c1 = f_cells[i][0] - 1;
    //     c2 = f_cells[i][1] - 1;
    //     if (c2 >= 0) {
    //         double tempx = (r_c[c1][0] - r_c[c2][0]) * (r_c[c1][0] - r_c[c2][0]);
    //         double tempy = (r_c[c1][1] - r_c[c2][1];
            
    //     }    
    // }
    // Output matrix
    // Add the mesh metadata as the first row
    vector<double> mesh_metadata = {static_cast<double>(n_nodes), static_cast<double>(n_faces), static_cast<double>(n_cells)};
    Mesh.push_back(mesh_metadata);
    // Push computed data to Mesh
    for (const auto& row : r_f) Mesh.push_back(row);
    for (const auto& row : f_nodes) Mesh.push_back(row); 
    for (const auto& row : f_cells) Mesh.push_back(row);
    for (const auto& row : n_f) Mesh.push_back(row);
    for (const auto& row : r_c) Mesh.push_back(row);
    for (const auto& row : r_node) Mesh.push_back(row);
    for (const auto& row : A) Mesh.push_back(row);
    for (const auto& row : V) Mesh.push_back(row);
    
}

#endif // MESH_READ_H
