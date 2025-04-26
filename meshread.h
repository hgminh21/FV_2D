#ifndef MESHREAD_H
#define MESHREAD_H

#include <iostream>
#include <fstream>
#include <cstdlib>    // For exit()
#include <string>
#include <Eigen/Dense>

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;
using namespace Eigen;

// Structure to hold mesh data and computed geometric quantities.
struct MeshData {
    int n_nodes;      // Number of nodes
    int n_faces;      // Number of faces
    int n_cells;      // Number of cells
    int n_fwalls;     // Number of wall faces

    MatrixXd r_node;  // Nodal coordinates (n_nodes x 2)
    MatrixXi f2n;     // Face-to-node connectivity (n_faces x 2)
    MatrixXi f2c;     // Face-to-cell connectivity (n_faces x 2)

    MatrixXd r_f;     // Face midpoints (n_faces x 2)
    MatrixXd r_w;     // Wall face midpoints (n_fwalls x 2)
    MatrixXd n_f;     // Face normal vectors (n_faces x 2)
    VectorXd A;       // Face "areas" (or lengths in 2D) (n_faces)
    
    MatrixXd r_c;     // Cell centroids (n_cells x 2)
    VectorXd V;       // Cell volumes (or areas) (n_cells)

    VectorXd Ixx;     // Moment/inertia-like quantity (n_cells)
    VectorXd Iyy;     // Moment/inertia-like quantity (n_cells)
    VectorXd Ixy;     // Moment/inertia-like quantity (n_cells)
    VectorXd delta;   // Determinant-like quantity (n_cells)

    MatrixXi c2n_tri; // Triangulated cell-to-node connectivity

};

// Function to read the mesh file, compute geometric data, and return a MeshData struct.
// The input file is assumed to be formatted as follows:
//   n_nodes n_faces n_cells
//   [n_nodes lines with 2 doubles (r_node)]
//   [n_faces lines with 2 integers (f2n, one-indexed)]
//   [n_faces lines with 2 integers (f2c, one-indexed; negative values for boundaries)]
MeshData readMesh(const string &filename) {
    MeshData mesh;
    ifstream in(filename);
    if (!in) {
        cerr << "Error opening mesh file: " << filename << endl;
        exit(1);
    }

    cout << "Reading mesh file ..." << endl;

    // Read mesh header data
    in >> mesh.n_nodes >> mesh.n_faces >> mesh.n_cells;
    
    // Print mesh summary
    cout << "  Number of nodes: " << mesh.n_nodes << endl;
    cout << "  Number of faces: " << mesh.n_faces << endl;
    cout << "  Number of cells: " << mesh.n_cells << endl; 

    // Resize containers appropriately.
    mesh.r_node.resize(mesh.n_nodes, 2);
    mesh.f2n.resize(mesh.n_faces, 2);
    mesh.f2c.resize(mesh.n_faces, 2);

    // Read nodal coordinates.
    for (int i = 0; i < mesh.n_nodes; ++i) {
        in >> mesh.r_node(i, 0) >> mesh.r_node(i, 1);
    }
    
    // Read face-to-node connectivity.
    for (int i = 0; i < mesh.n_faces; ++i) {
        int node1, node2;
        in >> node1 >> node2;
        mesh.f2n(i, 0) = node1;
        mesh.f2n(i, 1) = node2;
    }
    
    // Read face-to-cell connectivity.
    for (int i = 0; i < mesh.n_faces; ++i) {
        int cell1, cell2;
        in >> cell1 >> cell2;
        mesh.f2c(i, 0) = cell1;
        mesh.f2c(i, 1) = cell2;
    }
    
    in.close();
    
    // Allocate geometric data containers.
    mesh.r_f = MatrixXd::Zero(mesh.n_faces, 2);
    mesh.n_f = MatrixXd::Zero(mesh.n_faces, 2);
    mesh.A   = VectorXd::Zero(mesh.n_faces);
    
    mesh.r_c = MatrixXd::Zero(mesh.n_cells, 2);
    mesh.V   = VectorXd::Zero(mesh.n_cells);
    
    // Auxiliary vectors to accumulate moments for centroid calculation.
    VectorXd xc_n = VectorXd::Zero(mesh.n_cells);
    VectorXd yc_n = VectorXd::Zero(mesh.n_cells);
    
    mesh.n_fwalls = 0;
    mesh.r_w.resize(mesh.n_fwalls, 2);
    
    // Compute face midpoints, normals, face "areas", and contributions to cell volumes.
    for (int i = 0; i < mesh.n_faces; ++i) {
        // Adjust indices from one-indexed to zero-indexed.
        int c1 = mesh.f2c(i, 0) - 1;
        int c2 = mesh.f2c(i, 1) - 1;
        int n1 = mesh.f2n(i, 0) - 1;
        int n2 = mesh.f2n(i, 1) - 1;
        
        // Get nodal coordinates.
        Vector2d r1 = mesh.r_node.row(n1);
        Vector2d r2 = mesh.r_node.row(n2);
        
        // Compute the face midpoint.
        mesh.r_f(i, 0) = 0.5 * (r1(0) + r2(0));
        mesh.r_f(i, 1) = 0.5 * (r1(1) + r2(1));
        
        // Compute an unnormalized normal by rotating the edge (r2 - r1) 90° clockwise.
        mesh.n_f(i, 0) = r2(1) - r1(1);
        mesh.n_f(i, 1) = r1(0) - r2(0);
        
        // Compute face "area" (or length in 2D) and normalize the normal.
        mesh.A(i) = mesh.n_f.row(i).norm();
        mesh.n_f.row(i) /= mesh.A(i);
        
        // Compute contributions to cell volume and centroid moments.
        double factor = 0.5 * (mesh.r_f(i, 0) * mesh.n_f(i, 0) + mesh.r_f(i, 1) * mesh.n_f(i, 1)) * mesh.A(i);
        double tempx  = (mesh.r_f(i, 0) * mesh.n_f(i, 0) + mesh.r_f(i, 1) * mesh.n_f(i, 1)) * mesh.r_f(i, 0) * mesh.A(i);
        double tempy  = (mesh.r_f(i, 0) * mesh.n_f(i, 0) + mesh.r_f(i, 1) * mesh.n_f(i, 1)) * mesh.r_f(i, 1) * mesh.A(i);
        
        mesh.V(c1)   += factor;
        xc_n(c1)     += tempx;
        yc_n(c1)     += tempy;
        
        // If the second cell index is valid, update its contributions.
        if (c2 >= 0) {
            mesh.V(c2)   -= factor;
            xc_n(c2)     -= tempx;
            yc_n(c2)     -= tempy;
        }
        if (c2 == -1) {
            mesh.r_w.conservativeResize(mesh.n_fwalls + 1, Eigen::NoChange);
            mesh.r_w.row(mesh.n_fwalls) = mesh.r_f.row(i); 
            mesh.n_fwalls += 1;
        }
    }
    
    cout << "  Number of wall faces: " << mesh.n_fwalls << endl;

    // Compute cell centroids from accumulated moments.
    for (int i = 0; i < mesh.n_cells; ++i) {
        if (mesh.V(i) == 0) {
            cerr << "Warning: Zero cell volume encountered for cell " << i << endl;
            continue;
        }
        mesh.r_c(i, 0) = (1.0 / 3.0) * (xc_n(i) / mesh.V(i));
        mesh.r_c(i, 1) = (1.0 / 3.0) * (yc_n(i) / mesh.V(i));
    }

    // Compute moments for use in reconstruction.
    VectorXd Ixx_temp = VectorXd::Zero(mesh.n_cells); 
    VectorXd Iyy_temp = VectorXd::Zero(mesh.n_cells);
    VectorXd Ixy_temp = VectorXd::Zero(mesh.n_cells);

    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c(i, 0) - 1;
        int c2 = mesh.f2c(i, 1) - 1;
        if (c2 >= 0) {
            double tempIxx = (mesh.r_c(c1, 0) - mesh.r_c(c2, 0)) * (mesh.r_c(c1, 0) - mesh.r_c(c2, 0)); 
            double tempIyy = (mesh.r_c(c1, 1) - mesh.r_c(c2, 1)) * (mesh.r_c(c1, 1) - mesh.r_c(c2, 1));
            double tempIxy = (mesh.r_c(c1, 0) - mesh.r_c(c2, 0)) * (mesh.r_c(c1, 1) - mesh.r_c(c2, 1));
            Ixx_temp(c1) += tempIxx;
            Iyy_temp(c1) += tempIyy;
            Ixy_temp(c1) += tempIxy;
            Ixx_temp(c2) += tempIxx;
            Iyy_temp(c2) += tempIyy;
            Ixy_temp(c2) += tempIxy;
        }
        else {
            // double tempIxx = 4 * (mesh.r_c(c1, 0) - mesh.r_f(i, 0)) * (mesh.r_c(c1, 0) - mesh.r_f(i, 0));
            // double tempIyy = 4 * (mesh.r_c(c1, 1) - mesh.r_f(i, 1)) * (mesh.r_c(c1, 1) - mesh.r_f(i, 1));
            // double tempIxy = 4 * (mesh.r_c(c1, 0) - mesh.r_f(i, 0)) * (mesh.r_c(c1, 1) - mesh.r_f(i, 1));
            double dxface = mesh.r_c(c1, 0) - mesh.r_f(i, 0);
            double dyface = mesh.r_c(c1, 1) - mesh.r_f(i, 1);
            double tempIxx = 4.0 * (dxface * mesh.n_f(i, 0)) * (dxface * mesh.n_f(i, 0));
            double tempIyy = 4.0 * (dyface * mesh.n_f(i, 1)) * (dyface * mesh.n_f(i, 1));
            double tempIxy = 4.0 * (dxface * mesh.n_f(i, 0)) * (dyface * mesh.n_f(i, 1));
            Ixx_temp(c1) += tempIxx;
            Iyy_temp(c1) += tempIyy;
            Ixy_temp(c1) += tempIxy;
        }
    }
    
    mesh.delta = Ixx_temp.array() * Iyy_temp.array() - Ixy_temp.array().square();
    // Check for very small delta values before division
    for (int i = 0; i < mesh.n_cells; ++i) {
        if (abs(mesh.delta(i)) < 1e-12) {
            cerr << "Warning: delta is very small for cell " << i << endl;
        }
    }
    mesh.Ixx = Ixx_temp.array() / mesh.delta.array();
    mesh.Iyy = Iyy_temp.array() / mesh.delta.array();
    mesh.Ixy = Ixy_temp.array() / mesh.delta.array();

    // ---- Triangulate cells and fill mesh.c2n_tri ----
    cout << "Triangulating mesh file ..." << endl;

    // --- Optimized triangulation of cells with Eigen output ---
    vector<vector<int>> cell_to_faces(mesh.n_cells);

    // Step 1: Map each cell to its faces
    for (int i = 0; i < mesh.n_faces; ++i) {
        for (int s = 0; s < 2; ++s) {
            int c = mesh.f2c(i, s) - 1;
            if (c >= 0) cell_to_faces[c].push_back(i);
        }
    }

    // Step 2: Temporary triangle storage using vector
    vector<array<int, 3>> temp_triangles;

    // Buffers reused per cell
    unordered_map<int, int> next_node;
    vector<int> ordered_nodes;

    for (int c = 0; c < mesh.n_cells; ++c) {
        next_node.clear();
        ordered_nodes.clear();

        const auto& faces = cell_to_faces[c];

        // Build "next node" mapping from face edges
        for (int f : faces) {
            int n0 = mesh.f2n(f, 0);
            int n1 = mesh.f2n(f, 1);
            if (mesh.f2c(f, 0) - 1 == c)
                next_node[n0] = n1;
            else
                next_node[n1] = n0;
        }

        // Reconstruct ordered node loop
        int start = next_node.begin()->first;
        int curr = start;

        do {
            ordered_nodes.push_back(curr);
            curr = next_node[curr];
        } while (curr != start && ordered_nodes.size() <= faces.size() + 1);

        // Fan triangulation
        for (int i = 1; i + 1 < ordered_nodes.size(); ++i) {
            temp_triangles.push_back({ordered_nodes[0], ordered_nodes[i], ordered_nodes[i + 1]});
        }
    }

    // Step 3: Transfer to Eigen::MatrixXi
    int n_tri = temp_triangles.size();
    mesh.c2n_tri.resize(n_tri, 3);
    for (int i = 0; i < n_tri; ++i) {
        mesh.c2n_tri(i, 0) = temp_triangles[i][0];
        mesh.c2n_tri(i, 1) = temp_triangles[i][1];
        mesh.c2n_tri(i, 2) = temp_triangles[i][2];
    }
    cout << "Completed triangulation." << endl;

    return mesh;
}

// Function to output (write) the mesh data to a text file.
void outputMeshData(const MeshData &mesh, const string &filename) {
    ofstream out(filename);
    if (!out) {
        cerr << "Error opening file " << filename << " for output." << endl;
        return;
    }
    
    out << "# Mesh Data Output\n";
    out << "# n_nodes: " << mesh.n_nodes << "\n";
    out << "# n_faces: " << mesh.n_faces << "\n";
    out << "# n_cells: " << mesh.n_cells << "\n\n";
    
    out << "# Nodal Coordinates (r_node):\n" << mesh.r_node << "\n\n";
    out << "# Face-to-Node Connectivity (f2n):\n" << mesh.f2n << "\n\n";
    out << "# Face-to-Cell Connectivity (f2c):\n" << mesh.f2c << "\n\n";
    out << "# Face Midpoints (r_f):\n" << mesh.r_f << "\n\n";
    out << "# Face Normals (n_f):\n" << mesh.n_f << "\n\n";
    out << "# Face Areas (A):\n" << mesh.A << "\n\n";
    out << "# Cell Volumes (V):\n" << mesh.V << "\n\n";
    out << "# Cell Centroids (r_c):\n" << mesh.r_c << "\n\n";
    out << "# Ixx:\n" << mesh.Ixx << "\n\n";
    out << "# Iyy:\n" << mesh.Iyy << "\n\n";
    out << "# Ixy:\n" << mesh.Ixy << "\n\n";
    out << "# delta:\n" << mesh.delta << "\n\n";
    out << "# Triangulated Cell-to-Node Connectivity (c2n_tri): \n" << mesh.c2n_tri << "\n\n";
    
    out.close();
}

#endif // MESHREAD_H
