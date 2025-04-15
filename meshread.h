#ifndef MESHREAD_H
#define MESHREAD_H

#include <iostream>
#include <fstream>
#include <cstdlib>    // For exit()
#include <string>
#include <Eigen>

using namespace std;
using namespace Eigen;

// Structure to hold mesh data and computed geometric quantities.
struct MeshData {
    int n_nodes;      // Number of nodes
    int n_faces;      // Number of faces
    int n_cells;      // Number of cells

    MatrixXd r_node;  // Nodal coordinates (n_nodes x 2)
    MatrixXi f2n;     // Face-to-node connectivity (n_faces x 2)
    MatrixXi f2c;     // Face-to-cell connectivity (n_faces x 2)

    MatrixXd r_f;     // Face midpoints (n_faces x 2)
    MatrixXd n_f;     // Face normal vectors (n_faces x 2)
    VectorXd A;       // Face "areas" (or lengths in 2D) (n_faces)
    
    MatrixXd r_c;     // Cell centroids (n_cells x 2)
    VectorXd V;       // Cell volumes (or areas) (n_cells)

    VectorXd Ixx;     // Moment/inertia-like quantity (n_cells)
    VectorXd Iyy;     // Moment/inertia-like quantity (n_cells)
    VectorXd Ixy;     // Moment/inertia-like quantity (n_cells)
    VectorXd delta;   // Determinant-like quantity (n_cells)
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

    // Read mesh header data
    in >> mesh.n_nodes >> mesh.n_faces >> mesh.n_cells;
    
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
    }
    
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
            double tempIxx = 4 * (mesh.r_c(c1, 0) - mesh.r_f(i, 0)) * (mesh.r_c(c1, 0) - mesh.r_f(i, 0));
            double tempIyy = 4 * (mesh.r_c(c1, 1) - mesh.r_f(i, 1)) * (mesh.r_c(c1, 1) - mesh.r_f(i, 1));
            double tempIxy = 4 * (mesh.r_c(c1, 0) - mesh.r_f(i, 0)) * (mesh.r_c(c1, 1) - mesh.r_f(i, 1));
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
    
    out.close();
}

// Function to write wall face coordinates to a text file.
void writeWallFaceCoordinates(const MeshData &mesh, const std::string &filename = "wall.txt") {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening " << filename << " for writing.\n";
        return;
    }

    for (int i = 0; i < mesh.n_faces; ++i) {
        // Check if the second cell index indicates a wall (boundary)
        if (mesh.f2c(i, 1) == -1) {
            int node1 = mesh.f2n(i, 0) - 1;
            int node2 = mesh.f2n(i, 1) - 1;

            // Get nodal coordinates.
            Vector2d pt1 = mesh.r_node.row(node1);
            Vector2d pt2 = mesh.r_node.row(node2);

            file << pt1(0) << " " << pt1(1) << "\n";
            file << pt2(0) << " " << pt2(1) << "\n";
            file << "\n";  // Separate faces
        }
    }

    file.close();
    cout << "Boundary face node coordinates written to " << filename << endl;
}

#endif // MESHREAD_H
