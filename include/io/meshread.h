#ifndef MESHREAD_H
#define MESHREAD_H

#include <iostream>
#include <fstream>
#include <cstdlib>    // For exit()
#include <string>

#include <vector>
#include <array>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

// Structure to hold mesh data and computed geometric quantities.
struct MeshData {
    int n_nodes;      // Number of nodes
    int n_faces;      // Number of faces
    int n_cells;      // Number of cells
    int n_fwalls;     // Number of wall faces

    vector<double> r_node;  // Nodal coordinates (n_nodes x 2)
    vector<int> f2n;     // Face-to-node connectivity (n_faces x 2)
    vector<int> f2c;     // Face-to-cell connectivity (n_faces x 2)

    vector<double> r_f;     // Face midpoints (n_faces x 2)
    vector<double> r_w;     // Wall face midpoints (n_fwalls x 2)
    vector<double> n_f;     // Face normal vectors (n_faces x 2)
    vector<double> A;       // Face "areas" (or lengths in 2D) (n_faces)
    
    vector<double> r_c;     // Cell centroids (n_cells x 2)
    vector<double> V;       // Cell volumes (or areas) (n_cells)

    vector<double> Ixx;     // Moment/inertia-like quantity (n_cells)
    vector<double> Iyy;     // Moment/inertia-like quantity (n_cells)
    vector<double> Ixy;     // Moment/inertia-like quantity (n_cells)
    vector<double> delta;   // Determinant-like quantity (n_cells)

    vector<int> c2n_tri; // Triangulated cell-to-node connectivity

    // testing, this is use to replace accumulation sum in least-squares
    vector<int> neighbor_flat;     // flat neighbor list
    vector<int> neighbor_count;    // number of neighbors per cell
    vector<int> neighbor_offset;   // prefix sum for starting index in neighbor_flat

    vector<int> c2f_flat;    // flattened list of faces per cell
    vector<int> c2f_offset;
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
    mesh.r_node.resize(mesh.n_nodes * 2);
    mesh.f2n.resize(mesh.n_faces * 2);
    mesh.f2c.resize(mesh.n_faces * 2);

    // Read nodal coordinates.
    for (int i = 0; i < mesh.n_nodes; ++i) {
        in >> mesh.r_node[2*i] >> mesh.r_node[2*i+1];
    }
    
    // Read face-to-node connectivity.
    for (int i = 0; i < mesh.n_faces; ++i) {
        int node1, node2;
        in >> node1 >> node2;
        mesh.f2n[2*i] = node1;
        mesh.f2n[2*i+1] = node2;
    }
    
    // Read face-to-cell connectivity.
    for (int i = 0; i < mesh.n_faces; ++i) {
        int cell1, cell2;
        in >> cell1 >> cell2;
        mesh.f2c[2*i] = cell1;
        mesh.f2c[2*i+1] = cell2;
    }
    
    in.close();
    
    // Allocate geometric data containers.
    mesh.r_f.resize(mesh.n_faces * 2, 0.0);
    mesh.n_f.resize(mesh.n_faces * 2, 0.0);
    mesh.A.resize(mesh.n_faces);
    
    mesh.r_c.resize(mesh.n_cells * 2, 0.0);
    mesh.V.resize(mesh.n_cells, 0.0);
    
    // Auxiliary vectors to accumulate moments for centroid calculation.
    vector<double> xc_n(mesh.n_cells, 0.0);
    vector<double> yc_n(mesh.n_cells, 0.0);
    vector<double> r1(2, 0.0), r2(2, 0.0); // Temporary storage for nodal coordinates
    mesh.n_fwalls = 0;
    mesh.r_w.resize(mesh.n_fwalls * 2);
    
    // Compute face midpoints, normals, face "areas", and contributions to cell volumes.
    for (int i = 0; i < mesh.n_faces; ++i) {
        // Adjust indices from one-indexed to zero-indexed.
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;
        int n1 = mesh.f2n[2*i] - 1;
        int n2 = mesh.f2n[2*i+1] - 1;
        
        // Get nodal coordinates.
        r1[0] = mesh.r_node[2*n1];
        r1[1] = mesh.r_node[2*n1+1];
        r2[0] = mesh.r_node[2*n2];
        r2[1] = mesh.r_node[2*n2+1];
        
        // Compute the face midpoint.
        mesh.r_f[2*i] = 0.5 * (r1[0] + r2[0]);
        mesh.r_f[2*i+1] = 0.5 * (r1[1] + r2[1]);
        
        // Compute an unnormalized normal by rotating the edge (r2 - r1) 90° clockwise.
        mesh.n_f[2*i] = r2[1] - r1[1];
        mesh.n_f[2*i+1] = r1[0] - r2[0];
        
        // Compute face "area" (or length in 2D) and normalize the normal.
        mesh.A[i] = sqrt(mesh.n_f[2*i] * mesh.n_f[2*i] + mesh.n_f[2*i+1] * mesh.n_f[2*i+1]);
        if (mesh.A[i] < 1e-12) {
            cerr << "Warning: near-zero face area at face " << i << endl;
            mesh.A[i] = 1e-12; // or skip this face
        }
        mesh.n_f[2*i] /= mesh.A[i];
        mesh.n_f[2*i+1] /= mesh.A[i];
        
        // Compute contributions to cell volume and centroid moments.
        double factor = 0.5 * (mesh.r_f[2*i] * mesh.n_f[2*i] + mesh.r_f[2*i+1] * mesh.n_f[2*i+1]) * mesh.A[i];
        double tempx  = (mesh.r_f[2*i] * mesh.n_f[2*i] + mesh.r_f[2*i+1] * mesh.n_f[2*i+1]) * mesh.r_f[2*i] * mesh.A[i];
        double tempy  = (mesh.r_f[2*i] * mesh.n_f[2*i] + mesh.r_f[2*i+1] * mesh.n_f[2*i+1]) * mesh.r_f[2*i+1] * mesh.A[i];
        
        mesh.V[c1]   += factor;
        xc_n[c1]     += tempx;
        yc_n[c1]     += tempy;
        
        // If the second cell index is valid, update its contributions.
        if (c2 >= 0) {
            mesh.V[c2]   -= factor;
            xc_n[c2]     -= tempx;
            yc_n[c2]     -= tempy;
        }
        if (c2 == -1) {
            mesh.r_w.resize(2*mesh.n_fwalls + 2);
            mesh.r_w[2*mesh.n_fwalls] = mesh.r_f[2*i]; 
            mesh.r_w[2*mesh.n_fwalls+1] = mesh.r_f[2*i+1];
            mesh.n_fwalls += 1;
        }
    }
    
    cout << "  Number of wall faces: " << mesh.n_fwalls << endl;

    // Build cell-to-cell neighbors
    mesh.neighbor_count.resize(mesh.n_cells, 0);

    // count neighbors per cell
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;

        if (c1 >= 0) mesh.neighbor_count[c1] += (c2 >= 0 ? 1 : 0); // only valid neighbors
        if (c2 >= 0) mesh.neighbor_count[c2] += 1;                 // valid neighbors
    }

    // Compute prefix sum for neighbor offsets
    mesh.neighbor_offset.resize(mesh.n_cells + 1, 0);
    for (int i = 0; i < mesh.n_cells; ++i) {
        mesh.neighbor_offset[i+1] = mesh.neighbor_offset[i] + mesh.neighbor_count[i];
    }

    // Allocate flat neighbor array
    mesh.neighbor_flat.resize(mesh.neighbor_offset.back(), -1);

    // Temporary counters to keep track of insertion positions
    std::vector<int> insert_pos(mesh.n_cells, 0);
    for (int i = 0; i < mesh.n_cells; ++i) insert_pos[i] = mesh.neighbor_offset[i];

    // Fill the neighbor_flat array
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;

        if (c1 >= 0 && c2 >= 0) {
            mesh.neighbor_flat[insert_pos[c1]++] = c2;
            mesh.neighbor_flat[insert_pos[c2]++] = c1;
        }
    }

    // Compute cell-to-face connectivity
    // Step 1: Count number of faces per cell
    std::vector<int> face_count(mesh.n_cells, 0);
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;
        if (c1 >= 0) face_count[c1]++;
        if (c2 >= 0) face_count[c2]++;
    }

    // Step 2: Compute offsets (prefix sum)
    mesh.c2f_offset.resize(mesh.n_cells + 1, 0);
    for (int i = 0; i < mesh.n_cells; ++i)
        mesh.c2f_offset[i+1] = mesh.c2f_offset[i] + face_count[i];

    // Step 3: Fill flat array
    mesh.c2f_flat.resize(mesh.c2f_offset.back());
    std::vector<int> insert_pos = mesh.c2f_offset; // temporary counter
    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;
        if (c1 >= 0) mesh.c2f_flat[insert_pos[c1]++] = i;
        if (c2 >= 0) mesh.c2f_flat[insert_pos[c2]++] = i;
    }

    // Compute cell centroids from accumulated moments.
    for (int i = 0; i < mesh.n_cells; ++i) {
        if (mesh.V[i] == 0) {
            cerr << "Warning: Zero cell volume encountered for cell " << i << endl;
            continue;
        }
        mesh.r_c[2*i] = (1.0 / 3.0) * (xc_n[i] / mesh.V[i]);
        mesh.r_c[2*i+1] = (1.0 / 3.0) * (yc_n[i] / mesh.V[i]);
    }

    // Compute moments for use in reconstruction.
    vector<double> Ixx_temp(mesh.n_cells, 0.0); 
    vector<double> Iyy_temp(mesh.n_cells, 0.0);
    vector<double> Ixy_temp(mesh.n_cells, 0.0);

    for (int i = 0; i < mesh.n_faces; ++i) {
        int c1 = mesh.f2c[2*i] - 1;
        int c2 = mesh.f2c[2*i+1] - 1;
        if (c2 >= 0) {
            double tempIxx = (mesh.r_c[2*c1] - mesh.r_c[2*c2]) * (mesh.r_c[2*c1] - mesh.r_c[2*c2]); 
            double tempIyy = (mesh.r_c[2*c1+1] - mesh.r_c[2*c2+1]) * (mesh.r_c[2*c1+1] - mesh.r_c[2*c2+1]);
            double tempIxy = (mesh.r_c[2*c1] - mesh.r_c[2*c2]) * (mesh.r_c[2*c1+1] - mesh.r_c[2*c2+1]);
            Ixx_temp[c1] += tempIxx;
            Iyy_temp[c1] += tempIyy;
            Ixy_temp[c1] += tempIxy;
            Ixx_temp[c2] += tempIxx;
            Iyy_temp[c2] += tempIyy;
            Ixy_temp[c2] += tempIxy;
        }
        else {
            // double tempIxx = 4 * (mesh.r_c(c1, 0) - mesh.r_f(i, 0)) * (mesh.r_c(c1, 0) - mesh.r_f(i, 0));
            // double tempIyy = 4 * (mesh.r_c(c1, 1) - mesh.r_f(i, 1)) * (mesh.r_c(c1, 1) - mesh.r_f(i, 1));
            // double tempIxy = 4 * (mesh.r_c(c1, 0) - mesh.r_f(i, 0)) * (mesh.r_c(c1, 1) - mesh.r_f(i, 1));
            double dxface = mesh.r_c[2*c1] - mesh.r_f[2*i];
            double dyface = mesh.r_c[2*c1+1] - mesh.r_f[2*i+1];
            double tempIxx = 4.0 * (dxface * mesh.n_f[2*i]) * (dxface * mesh.n_f[2*i]);
            double tempIyy = 4.0 * (dyface * mesh.n_f[2*i+1]) * (dyface * mesh.n_f[2*i+1]);
            double tempIxy = 4.0 * (dxface * mesh.n_f[2*i]) * (dyface * mesh.n_f[2*i+1]);
            Ixx_temp[c1] += tempIxx;
            Iyy_temp[c1] += tempIyy;
            Ixy_temp[c1] += tempIxy;
        }
    }

    mesh.delta.resize(mesh.n_cells);
    mesh.Ixx.resize(mesh.n_cells);
    mesh.Iyy.resize(mesh.n_cells);
    mesh.Ixy.resize(mesh.n_cells);
    for (int i = 0; i < mesh.n_cells; ++i) {
        mesh.delta[i] = Ixx_temp[i] * Iyy_temp[i] - Ixy_temp[i] * Ixy_temp[i];
        if (std::abs(mesh.delta[i]) < 1e-12) {
            cerr << "Warning: delta is very small for cell " << i << endl;
        }
        mesh.Ixx[i] = Ixx_temp[i] / mesh.delta[i];
        mesh.Iyy[i] = Iyy_temp[i] / mesh.delta[i];
        mesh.Ixy[i] = Ixy_temp[i] / mesh.delta[i];
    }

    // ---- Triangulate cells and fill mesh.c2n_tri ----
    cout << "Triangulating mesh file ..." << endl;

    // --- Optimized triangulation of cells with Eigen output ---
    vector<vector<int>> cell_to_faces(mesh.n_cells);

    // Step 1: Map each cell to its faces
    for (int i = 0; i < mesh.n_faces; ++i) {
        for (int s = 0; s < 2; ++s) {
            int c = mesh.f2c[2*i + s] - 1;
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
            int n0 = mesh.f2n[2*f];
            int n1 = mesh.f2n[2*f+1];
            if (mesh.f2c[2*f] - 1 == c)
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
    mesh.c2n_tri.resize(n_tri*3);
    for (int i = 0; i < n_tri; ++i) {
        mesh.c2n_tri[3*i] = temp_triangles[i][0];
        mesh.c2n_tri[3*i+1] = temp_triangles[i][1];
        mesh.c2n_tri[3*i+2] = temp_triangles[i][2];
    }
    cout << "Completed triangulation." << endl;

    return mesh;
}

// Helper to print 2D matrix stored in flat vector
template<typename T>
void writeMatrix(std::ostream &out, const std::vector<T> &vec, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            out << vec[i * n_cols + j];
            if (j < n_cols - 1) out << " ";
        }
        out << "\n";
    }
}

#endif // MESHREAD_H
