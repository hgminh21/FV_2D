#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "meshread.h"

using namespace Eigen;
using std::string;
// using std::cout;
// using std::endl;

// Structures for flow and solver
struct Flow {
    double rho;
    double p;
    double u;
    double v;
    double gamma;
    double Pr;
    double R;
    double mu;
    double Cp;
    double T;
    double k;
    int type;
};

struct Solver {
    double CFL;
    int n_step;
    int order;
    int m_step;
    int o_step;
};

// Simple input parser (INI-style)
bool parse_input_file(const std::string& filename, Flow &flow, Solver &solver, std::string &mesh_file) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Cannot open input file: " << filename << std::endl;
        return false;
    }

    std::string line, section;
    while (std::getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '[') {
            section = line;
            continue;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (section == "[solver]") {
            if (key == "order_accuracy") solver.order = std::stoi(value);
            else if (key == "max_step") solver.n_step = std::stoi(value);
            else if (key == "monitor_step") solver.m_step = std::stoi(value);
            else if (key == "output_step") solver.o_step = std::stoi(value);
            else if (key == "CFL") solver.CFL = std::stod(value);
        } else if (section == "[meshfile]") {
            if (key == "file") mesh_file = value;
        } else if (section == "[flow]") {
            if (key == "type") flow.type = std::stoi(value);
            else if (key == "rho") flow.rho = std::stod(value);
            else if (key == "u") flow.u = std::stod(value);
            else if (key == "v") flow.v = std::stod(value);
            else if (key == "p") flow.p = std::stod(value);
            else if (key == "gamma") flow.gamma = std::stod(value);
            else if (key == "Pr") flow.Pr = std::stod(value);
            else if (key == "R") flow.R = std::stod(value);
            else if (key == "mu") flow.mu = std::stod(value);
        }
    }

    return true;
}

// Main initialization function
void initialize(const std::string &input_file, MeshData &mesh, Flow &flow, Solver &solver, Vector4d &Q_init, MatrixXd &Q)
{
    std::string mesh_filename;

    // Parse config file
    if (!parse_input_file(input_file, flow, solver, mesh_filename)) {
        std::cerr << "Input parsing failed. Check the input file format." << std::endl;
        exit(1);
    }

    // Compute derived physical properties
    if (flow.type == 1) { // Euler
        flow.Pr = 0;
        flow.R = 0;
        flow.mu = 0;
        flow.Cp = 0;
        flow.T = 0;
        flow.k = 0;
    } else if (flow.type == 2) { // Navier-Stokes
        // Use parsed values for Pr, R, mu
        flow.Cp = flow.gamma / (flow.gamma - 1.0) * flow.Pr;
        flow.T = flow.p / (flow.rho * flow.R);
        flow.k = flow.mu * flow.Cp / flow.Pr;
    }

    // Load mesh
    mesh = readMesh(mesh_filename);
    cout << "Finished reading mesh from " << mesh_filename << endl;

    if (flow.type == 1) {std::cout << "Equation : Euler" << std::endl;}
    else if (flow.type == 2) {std::cout << "Equation : Navier-Stokes" << std::endl;}
    if (solver.order == 1) {std::cout << "Order of accuracy = 1" << std::endl;}
    else if (solver.order == 2) {std::cout << "Order of accuracy = 2" << std::endl;}

    // Initial conserved variables
    double E = flow.p / (flow.gamma - 1.0) + 0.5 * flow.rho * (flow.u * flow.u + flow.v * flow.v);
    Q_init << flow.rho, flow.rho * flow.u, flow.rho * flow.v, E;

    Q = MatrixXd::Zero(mesh.n_cells, 4);
    for (int i = 0; i < mesh.n_cells; ++i) {
        Q.row(i) = Q_init.transpose();
    }
}

#endif // INITIALIZE_H
