# FV_2D

A 2D finite volume solver for compressible flows.

## Features

- Flexible Governing Equations
   + Supports both inviscid (Euler) and viscous (Navier-Stokes) flow simulations (2nd-order only for Navier-Stokes).

- Hybrid Mesh Support
   + Reads unstructured 2D meshes from external mesh files.

- Initial Flow Specification
   + User-defined initial conditions for density, velocity components, pressure, and thermodynamic properties.

- Second-Order Reconstruction Schemes
   + Gradient reconstruction via linear, least-square, or gauss-green methods.
   + Optional slope limiters: Squeeze or Venkatakrishnan (2nd-order only).

- Multiple Riemann Solvers
   + Choice of flux calculation methods: Rusanov, Lax-Friedrichs, or Roe.

- Flexible Time Integration

   + Supports both explicit and implicit schemes (implicit under development).
   + Time step control: Fixed or CFL-based.
   + Global or local time stepping.

- Simulation Control
   + Configurable number of steps, output frequency, and residual monitoring.
   + Clean and modular input through a human-readable config file.
   + Real-time visualization capability with Tecplot 360.

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/) (header-only library)
- [PETSc](https://petsc.org/release/) (required for implicit time-stepping)

## Build Instructions

```bash
mkdir build && cd build
cmake ..
cmake --build .
```
## Execute instructions

- Prepare your mesh file 
   + Mesh file need to be `.in` format
   + Only support face-based mesh (need face-to-node and face-to-cell connectivity)
   + Support arbitrary mesh (hybrid elements, hanging nodes, ghost cells, etc)
   + Ghost cell index 0 : wall boundary
   + Ghost cell index -1: free-stream boundary

- Prepare your input file

```bash
./FV_2D ./input.in
```


## Input File Instructions

Example input file (`.in` format):

```in
[solver]
max_step = 10000         # Maximum number of time steps
monitor_step = 100       # Frequency of printing residuals and coefficients
output_step = 1000       # Frequency of writing output files

[meshfile]
file = ./grid_file.in    # Path to mesh file

[flow]
type = 1                 # 1: Euler equations, 2: Navier-Stokes equations(only for 2nd-order)
rho = 1.4                # Initial density
u = 0.85                 # Initial velocity in X-direction
v = 0                    # Initial velocity in Y-direction
p = 1                    # Initial pressure
gamma = 1.4              # Specific heat ratio
Pr = 0.72                # Prandtl number    (ignored if `type` = 1)
R = 287                  # Gas constant      (ignored if `type` = 1)
mu = 7.08662e-2          # Dynamic viscosity (ignored if `type` = 1)

[reconstruct]
method = least-square    # reconstruction method: "linear", "least-square" or "gauss-green"
use_lim = 1              # 0: Not using limiter, 1: Squeeze limiter, 2: Venkat limiter (Only for 2nd-order)

[flux]
method = rusanov         # Riemann solver: "rusanov", "lax-friedrichs" or "roe"

[time]
dt = 1e-4                # Fixed time step (ignored if `use_cfl` = 1)
use_cfl = 0              # 0: Fixed time step, 1: CFL-based time step
CFL = 1                  # CFL number (should be < 1)
method = implicit        # Time-stepping method: "explicit" or "implicit" (implicit under development, only works for 2nd-order)
local_dt = 0             # 0: Global time step, 1: Local time step
```

## Live Update with Tecplot 360

To use live visualization with Tecplot:

1. Make sure you have **Tecplot 360 EX 2021 R2** (or newer).
2. Install **Python 3.11** (or newer).
3. In Tecplot:  
   Go to `Scripting -> PyTecplot Connections -> Accept Connections` and check the box.
4. Run the live viewer script:

```bash
python live_viewer.py
```