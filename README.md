# FV_2D

A 2D finite volume solver for compressible flows.

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/) (header-only library)
- [PETSc](https://petsc.org/release/) (required for implicit time-stepping)

## Build Instructions

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Input File Instructions

Example input file (`.in` format):

```in
[solver]
order_accuracy = 2       # Order of accuracy (e.g., 1 for first-order, 2 for second-order)
max_step = 100           # Maximum number of time steps
monitor_step = 10        # Frequency of printing residuals and coefficients
output_step = 2500       # Frequency of writing output files

[meshfile]
file = ./grid_file.in    # Path to mesh file

[flow]
type = 1                 # 1: Euler equations, 2: Navier-Stokes equations
rho = 1.4                # Initial density
u = 0.85                 # Initial velocity in X-direction
v = 0                    # Initial velocity in Y-direction
p = 1                    # Initial pressure
gamma = 1.4              # Specific heat ratio

[time]
dt = 1e-4                # Fixed time step (ignored if `use_cfl` = 1)
use_cfl = 0              # 0: Fixed time step, 1: CFL-based time step
CFL = 1                  # CFL number (should be < 1)
method = implicit        # Time-stepping method: "explicit" or "implicit" (implicit under development)
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