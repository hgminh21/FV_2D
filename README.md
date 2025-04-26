# FV_2D

A 2D finite volume solver.

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/) (header-only library)
- [PETSc](https://petsc.org/release/) (if want to use implitcit)

## Build Instructions

```bash
mkdir build && cd build
cmake ..
cmake --build .

## Input Instructions

```in
[solver]
order_accuracy = 2      # order of accuracy input
max_step = 100          # maximum iterations
monitor_step = 10       # print residuals and coefficients
output_step = 2500      # output result files

[meshfile]
file = ./grid_file.in   # mesh file location

[flow]
type = 1                # governing equation 1: Euler, 2: Navier-Stokes
rho = 1.4               # density
u = 0.85                # X-direction velocity
v = 0                   # Y-direction velocity
p = 1                   # pressure
gamma = 1.4             # specific heat reatio

[time]
dt = 1e-4               # fixed time step (will be avoided if use cfl)
use_cfl = 0             # 0: use fixed time step, 1: use cfl number
CFL = 1                 # input cfl number < 1
method = implicit       # explicit or implicit time scheme (implicit is still in develop)
local_dt = 0            # 0: use global time step, 1: use local time step

## To use live update with Tecplot 360

- Have access to Tecplot 360 EX 2021 R2 or later
- Python 3.11 or later 
- Scripting -> PyTecplot Connections -> check Accept connections
```bash
python live_viewer.py