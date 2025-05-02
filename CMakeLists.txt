cmake_minimum_required(VERSION 3.10)
project(FV_2D VERSION 0.1.0 LANGUAGES C CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# PETSc install location
set(PETSC_DIR /path/to/petsc)

# Eigen install location
set(EIGEN_DIR /path/to/eigen3)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add executable
add_executable(FV_2D ./src/main.cpp)

# Include directories
target_include_directories(FV_2D PRIVATE
    ${PETSC_DIR}/include
    ${EIGEN_DIR}
    ${MPI_C_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/include
)

# Link PETSc and MPI (use full path to libpetsc.so)
target_link_libraries(FV_2D PRIVATE
    OpenMP::OpenMP_CXX
    ${PETSC_DIR}/lib/libpetsc.so
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
)

# Output directory
set_target_properties(FV_2D PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY /path/to/ouput/folder
)
