cmake_minimum_required(VERSION 3.10)
project(FV_2D VERSION 0.1.0 LANGUAGES C CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# PETSc install location
set(PETSC_DIR "/path/to/petsc")

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add executable
add_executable(FV_2D ./src/main.cpp)

# Include directories
target_include_directories(FV_2D PRIVATE
    ${PETSC_DIR}/include
    ${MPI_C_INCLUDE_PATH}
    /path/to/eigen3
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/flux
    ${CMAKE_SOURCE_DIR}/include/implicit
    ${CMAKE_SOURCE_DIR}/include/io
)

# Link PETSc and MPI (use full path to libpetsc.so)
target_link_libraries(FV_2D PRIVATE
    ${PETSC_DIR}/lib/libpetsc.so
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
)

# Add OpenMP flags and link
target_link_libraries(FV_2D PRIVATE OpenMP::OpenMP_CXX)

# Might need this 
target_compile_definitions(FV_2D PRIVATE EIGEN_DONT_PARALLELIZE)

# Output directory
set_target_properties(FV_2D PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY /path/to/output/directory
)
