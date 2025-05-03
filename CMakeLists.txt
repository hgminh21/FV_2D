cmake_minimum_required(VERSION 3.10)
project(FV_2D VERSION 0.1.0 LANGUAGES C CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find PETSc
# PETSc directory hint from user
# set(PETSC_DIR "" CACHE PATH "Path to the PETSc installation directory")
set(PETSC_DIR $ENV{PETSC_DIR} CACHE PATH "Path to the PETSc installation directory")

# Try to find PETSc include dir
find_path(PETSC_INCLUDE_DIR
    NAMES petscsys.h
    HINTS ${PETSC_DIR}
    PATH_SUFFIXES include
    DOC "Path to PETSc include directory"
)

# Try to find PETSc library (libpetsc.so or libpetsc.a)
find_library(PETSC_LIBRARY
    NAMES petsc
    HINTS ${PETSC_DIR}
    PATH_SUFFIXES lib
    DOC "Path to PETSc library"
)

if(NOT PETSC_INCLUDE_DIR OR NOT PETSC_LIBRARY)
    message(FATAL_ERROR "PETSc not found. Please set PETSC_DIR with -DPETSC_DIR=/path/to/petsc")
endif()

message(STATUS "Found PETSc include dir: ${PETSC_INCLUDE_DIR}")
message(STATUS "Found PETSc library: ${PETSC_LIBRARY}")


# Find Eigen
set(EIGEN_DIR $ENV{EIGEN_DIR} CACHE PATH "Path to Eigen3 headers")

# If EIGEN_DIR is not set from the environment, attempt to find it
if(NOT EIGEN_DIR)
    find_path(EIGEN_DIR
        NAMES Eigen
        PATH_SUFFIXES eigen3
        DOC "Path to Eigen3 headers"
    )
    
    # If EIGEN_DIR is still not found, give an error
    if(NOT EIGEN_DIR)
        message(FATAL_ERROR "Could not find Eigen3. Please set EIGEN_DIR manually with -DEIGEN_DIR=/path/to/eigen3")
    endif()
endif()

message(STATUS "Found Eigen3 headers in: ${EIGEN_DIR}")

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add executable
add_executable(FV_2D ./src/main.cpp)

# Include directories
target_include_directories(FV_2D PRIVATE
    ${PETSC_INCLUDE_DIR}
    ${EIGEN_DIR}
    ${MPI_C_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/include
)

# Link PETSc and MPI
target_link_libraries(FV_2D PRIVATE
    OpenMP::OpenMP_CXX
    ${PETSC_LIBRARY}
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
)

# Optional: Set custom output directory via -DOUTPUT_DIR=...
if(NOT DEFINED OUTPUT_DIR)
    set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
endif()
set_target_properties(FV_2D PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)
