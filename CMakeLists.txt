cmake_minimum_required(VERSION 3.10)
project(FV_2D VERSION 1.1.0 LANGUAGES C CXX)

# ===============================
# Compiler & Language Settings
# ===============================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===============================
# Build Options
# ===============================
option(BUILD_CPU   "Build the CPU/OpenMP version" ON)
option(BUILD_CUDA  "Build the CUDA version" OFF)
option(BUILD_OMP   "Build the OpenMP version" OFF)
option(BUILD_SYCL  "Build the SYCL version" OFF)
option(BUILD_HIP   "Build the HIP version" OFF)

# ===============================
# CPU / Reference Build
# ===============================
if(BUILD_CPU)
    add_executable(FV_2D ./src/main.cpp)
    target_include_directories(FV_2D PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    # Optional MPI/OpenMP support
    # find_package(MPI REQUIRED)
    # find_package(OpenMP REQUIRED)
    # target_link_libraries(FV_2D PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

    # Output directory
    if(NOT DEFINED OUTPUT_DIR)
        set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
    endif()
    set_target_properties(FV_2D PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
endif()

# ===============================
# CUDA Backend
# ===============================
if(BUILD_CUDA)
    cmake_policy(SET CMP0104 NEW)
    enable_language(CUDA)
    add_compile_options(-D USECUDA)
    add_executable(FV_2D_CUDA ./src/main.cu)
    target_include_directories(FV_2D_CUDA PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(FV_2D_CUDA PRIVATE -O3 --use_fast_math -arch=sm_70)
    set_target_properties(FV_2D_CUDA PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ===============================
# OpenMP Backend
# ===============================
if(BUILD_OMP)
    find_package(OpenMP REQUIRED)
    add_compile_options(-D USEOMP)
    add_executable(FV_2D_OMP ./src/main.cpp)
    target_include_directories(FV_2D_OMP PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(FV_2D_OMP PUBLIC OpenMP::OpenMP_CXX)
    set_target_properties(FV_2D_OMP PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ===============================
# SYCL Backend
# ===============================
if(BUILD_SYCL)
    set(SYCL_LIBRARY_DIR "/kuhpc/work/wang/shared/sw/adaptivecpp/25.02.0/lib")
    link_directories(${SYCL_LIBRARY_DIR})
    add_compile_options(-std=c++20 -D USESYCL)
    add_executable(FV_2D_SYCL ./src/main.cpp)
    target_include_directories(FV_2D_SYCL PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(FV_2D_SYCL PRIVATE -O3)
    set_target_properties(FV_2D_SYCL PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ===============================
# HIP Backend
# ===============================
if(BUILD_HIP)
    set(HIP_DIR "/kuhpc/sw/rocm/6.4.1")
    add_compile_options(-D USEHIP)
    add_executable(FV_2D_HIP ./src/main.cpp)
    target_include_directories(FV_2D_HIP PRIVATE ${CMAKE_SOURCE_DIR}/include "${HIP_DIR}/include")
    link_directories("$ENV{HIP_DIR}/lib")
    target_compile_options(FV_2D_HIP PRIVATE -O3 --offload-arch=gfx90a -ffast-math)
    target_link_options(FV_2D_HIP PRIVATE --offload-arch=gfx90a)
    set_target_properties(FV_2D_HIP PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    message(STATUS "Using HIP_DIR: ${HIP_DIR}")
endif()
