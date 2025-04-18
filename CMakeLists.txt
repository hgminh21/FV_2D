cmake_minimum_required(VERSION 3.10.0)
project(FV_2D VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler-specific flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops")
endif()

# Add the executable
add_executable(FV_2D main.cpp)

# Include Eigen headers
target_include_directories(FV_2D PRIVATE /path/to/eigen)
