mkdir build
cd build
cmake -DCMAKE_CUDA_HOST_COMPILER=clang++-18 -DBUILD_CUDA=ON ..
make VERBOSE=1
cmake --fresh -DBUILD_OMP=ON ..
make VERBOSE=1
cmake --fresh -DCMAKE_CXX_COMPILER=acpp -DBUILD_SYCL=ON ..
make VERBOSE=1
cmake --fresh -DCMAKE_CXX_COMPILER=hipcc -DBUILD_HIP=ON ..
make VERBOSE=1