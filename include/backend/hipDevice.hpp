#ifndef HIPDEVICE_HPP
#define HIPDEVICE_HPP

#include <hip/hip_runtime.h>
#include <iostream>
#define deviceFunction __device__

#define HIP_CHECK(cmd) \
    { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << hipGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    }

// Kernel wrapper that calls the functor operator() on the device
template <typename T>
__global__ void deviceLaunchFunctorWrapper(T functor) {
    functor(threadIdx.x + blockIdx.x * blockDim.x);
    // functor(idx);
}

class HIPdevice {
public:
    void Malloc(void** ptr, size_t N) noexcept {
        HIP_CHECK(hipMalloc(ptr, N));
    }

    void MemcpyDeviceToHost(void* dst, const void* src, size_t count) noexcept {
        HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
        // HIP_CHECK(hipDeviceSynchronize());
    }

    void MemcpyHostToDevice(void* dst, const void* src, size_t count) noexcept {
        HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
        // HIP_CHECK(hipDeviceSynchronize());
    }

    void Free(void* ptr) noexcept {
        HIP_CHECK(hipFree(ptr));
    }

    // void Synchronize() const noexcept {
    //     HIP_CHECK(hipDeviceSynchronize());
    // }

    template <typename T>
    void LaunchKernel(unsigned int Nblock, unsigned int Nthread, const T& functor) {
        deviceLaunchFunctorWrapper<T><<<Nblock, Nthread>>>(functor);
        // HIP_CHECK(hipLaunchKernelGGL(deviceLaunchFunctorWrapper, dim3(Nblock), dim3(Nthread), 0, 0, functor));
        // HIP_CHECK(hipGetLastError());
        // HIP_CHECK(hipDeviceSynchronize());
    }
};

#endif // HIPDEVICE_HPP