#ifdef USECUDA
    #include "backend/cudaDevice.cuh"
    typedef CUDAdevice device;
#elif defined USESYCL
    #include "backend/syclDevice.hpp"
    typedef SYCLdevice device; 
#elif defined USEHIP
    #include "backend/hipDevice.hpp"
    typedef HIPdevice device;      
#else
     #include "backend/ompDevice.hpp"
     typedef OMPdevice device;
#endif

// // Cross-platform helper to get optimal threads per block / work-group size
#include <omp.h>  // at the top of your header/source
unsigned int getOptimalThreadsPerBlock(device& d) {
#ifdef USECUDA
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    return props.warpSize * 4; // safe multiple of warp (e.g., 128)
#elif defined USESYCL
    auto dev = d.queue.get_device();
    return dev.get_info<sycl::info::device::max_work_group_size>() / 2; 
#elif defined USEOMP
    return omp_get_max_threads();
#else
    return 64; // fallback
#endif
}

