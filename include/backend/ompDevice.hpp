#ifndef OMPDEVICE_HPP
#define OMPDEVICE_HPP

#include <stdlib.h>
#include <cstring>
#define deviceFunction
class OMPdevice{
public:
    int Malloc(void** ptr, size_t N){
        *ptr = malloc(N);
        return *ptr == nullptr;
    }
    void MemcpyDeviceToHost(void* dst, const void* src, size_t count){
        memcpy(dst, src, count);
    }
    void MemcpyHostToDevice(void* dst, const void* src, size_t count){
        memcpy(dst, src, count);
    }
    void Free(void* ptr){
        free(ptr);
    } 
    
    template <typename T>
    void LaunchKernel(const unsigned int Nblock, const unsigned int Nthread, const T& functor) const {
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(Nblock); i++) {
			const int64_t offset = i * static_cast<int64_t>(Nthread);
			for(int64_t j = offset; j < static_cast<int64_t>(offset+Nthread); functor(j++)){}
		}
	}
};

#endif // OMPDEVICE_HPP