#ifndef SYCLDEVICE_HPP
#define SYCLDEVICE_HPP

#include <sycl/sycl.hpp>
#include <iostream>
#define deviceFunction

template <typename T>
class KernelNameWrapper {}; 

class SYCLdevice{
    sycl::queue queue;
public:
    SYCLdevice() {
        sycl::device selected_device;

        // Scan for a GPU device
        std::cout << "Scanning for available devices:\n";
        for (const auto& platform : sycl::platform::get_platforms()) {
            for (const auto& device : platform.get_devices()) {
                std::cout << "  Found: " << device.get_info<sycl::info::device::name>() << " ("
                        << (device.is_gpu() ? "GPU" :
                            device.is_cpu() ? "CPU" :
                            device.is_accelerator() ? "Accelerator" : "Unknown")
                        << ")\n";

                if (device.is_gpu()) {
                    selected_device = device;
                    break;
                }
            }
            if (selected_device.is_gpu()) break;
        }

        // If no GPU found, fall back to default
        if (!selected_device.is_gpu()) {
            std::cout << "No GPU found. Falling back to default device.\n";
            selected_device = sycl::device{ sycl::default_selector_v };
        }

        queue = sycl::queue{ selected_device, sycl::property::queue::in_order() };

        std::cout << "\nUsing device: " 
                << selected_device.get_info<sycl::info::device::name>() << "\n"
                << "Vendor: " 
                << selected_device.get_info<sycl::info::device::vendor>() << "\n"
                << "Driver version: "
                << selected_device.get_info<sycl::info::device::driver_version>() << "\n\n";
    }

    int Malloc(void** ptr, const int64_t N) {
		(*ptr) = sycl::malloc_device(N, queue.get_device(), queue.get_context());
		queue.wait();
		return 0;
	}

    void MemcpyDeviceToHost(void* dst, const void* src, const int64_t count) {
		// queue.wait(); // dont need to wait here, as the memcpy will block until done
		queue.memcpy(dst, src, count);
		queue.wait();
	}

    void MemcpyHostToDevice(void* dst, const void* src, const int64_t count) {
        // queue.wait(); // same
        queue.memcpy(dst, src, count);
        queue.wait();
    }

    void Free(void* block) {
		queue.wait(); 
		sycl::free(block, queue);
	}
    
    template <typename T>
    void LaunchKernel(const unsigned int Nblock, const unsigned int Nthread, const T& functor) {
        auto i = Nblock * Nthread;
        queue.submit([&](sycl::handler& h) {
            // h.parallel_for(i, functor);
            h.parallel_for<KernelNameWrapper<T>>(sycl::range<1>{i}, functor);
            });
    }
};

#endif // SYCLDEVICE_HPP
