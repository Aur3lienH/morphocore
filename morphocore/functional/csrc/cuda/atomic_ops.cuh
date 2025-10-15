#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============ DOUBLE PRECISION ATOMIC ADD ============
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native support available on compute capability >= 6.0
#else
__device__ inline double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ============ HALF PRECISION ATOMIC ADD ============
// Only define if not already available (CUDA < 10.0 or compute capability < 7.0)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700 && !defined(__CUDA_ATOMIC_HALF__)
// Custom implementation for compute capability < 7.0
__device__ inline __half atomicAdd(__half* address, __half val) {
    unsigned int* address_as_uint = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    unsigned short old_as_us;

    do {
        assumed = old;
        old_as_us = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        
        __half sum = __float2half(__half2float(*(__half*)&old_as_us) + __half2float(val));
        unsigned short sum_as_us = *((unsigned short*)&sum);
        
        unsigned int new_val = (size_t)address & 2 
            ? (old & 0xffff) | (sum_as_us << 16)
            : (old & 0xffff0000) | sum_as_us;
            
        old = atomicCAS(address_as_uint, assumed, new_val);
    } while (assumed != old);
    
    old_as_us = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return *(__half*)&old_as_us;
}
#endif