#include "smorph.cuh"
#include "smorph_kernels.cuh"
#include "../cpp/morpho_op_enum.hpp"
#include <stdio.h>
#include <torch/torch.h>
#include <iostream>

template <MORPHO_MERGE merge, typename scalar_t>
void morpho_smorph_forward_kernel(const scalar_t* input, const scalar_t* weight, const scalar_t* alpha, scalar_t* output,
                                  scalar_t* references_indexes, scalar_t* sum_values, int batch_size, int input_channel, int output_channel,
                                  int height, int width, int kernel_height, int kernel_width, int kernel_height_half, int kernel_width_half)
{
    const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
    const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;

    int64_t total_weight_size = output_channel * input_channel * kernel_height * kernel_width;
    int64_t total_input_size = input_channel * PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;
    int64_t shared_memory_size = sizeof(scalar_t) * (total_weight_size + total_input_size);


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (shared_memory_size > prop.sharedMemPerBlock) {
        printf("ERROR: Shared memory EXCEEDS device limit!\n");
    }
    
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, batch_size);

    smorph_forward_kernel<merge, scalar_t>
        <<<blocks, threads, shared_memory_size>>>(input, weight, alpha, output, references_indexes, sum_values, batch_size, input_channel,
                                                  output_channel, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
}

template <MORPHO_MERGE merge, typename scalar_t>
void morpho_smorph_backward_kernel(const scalar_t* grad_output, scalar_t* grad_input, scalar_t* grad_weight, scalar_t* grad_alpha,
                                   const scalar_t* alpha, const scalar_t* input, const scalar_t* weights, int batch_size, int input_channels,
                                   int output_channels, int height, int width, int kernel_height, int kernel_width, int kernel_height_half,
                                   int kernel_width_half)
{
    dim3 threads(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 blocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, batch_size);

    const int threads_per_block = TILE_WIDTH * TILE_HEIGHT;
    const int kernel_size = kernel_height * kernel_width;
    size_t shared_mem_size = 2 * threads_per_block * kernel_size * sizeof(scalar_t);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (shared_mem_size > prop.sharedMemPerBlock)
    {
        if (shared_mem_size <= prop.sharedMemPerBlockOptin)
        {
            printf("Requesting extended shared memory: %zu bytes (default: %zu, max: %zu)\n",
                   shared_mem_size, prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
            
            cudaError_t err = cudaFuncSetAttribute(
                smorph_backward_kernel<merge, scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_size
            );
            
            if (err != cudaSuccess)
            {
                printf("ERROR: Failed to set shared memory size: %s\n", cudaGetErrorString(err));
                return;
            }
            
            cudaFuncSetAttribute(
                smorph_backward_kernel<merge, scalar_t>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100
            );
        }
        else
        {
            printf("ERROR: Shared memory required (%zu bytes) EXCEEDS device maximum (%zu bytes)!\n",
                   shared_mem_size, prop.sharedMemPerBlockOptin);
            printf("Kernel: %dx%d=%d, Tiles: %dx%d=%d threads, Type size: %zu bytes\n",
                   kernel_height, kernel_width, kernel_size,
                   TILE_WIDTH, TILE_HEIGHT, threads_per_block,
                   sizeof(scalar_t));
            printf("Consider: reducing TILE_WIDTH/TILE_HEIGHT or using global memory\n");
            return;
        }
    }
    
    smorph_backward_kernel<merge, scalar_t><<<blocks, threads, shared_mem_size>>>(
        grad_output, grad_input, grad_weight, grad_alpha, alpha, input,
        weights, batch_size, input_channels, output_channels, height, width,
        kernel_height, kernel_width, kernel_height_half, kernel_width_half);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error in smorph_backward: %s\n", cudaGetErrorString(err));
    }
}

#define INSTANTIATE_SMORPH_KERNELS(MERGE_TYPE, SCALAR_T)                                                                                             \
    template void morpho_smorph_forward_kernel<MERGE_TYPE, SCALAR_T>(                                                                                \
        const SCALAR_T* input, const SCALAR_T* weight, const SCALAR_T* alpha, SCALAR_T* output, SCALAR_T* references_indexes, SCALAR_T* sum_values,  \
        int batch_size, int input_channel, int output_channel, int height, int width, int kernel_height, int kernel_width, int kernel_height_half,   \
        int kernel_width_half);                                                                                                                      \
                                                                                                                                                     \
    template void morpho_smorph_backward_kernel<MERGE_TYPE, SCALAR_T>(                                                                               \
        const SCALAR_T* grad_output, SCALAR_T* grad_input, SCALAR_T* grad_weight, SCALAR_T* grad_alpha, const SCALAR_T* alpha,                       \
        const SCALAR_T* input, const SCALAR_T* weights, int batch_size, int input_channels, int output_channels, int height, int width,              \
        int kernel_height, int kernel_width, int kernel_height_half, int kernel_width_half);

INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::ADD, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::AVERAGE, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MIN, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MAX, float)

INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::ADD, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::AVERAGE, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MIN, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MAX, double)