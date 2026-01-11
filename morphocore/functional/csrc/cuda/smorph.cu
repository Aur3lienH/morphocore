#include "smorph.cuh"
#include "smorph_kernels.cuh"
#include "smorph_backward.cuh"
#include "smorph_forward_identity.cuh"
#include "smorph_backward_identity.cuh"
#include "../cpp/morpho_op_enum.hpp"
#include <stdio.h>
#include <torch/torch.h>
#include <iostream>

template <MORPHO_MERGE merge, typename scalar_t>
void morpho_smorph_forward_kernel(const scalar_t* input, const scalar_t* weight, const scalar_t* alpha, scalar_t* output,
                                  scalar_t* references_indexes, scalar_t* sum_values, int batch_size, int input_channel, int output_channel,
                                  int height, int width, int kernel_height, int kernel_width, int kernel_height_half, int kernel_width_half)
{
    if constexpr (merge == MORPHO_MERGE::IDENTITY)
    {
        // IDENTITY mode: input_channels == output_channels
        const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
        const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;

        int64_t total_weight_size = input_channel * kernel_height * kernel_width;
        int64_t total_input_size = input_channel * PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;
        int64_t shared_memory_size = sizeof(scalar_t) * (total_weight_size + total_input_size);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        if (shared_memory_size > prop.sharedMemPerBlock) {
            cudaFuncSetAttribute(
                smorph_forward_identity_kernel<scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_memory_size
            );
        }
        
        dim3 threads(TILE_WIDTH, TILE_HEIGHT);
        dim3 blocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, batch_size);

        smorph_forward_identity_kernel<scalar_t>
            <<<blocks, threads, shared_memory_size>>>(input, weight, alpha, output, references_indexes, sum_values, batch_size, input_channel,
                                                      height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
    }
    else
    {
        // Regular merge modes
        const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
        const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;

        int64_t total_weight_size = output_channel * input_channel * kernel_height * kernel_width;
        int64_t total_input_size = input_channel * PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;
        int64_t shared_memory_size = sizeof(scalar_t) * (total_weight_size + total_input_size);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        if (shared_memory_size > prop.sharedMemPerBlock) {
            cudaFuncSetAttribute(
                smorph_forward_kernel<merge, scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_memory_size
            );
        }
        
        dim3 threads(TILE_WIDTH, TILE_HEIGHT);
        dim3 blocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT, batch_size);

        smorph_forward_kernel<merge, scalar_t>
            <<<blocks, threads, shared_memory_size>>>(input, weight, alpha, output, references_indexes, sum_values, batch_size, input_channel,
                                                      output_channel, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in smorph_forward: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Merge mode: " << static_cast<int>(merge) << std::endl;
        std::cerr.flush();
        throw std::runtime_error(std::string("CUDA Error in forward: ") + cudaGetErrorString(err));
    }
}

template <MORPHO_MERGE merge, typename scalar_t>
void morpho_smorph_backward_kernel(
    const scalar_t* grad_output, scalar_t* grad_input, scalar_t* grad_weight, 
    scalar_t* grad_alpha, const scalar_t* alpha, const scalar_t* input, 
    const scalar_t* weights, int batch_size, int input_channels,
    int output_channels, int height, int width, int kernel_height, 
    int kernel_width, int kernel_height_half, int kernel_width_half)
{
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (width + TILE_WIDTH - 1) / TILE_WIDTH, 
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT, 
        batch_size
    );

    if constexpr (merge == MORPHO_MERGE::IDENTITY)
    {
        // IDENTITY mode: input_channels == output_channels
        smorph_backward_identity_kernel<scalar_t><<<blocks, threads>>>(
            grad_output, grad_input, grad_weight, grad_alpha, alpha, input,
            weights, batch_size, input_channels, height, width,
            kernel_height, kernel_width, kernel_height_half, kernel_width_half
        );
    }
    else if constexpr (merge == MORPHO_MERGE::MIN || merge == MORPHO_MERGE::MAX)
    {
        // MIN/MAX merge modes
        smorph_backward_minmax_kernel<merge, scalar_t><<<blocks, threads>>>(
            grad_output, grad_input, grad_weight, grad_alpha, alpha, input,
            weights, batch_size, input_channels, output_channels, height, width,
            kernel_height, kernel_width, kernel_height_half, kernel_width_half
        );
    }
    else
    {
        // ADD/AVERAGE merge modes
        smorph_backward_kernel<merge, scalar_t><<<blocks, threads>>>(
            grad_output, grad_input, grad_weight, grad_alpha, alpha, input,
            weights, batch_size, input_channels, output_channels, height, width,
            kernel_height, kernel_width, kernel_height_half, kernel_width_half
        );
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        throw std::runtime_error(
            "CUDA Error in smorph_backward: " + std::string(cudaGetErrorString(sync_err)) +
            "\nMerge mode: " + std::to_string(static_cast<int>(merge)) +
            "\nInput channels: " + std::to_string(input_channels) +
            "\nKernel size: " + std::to_string(kernel_height * kernel_width)
        );
    }
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        throw std::runtime_error(
            "CUDA Launch Error in smorph_backward: " + std::string(cudaGetErrorString(launch_err))
        );
    }
}

// Template instantiation macro
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

// Instantiate for all merge modes and scalar types
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::ADD, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::AVERAGE, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MIN, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MAX, float)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::IDENTITY, float)

INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::ADD, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::AVERAGE, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MIN, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::MAX, double)
INSTANTIATE_SMORPH_KERNELS(MORPHO_MERGE::IDENTITY, double)