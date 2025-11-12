#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "../cpp/morpho_merge_policy.hpp"
#include "morpho_merge_policy.cuh"
#include "morpho_op_policy.cuh"
#include "morpho_op_kernels.cuh"
#include "morpho_op_kernel_forward_index.cuh"
#include "morpho_op_kernel_backward_index.cuh"
#include "morpho_op_kernel_backward.cuh"
#include "morpho_op_kernel_forward.cuh"
#include <iostream>

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t, bool save_indices>
void morpho_forward_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* output, int64_t* max_indexes, int batch_size, int in_channels,
                           int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batch_size);

    size_t shared_memory_size = 0;

    if constexpr (save_indices)
    {
        morpho_operation_forward_kernel_indices<Op, merge, scalar_t><<<blocks, threads, shared_memory_size>>>(
            input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width, kernel_height, kernel_width);
    }
    else
    {
        morpho_operation_forward_kernel<Op, merge, scalar_t><<<blocks, threads, shared_memory_size>>>(
            input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width, kernel_height, kernel_width);
    }
}

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t, bool save_indices>
void morpho_backward_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                            const scalar_t* grad_output, int batch_size, int in_channels, int out_channels, int height, int width, int padded_height,
                            int padded_width, int kernel_height, int kernel_width, const int64_t* max_indexes)
{
    // Common thread block size
    dim3 threads(16, 16);

    if constexpr (save_indices)
    {
        // --- OLD: Kernel 1: Calculate grad_input (Gather) ---
        /*
        dim3 blocks_gi((width + threads.x - 1) / threads.x,
                       (height + threads.y - 1) / threads.y,
                       batch_size * in_channels);

        morpho_operation_backward_kernel_grad_input<Op, scalar_t>
            <<<blocks_gi, threads>>>(max_indexes, grad_input, grad_output, batch_size,
                                     in_channels, out_channels, height, width,
                                     kernel_height, kernel_width);
        */
        
        // --- OLD: Kernel 2: Calculate grad_weight (Scatter) ---
        /*
        dim3 blocks_gw((width + threads.x - 1) / threads.x,
                       (height + threads.y - 1) / threads.y,
                       batch_size);

        morpho_operation_backward_kernel_grad_weight<Op, merge, scalar_t>
            <<<blocks_gw, threads>>>(max_indexes, grad_weight, grad_output, batch_size,
                                      in_channels, out_channels, height, width,
                                      kernel_height, kernel_width);
        */

        // --- NEW: Combined, Optimized Kernel (Scatter-Scatter) ---
        // Launch grid is (width, height, batch_size * out_channels)
        dim3 blocks_optimized((width + threads.x - 1) / threads.x,
                              (height + threads.y - 1) / threads.y,
                              batch_size * out_channels);

        // Call the new combined kernel
        // (Assuming it's named 'morpho_operation_backward_kernel_indices_OPTIMIZED'
        // and located in 'morpho_op_kernel_backward_index.cuh')
        morpho_operation_backward_kernel_indices<Op, merge, scalar_t>
            <<<blocks_optimized, threads>>>(max_indexes, grad_input, grad_weight, grad_output,
                                            batch_size, in_channels, out_channels, height,
                                            width, kernel_height, kernel_width);
    }
    else
    {
        // --- Re-computation Path ---
        dim3 blocks((width + threads.x - 1) / threads.x, 
                    (height + threads.y - 1) / threads.y, 
                    batch_size);
        
        // Call the wrapper function (NOT a kernel, so no <<<>>> syntax)
        morpho_operation_backward_kernel<Op, merge, scalar_t>(
            padded_input, weight, grad_input, grad_weight, grad_output, 
            batch_size, in_channels, out_channels, height, width, 
            padded_height, padded_width, kernel_height, kernel_width,
            blocks, threads);  // Pass blocks and threads as parameters
    }
}

template struct MorphoPolicyGPU<float, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicyGPU<float, MORPHO_OPERATION::EROSION>;
template struct MorphoPolicyGPU<double, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicyGPU<double, MORPHO_OPERATION::EROSION>;

#define INSTANTIATE_MORPHO_FORWARD_KERNEL(OPERATION, TYPE)                                                                                           \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE, true>(                                                                   \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE, true>(                                                                   \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE, false>(                                                                  \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE, false>(                                                                  \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::ADD, TYPE, false>(                                                                  \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::AVERAGE, TYPE, false>(                                                              \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template __global__ void morpho_operation_forward_kernel_indices<OPERATION, MORPHO_MERGE::MAX, TYPE>(                                            \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int kernel_height, int kernel_width);                                                                                             \
    template __global__ void morpho_operation_forward_kernel_indices<OPERATION, MORPHO_MERGE::MIN, TYPE>(                                            \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int kernel_height, int kernel_width);

INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::DILATION, float)
INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::DILATION, double)
INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::EROSION, float)
INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::EROSION, double)

#define INSTANTIATE_MORPHO_BACKWARD_KERNEL(OPERATION, TYPE)                                                                                          \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE, true>(                                                                  \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, \
        int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width,                           \
        const int64_t* max_indices);                                                                                                                 \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE, true>(                                                                  \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, \
        int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width,                           \
        const int64_t* max_indices);                                                                                                                 \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE, false>(                                                                 \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, \
        int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width,                           \
        const int64_t* max_indices);                                                                                                                 \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE, false>(                                                                 \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, \
        int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width,                           \
        const int64_t* max_indices);                                                                                                                 \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::ADD, TYPE, false>(                                                                 \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, \
        int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width,                           \
        const int64_t* max_indices);                                                                                                                 \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::AVERAGE, TYPE, false>(                                                             \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, \
        int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width,                           \
        const int64_t* max_indices);                                                                                                                 \
                                                                                                                                                     \
    /* --- NEW KERNEL INSTANTIATION --- */                                                                                                           \
    template __global__ void morpho_operation_backward_kernel_indices<OPERATION, MORPHO_MERGE::MAX, TYPE>(                                 \
        const int64_t* max_indexes, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output,                                                    \
        int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height, int kernel_width);                              \
    template __global__ void morpho_operation_backward_kernel_indices<OPERATION, MORPHO_MERGE::MIN, TYPE>(                                 \
        const int64_t* max_indexes, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output,                                                    \
        int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height, int kernel_width);                              \
                                                                                                                                                     \
    /* --- OLD KERNEL INSTANTIATIONS (UNTOUCHED) --- */                                                                                              \
    template __global__ void morpho_operation_backward_kernel_grad_input<OPERATION, TYPE>(                                                           \
        const int64_t* max_indexes, TYPE* grad_input, const TYPE* grad_output, int batch_size, int in_channels, int out_channels,                    \
        int height, int width, int kernel_height, int kernel_width);                                                                                 \
    template __global__ void morpho_operation_backward_kernel_grad_weight<OPERATION, MORPHO_MERGE::MAX, TYPE>(                                       \
        const int64_t* max_indexes, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, int out_channels,                   \
        int height, int width, int kernel_height, int kernel_width);                                                                                 \
    template __global__ void morpho_operation_backward_kernel_grad_weight<OPERATION, MORPHO_MERGE::MIN, TYPE>(                                       \
        const int64_t* max_indexes, TYPE* grad_weight, const TYPE* grad_output, int batch_size, int in_channels, int out_channels,                   \
        int height, int width, int kernel_height, int kernel_width);                                                                                 \
                                                                                                                                                     \
    /* --- RE-COMPUTATION KERNEL INSTANTIATIONS (UNTOUCHED) --- */                                                                                   \
    template void morpho_operation_backward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE>(                                                              \
        const TYPE* input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output,                                         \
        int batch_size, int in_channels, int out_channels, int height, int width,                                                                    \
        int padded_height, int padded_width, int kernel_height, int kernel_width, dim3 blocks, dim3 threads);                                        \
    template void morpho_operation_backward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE>(                                                              \
        const TYPE* input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output,                                         \
        int batch_size, int in_channels, int out_channels, int height, int width,                                                                    \
        int padded_height, int padded_width, int kernel_height, int kernel_width, dim3 blocks, dim3 threads);                                        \
    template void morpho_operation_backward_kernel<OPERATION, MORPHO_MERGE::ADD, TYPE>(                                                              \
        const TYPE* input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output,                                         \
        int batch_size, int in_channels, int out_channels, int height, int width,                                                                    \
        int padded_height, int padded_width, int kernel_height, int kernel_width, dim3 blocks, dim3 threads);                                        \
    template void morpho_operation_backward_kernel<OPERATION, MORPHO_MERGE::AVERAGE, TYPE>(                                                          \
        const TYPE* input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output,                                         \
        int batch_size, int in_channels, int out_channels, int height, int width,                                                                    \
        int padded_height, int padded_width, int kernel_height, int kernel_width, dim3 blocks, dim3 threads);

INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::DILATION, float)
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::DILATION, double)
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::EROSION, float)
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::EROSION, double)