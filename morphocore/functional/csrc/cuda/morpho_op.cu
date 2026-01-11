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
#include "morpho_op_kernel_forward_identity.cuh"
#include "morpho_op_kernel_backward_identity.cuh"
#include <iostream>

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t, bool save_indices>
void morpho_forward_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* output, int64_t* max_indexes, int batch_size, int in_channels,
                           int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batch_size);

    size_t shared_memory_size = 0;
    
    if constexpr (merge == MORPHO_MERGE::IDENTITY)
    {
        // IDENTITY only supports save_indices=false
        static_assert(!save_indices, "IDENTITY merge does not support index saving");
        
        morpho_operation_forward_identity_kernel<Op, scalar_t, false><<<blocks, threads, shared_memory_size>>>(
            input, weight, output, max_indexes, batch_size, in_channels, height, width, kernel_height, kernel_width);
    }
    else
    {
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
}

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t, bool save_indices>
void morpho_backward_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                            const scalar_t* grad_output, int batch_size, int in_channels, int out_channels, int height, int width, int padded_height,
                            int padded_width, int kernel_height, int kernel_width, const int64_t* max_indexes)
{
    // Common thread block size
    dim3 threads(16, 16);

    if constexpr (merge == MORPHO_MERGE::IDENTITY)
    {
        // IDENTITY only supports save_indices=false
        static_assert(!save_indices, "IDENTITY merge does not support index saving");
        
        dim3 blocks((width + threads.x - 1) / threads.x, 
                    (height + threads.y - 1) / threads.y, 
                    batch_size);
        
        morpho_operation_backward_identity_kernel<Op, scalar_t><<<blocks, threads>>>(
            padded_input, weight, grad_input, grad_weight, grad_output,
            batch_size, in_channels, height, width, kernel_height, kernel_width);
    }
    else if constexpr (save_indices)
    {
        // --- Combined, Optimized Kernel (Scatter-Scatter) ---
        // Launch grid is (width, height, batch_size * out_channels)
        dim3 blocks_optimized((width + threads.x - 1) / threads.x,
                              (height + threads.y - 1) / threads.y,
                              batch_size * out_channels);

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

// MorphoPolicyGPU instantiations
template struct MorphoPolicyGPU<float, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicyGPU<float, MORPHO_OPERATION::EROSION>;
template struct MorphoPolicyGPU<double, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicyGPU<double, MORPHO_OPERATION::EROSION>;

// ============================================================================
// TEMPLATE INSTANTIATIONS - ONLY VALID COMBINATIONS
// ============================================================================

// Helper macros
#define INSTANTIATE_KERNELS(OP, MERGE, SCALAR_T, SAVE_INDICES) \
    template void morpho_forward_kernel<MORPHO_OPERATION::OP, MORPHO_MERGE::MERGE, SCALAR_T, SAVE_INDICES>( \
        const SCALAR_T* input, const SCALAR_T* weight, SCALAR_T* output, int64_t* max_indexes, \
        int batch_size, int in_channels, int out_channels, int height, int width, \
        int padded_height, int padded_width, int kernel_height, int kernel_width); \
    \
    template void morpho_backward_kernel<MORPHO_OPERATION::OP, MORPHO_MERGE::MERGE, SCALAR_T, SAVE_INDICES>( \
        const SCALAR_T* padded_input, const SCALAR_T* weight, SCALAR_T* grad_input, SCALAR_T* grad_weight, \
        const SCALAR_T* grad_output, int batch_size, int in_channels, int out_channels, int height, int width, \
        int padded_height, int padded_width, int kernel_height, int kernel_width, const int64_t* max_indexes);

// MAX and MIN: support both save_indices=true and save_indices=false
#define INSTANTIATE_MAX_MIN(OP, SCALAR_T) \
    INSTANTIATE_KERNELS(OP, MAX, SCALAR_T, true) \
    INSTANTIATE_KERNELS(OP, MAX, SCALAR_T, false) \
    INSTANTIATE_KERNELS(OP, MIN, SCALAR_T, true) \
    INSTANTIATE_KERNELS(OP, MIN, SCALAR_T, false)

// ADD, AVERAGE, and IDENTITY: only support save_indices=false
#define INSTANTIATE_ADD_AVERAGE_IDENTITY(OP, SCALAR_T) \
    INSTANTIATE_KERNELS(OP, ADD, SCALAR_T, false) \
    INSTANTIATE_KERNELS(OP, AVERAGE, SCALAR_T, false) \
    INSTANTIATE_KERNELS(OP, IDENTITY, SCALAR_T, false)

// Complete instantiation for one operation and one scalar type
#define INSTANTIATE_ALL_FOR_OP_TYPE(OP, SCALAR_T) \
    INSTANTIATE_MAX_MIN(OP, SCALAR_T) \
    INSTANTIATE_ADD_AVERAGE_IDENTITY(OP, SCALAR_T)

// Instantiate for both operations and both scalar types
INSTANTIATE_ALL_FOR_OP_TYPE(DILATION, float)
INSTANTIATE_ALL_FOR_OP_TYPE(DILATION, double)
INSTANTIATE_ALL_FOR_OP_TYPE(EROSION, float)
INSTANTIATE_ALL_FOR_OP_TYPE(EROSION, double)

// Total: 2 ops × 2 types × (4 MAX/MIN + 3 ADD/AVERAGE/IDENTITY) × 2 kernels = 56 instantiations