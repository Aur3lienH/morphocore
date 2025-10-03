#include "morpho_op_kernels.cuh"
#include "morpho_op_kernel_backward.cuh"
#include "morpho_op_kernel_forward.cuh"
#include "morpho_op.cuh"
#include "../cpp/morpho_op_enum.hpp"

#define INSTANTIATE_MORPHO_KERNELS(OP, MERGE, SCALAR_T) \
    template __global__ void morpho_operation_forward_kernel<MORPHO_OPERATION::OP, MORPHO_MERGE::MERGE, SCALAR_T>( \
        const SCALAR_T* padded_input, const SCALAR_T* weight, SCALAR_T* output, int64_t* max_indexes, \
        int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height, int kernel_width); \
    \
    template __global__ void morpho_operation_backward_sum_avg_kernel<MORPHO_OPERATION::OP, MORPHO_MERGE::MERGE, SCALAR_T>( \
        const SCALAR_T* input, const SCALAR_T* weight, SCALAR_T* grad_input, SCALAR_T* grad_weight, \
        const SCALAR_T* grad_output, int batch_size, int in_channels, int out_channels, int height, \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width); \
    \
    template __global__ void morpho_operation_backward_min_max_kernel<MORPHO_OPERATION::OP, MORPHO_MERGE::MERGE, SCALAR_T>( \
        const SCALAR_T* input, const SCALAR_T* weight, SCALAR_T* grad_input, SCALAR_T* grad_weight, \
        const SCALAR_T* grad_output, int batch_size, int in_channels, int out_channels, int height, \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width); \
    \
    template void morpho_operation_backward_kernel<MORPHO_OPERATION::OP, MORPHO_MERGE::MERGE, SCALAR_T>( \
        const SCALAR_T* input, const SCALAR_T* weight, SCALAR_T* grad_input, SCALAR_T* grad_weight, \
        const SCALAR_T* grad_output, int batch_size, int in_channels, int out_channels, int height, int width, \
        int padded_height, int padded_width, int kernel_height, int kernel_width, dim3 blocks, dim3 threads);

#define INSTANTIATE_MORPHO_ALL_MERGE(OP, SCALAR_T) \
    INSTANTIATE_MORPHO_KERNELS(OP, ADD, SCALAR_T) \
    INSTANTIATE_MORPHO_KERNELS(OP, AVERAGE, SCALAR_T) \
    INSTANTIATE_MORPHO_KERNELS(OP, MIN, SCALAR_T) \
    INSTANTIATE_MORPHO_KERNELS(OP, MAX, SCALAR_T)

#define INSTANTIATE_MORPHO_ALL_OPS(SCALAR_T) \
    INSTANTIATE_MORPHO_ALL_MERGE(DILATION, SCALAR_T) \
    INSTANTIATE_MORPHO_ALL_MERGE(EROSION, SCALAR_T)

#define INSTANTIATE_MORPHO_ALL() \
    INSTANTIATE_MORPHO_ALL_OPS(float) \
    INSTANTIATE_MORPHO_ALL_OPS(double)

INSTANTIATE_MORPHO_ALL()