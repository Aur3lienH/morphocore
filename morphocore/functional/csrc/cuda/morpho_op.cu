#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "../cpp/morpho_merge_policy.hpp"
#include "morpho_merge_policy.cuh"
#include "morpho_op_policy.cuh"
#include "morpho_op_kernels.cuh"

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
void morpho_forward_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* output, int64_t* max_indexes, int batch_size, int in_channels,
                           int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height, int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batch_size);

    size_t shared_memory_size = 0;

    // if (kernel_height == 3 && kernel_width == 3)
    // {
    //     morpho_operation_forward_kernel<Op, float, 3, 3>
    //         <<<blocks, threads, shared_memory_size>>>(input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
    //     return;
    // }
    // else if (kernel_height == 5 && kernel_width == 5)
    // {
    //     morpho_operation_forward_kernel<Op, float, 5, 5>
    //         <<<blocks, threads, shared_memory_size>>>(input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
    //     return;
    // }
    // else if (kernel_height == 7 && kernel_width == 7)
    // {
    //     morpho_operation_forward_kernel<Op, float, 7, 7>
    //         <<<blocks, threads, shared_memory_size>>>(input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
    //     return;
    // }
    morpho_operation_forward_kernel<Op, merge, scalar_t, -1, -1><<<blocks, threads, shared_memory_size>>>(
        input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width, kernel_height, kernel_width);
}

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
void morpho_backward_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                            const scalar_t* grad_output, const int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,
                            int width, int padded_height, int padded_width, int kernel_height, int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batch_size);

    size_t shared_memory_size = 0;

    // if (kernel_height == 3 && kernel_width == 3)
    // {
    //     morpho_operation_backward_kernel<Op, float, 3, 3><<<blocks, threads, shared_memory_size>>>(
    //         padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, batch_size, in_channels, out_channels, height, width);
    //     return;
    // }
    // else if (kernel_height == 5 && kernel_width == 5)
    // {
    //     morpho_operation_backward_kernel<Op, float, 5, 5><<<blocks, threads, shared_memory_size>>>(
    //         padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, batch_size, in_channels, out_channels, height, width);
    //     return;
    // }
    // else if (kernel_height == 7 && kernel_width == 7)
    // {
    //     morpho_operation_backward_kernel<Op, float, 7, 7><<<blocks, threads, shared_memory_size>>>(
    //         padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, batch_size, in_channels, out_channels, height, width);
    //     return;
    // }
    morpho_operation_backward_kernel<Op, merge, scalar_t, -1, -1>
        <<<blocks, threads, shared_memory_size>>>(padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, batch_size, in_channels,
                                                  out_channels, height, width, kernel_height, kernel_width);
}

#define INSTANTIATE_MORPHO_OP(T, H, W, OP)                                                                                                           \
    template __global__ void morpho_operation_forward_kernel<MORPHO_OPERATION::OP, T, H, W>(const T*, const T*, T*, int64_t*, int, int, int, int,    \
                                                                                            int);                                                    \
    template __global__ void morpho_operation_backward_kernel<MORPHO_OPERATION::OP, T, H, W>(const T*, const T*, T*, T*, const T*, const int64_t*,   \
                                                                                             int, int, int, int, int);

template struct MorphoPolicy<float, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicy<float, MORPHO_OPERATION::EROSION>;
template struct MorphoPolicy<double, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicy<double, MORPHO_OPERATION::EROSION>;

// INSTANTIATE_MORPHO_OP(float, 3, 3, DILATION)
// INSTANTIATE_MORPHO_OP(float, 3, 3, EROSION)
// INSTANTIATE_MORPHO_OP(float, 5, 5, DILATION)
// INSTANTIATE_MORPHO_OP(float, 5, 5, EROSION)
// INSTANTIATE_MORPHO_OP(float, 7, 7, DILATION)
// INSTANTIATE_MORPHO_OP(float, 7, 7, EROSION)
// INSTANTIATE_MORPHO_OP(double, 3, 3, DILATION)
// INSTANTIATE_MORPHO_OP(double, 3, 3, EROSION)
// INSTANTIATE_MORPHO_OP(double, 5, 5, DILATION)
// INSTANTIATE_MORPHO_OP(double, 5, 5, EROSION)
// INSTANTIATE_MORPHO_OP(double, 7, 7, DILATION)
// INSTANTIATE_MORPHO_OP(double, 7, 7, EROSION)

// Template instantiation for dilation !
#define INSTANTIATE_MORPHO_FORWARD_KERNEL(OPERATION, TYPE)                                                                                           \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE>(                                                                         \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE>(                                                                         \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::ADD, TYPE>(                                                                         \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);                                                        \
    template void morpho_forward_kernel<OPERATION, MORPHO_MERGE::AVERAGE, TYPE>(                                                                     \
        const TYPE* input, const TYPE* weight, TYPE* output, int64_t* max_indexes, int batch_size, int in_channels, int out_channels, int height,    \
        int width, int padded_height, int padded_width, int kernel_height, int kernel_width);

INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::DILATION, float)
INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::DILATION, double)
INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::EROSION, float)
INSTANTIATE_MORPHO_FORWARD_KERNEL(MORPHO_OPERATION::EROSION, double)

#define INSTANTIATE_MORPHO_BACKWARD_KERNEL(OPERATION, TYPE)                                                                                          \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::MAX, TYPE>(                                                                        \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, const int64_t* max_indexes,      \
        int batch_size, int in_channels, int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height,            \
        int kernel_width);                                                                                                                           \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::MIN, TYPE>(                                                                        \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, const int64_t* max_indexes,      \
        int batch_size, int in_channels, int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height,            \
        int kernel_width);                                                                                                                           \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::ADD, TYPE>(                                                                        \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, const int64_t* max_indexes,      \
        int batch_size, int in_channels, int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height,            \
        int kernel_width);                                                                                                                           \
    template void morpho_backward_kernel<OPERATION, MORPHO_MERGE::AVERAGE, TYPE>(                                                                    \
        const TYPE* padded_input, const TYPE* weight, TYPE* grad_input, TYPE* grad_weight, const TYPE* grad_output, const int64_t* max_indexes,      \
        int batch_size, int in_channels, int out_channels, int height, int width, int padded_height, int padded_width, int kernel_height,            \
        int kernel_width);
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::DILATION, float)
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::DILATION, double)
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::EROSION, float)
INSTANTIATE_MORPHO_BACKWARD_KERNEL(MORPHO_OPERATION::EROSION, double)