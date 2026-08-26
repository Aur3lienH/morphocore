#include "smorph_kernels.cuh"
#include "smorph_forward.cuh"
#include "smorph_backward.cuh"
#include "morpho_merge_policy.cuh"

#define INSTANTIATE_SMORPH_DEVICE_KERNELS(MERGE_TYPE, DTYPE) \
template __global__ void smorph_forward_kernel<MERGE_TYPE, DTYPE>( \
    const DTYPE* input, \
    const DTYPE* weight, \
    const DTYPE* alpha, \
    DTYPE* output, \
    DTYPE* references_indexes, \
    DTYPE* sum_values, \
    int batch_size, \
    int input_channel, \
    int output_channel, \
    int height, \
    int width, \
    int kernel_height, \
    int kernel_width, \
    int kernel_height_half, \
    int kernel_width_half \
); \
\
template __global__ void smorph_backward_kernel<MERGE_TYPE, DTYPE>( \
    const DTYPE* grad_output, \
    DTYPE* grad_input, \
    DTYPE* grad_weight, \
    DTYPE* grad_alpha, \
    const DTYPE* alpha, \
    const DTYPE* input, \
    const DTYPE* weights, \
    int batch_size, \
    int input_channels, \
    int output_channels, \
    int height, \
    int width, \
    int kernel_height, \
    int kernel_width, \
    int kernel_height_half, \
    int kernel_width_half \
);

INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::ADD, float)
INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::AVERAGE, float)
INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::MIN, float)
INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::MAX, float)

INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::ADD, double)
INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::AVERAGE, double)
INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::MIN, double)
INSTANTIATE_SMORPH_DEVICE_KERNELS(MORPHO_MERGE::MAX, double)