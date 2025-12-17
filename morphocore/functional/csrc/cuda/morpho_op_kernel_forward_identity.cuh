#pragma once
#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "morpho_op_policy.cuh"

/// @brief GPU kernel for morphological operations with IDENTITY channel merge.
/// Each output channel processes only its corresponding input channel.
/// Constraint: in_channels == out_channels
/// Weight shape: [channels, 1, kernel_height, kernel_width]
/// @tparam Op Morphological operation type (DILATION or EROSION).
/// @tparam Padding Padding mode (GEODESIC or REPLICATE).
/// @tparam scalar_t Data type of tensor elements.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position.
template <MORPHO_OPERATION Op, MORPHO_PADDING Padding, typename scalar_t, bool save_index>
__global__ void morpho_operation_forward_identity_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width)
{
    using Policy = MorphoPolicyGPU<scalar_t, Op>;

    // Thread indices
    const int b = blockIdx.z;
    const int c = blockIdx.y;
    const int h = blockIdx.x * blockDim.y + threadIdx.y;
    const int w = threadIdx.x;

    if (b >= batch_size || c >= channels || h >= height || w >= width)
        return;

    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    constexpr scalar_t PADDING_VAL = Policy::padding_value();
    constexpr scalar_t INITIAL_VAL = Policy::initial_value();

    // Strides
    const int input_batch_stride = channels * height * width;
    const int input_channel_stride = height * width;
    const int weight_channel_stride = kernel_height * kernel_width;

    // Pointers to current channel
    const scalar_t* input_channel = input + b * input_batch_stride + c * input_channel_stride;
    const scalar_t* weight_channel = weight + c * weight_channel_stride;

    scalar_t best_value = INITIAL_VAL;
    int64_t best_k = 0;
    int64_t k = 0;

    // Loop over kernel
    #pragma unroll
    for (int kh = 0; kh < kernel_height; kh++)
    {
        const int ph = h + kernel_height_half - kh;
        const bool h_valid = (ph >= 0 && ph < height);

        #pragma unroll
        for (int kw = 0; kw < kernel_width; kw++, k++)
        {
            const int pw = w + kernel_width_half - kw;
            const bool w_valid = (pw >= 0 && pw < width);

            scalar_t input_val;

            if constexpr (Padding == MORPHO_PADDING::GEODESIC)
            {
                input_val = (h_valid && w_valid) ? input_channel[ph * width + pw] : PADDING_VAL;
            }
            else if constexpr (Padding == MORPHO_PADDING::REPLICATE)
            {
                const int ph_rep = min(max(ph, 0), height - 1);
                const int pw_rep = min(max(pw, 0), width - 1);
                input_val = input_channel[ph_rep * width + pw_rep];
            }

            const scalar_t val = Policy::combine(input_val, weight_channel[k]);

            if (Policy::is_better(val, best_value))
            {
                best_value = val;
                if constexpr (save_index)
                {
                    best_k = k;
                }
            }
        }
    }

    // Write output
    const int out_idx = b * (channels * height * width) + c * (height * width) + h * width + w;
    output[out_idx] = best_value;

    if constexpr (save_index)
    {
        max_indexes[out_idx] = best_k;
    }
}