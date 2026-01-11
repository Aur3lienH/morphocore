#pragma once
#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "morpho_op_policy.cuh"

/// @brief GPU kernel for morphological operations with IDENTITY channel merge.
/// Each output channel processes only its corresponding input channel.
/// Constraint: in_channels == out_channels
/// Input is pre-padded.
/// Weight shape: [channels, 1, kernel_height, kernel_width]
/// @tparam Op Morphological operation type (DILATION or EROSION).
/// @tparam scalar_t Data type of tensor elements.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position.
template <MORPHO_OPERATION Op, typename scalar_t, bool save_index>
__global__ void morpho_operation_forward_identity_kernel(
    const scalar_t* __restrict__ padded_input,
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
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= height || w >= width)
        return;

    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;
    const int padded_height = height + (kernel_height / 2) * 2;
    const int padded_width = width + (kernel_width / 2) * 2;

    constexpr scalar_t INITIAL_VAL = Policy::initial_value();

    // Process each channel independently (IDENTITY merge)
    for (int c = 0; c < channels; c++)
    {
        // Strides for padded input
        const int padded_batch_stride = channels * padded_height * padded_width;
        const int padded_channel_stride = padded_height * padded_width;
        const int weight_channel_stride = kernel_height * kernel_width;

        // Pointers to current channel
        const scalar_t* input_channel = padded_input + b * padded_batch_stride + c * padded_channel_stride;
        const scalar_t* weight_channel = weight + c * weight_channel_stride;

        scalar_t best_value = INITIAL_VAL;
        int64_t best_k = 0;

        // Loop over kernel
        #pragma unroll
        for (int kh = 0; kh < kernel_height; kh++)
        {
            #pragma unroll
            for (int kw = 0; kw < kernel_width; kw++)
            {
                // Calculate position in padded input
                int padded_h = h - kh + 2 * kernel_height_half;
                int padded_w = w - kw + 2 * kernel_width_half;

                scalar_t input_val = input_channel[padded_h * padded_width + padded_w];
                scalar_t weight_val = weight_channel[kh * kernel_width + kw];
                scalar_t val = Policy::combine(input_val, weight_val);

                if (Policy::is_better(val, best_value))
                {
                    best_value = val;
                    if constexpr (save_index)
                    {
                        best_k = kh * kernel_width + kw;
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
}