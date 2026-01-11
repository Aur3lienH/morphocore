#pragma once
#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "morpho_op_policy.cuh"

/// @brief GPU kernel for backward pass with IDENTITY channel merge.
/// Each channel's gradient is computed independently.
/// Input is pre-padded.
/// Constraint: in_channels == out_channels
/// @tparam Op Morphological operation type (DILATION or EROSION).
/// @tparam scalar_t Data type of tensor elements.
template <MORPHO_OPERATION Op, typename scalar_t>
__global__ void morpho_operation_backward_identity_kernel(
    const scalar_t* __restrict__ padded_input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_weight,
    const scalar_t* __restrict__ grad_output,
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
        const int grad_output_batch_stride = channels * height * width;
        const int grad_output_channel_stride = height * width;

        // Pointers to current channel
        const scalar_t* input_channel = padded_input + b * padded_batch_stride + c * padded_channel_stride;
        const scalar_t* weight_channel = weight + c * weight_channel_stride;
        const scalar_t* grad_output_channel = grad_output + b * grad_output_batch_stride + c * grad_output_channel_stride;

        const scalar_t grad_val = grad_output_channel[h * width + w];
        
        // Early exit if gradient is zero
        if (grad_val == static_cast<scalar_t>(0))
            continue;

        // Find the best kernel position for this output location
        scalar_t best_value = INITIAL_VAL;
        int best_kh = -1;
        int best_kw = -1;

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
                    best_kh = kh;
                    best_kw = kw;
                }
            }
        }

        // Backprop to input (unpadded coordinates)
        if (best_kh != -1)
        {
            const int ih = h + kernel_height_half - best_kh;
            const int iw = w + kernel_width_half - best_kw;

            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
            {
                const int grad_input_idx = b * (channels * height * width) + c * (height * width) + ih * width + iw;
                atomicAdd(&grad_input[grad_input_idx], grad_val);
            }

            // Backprop to weight
            const int grad_weight_idx = c * weight_channel_stride + best_kh * kernel_width + best_kw;

            if constexpr (Op == MORPHO_OPERATION::DILATION)
            {
                atomicAdd(&grad_weight[grad_weight_idx], grad_val);
            }
            else  // EROSION
            {
                atomicAdd(&grad_weight[grad_weight_idx], -grad_val);
            }
        }
    }
}