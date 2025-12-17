#pragma once
#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "morpho_op_policy.cuh"

/// @brief GPU kernel for backward pass with IDENTITY channel merge.
/// Each channel's gradient is computed independently.
/// Constraint: in_channels == out_channels
/// @tparam Op Morphological operation type (DILATION or EROSION).
/// @tparam Padding Padding mode (GEODESIC or REPLICATE).
/// @tparam scalar_t Data type of tensor elements.
template <MORPHO_OPERATION Op, MORPHO_PADDING Padding, typename scalar_t>
__global__ void morpho_operation_backward_identity_kernel(
    const scalar_t* __restrict__ input,
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
    const int grad_output_batch_stride = channels * height * width;
    const int grad_output_channel_stride = height * width;

    // Pointers to current channel
    const scalar_t* input_channel = input + b * input_batch_stride + c * input_channel_stride;
    const scalar_t* weight_channel = weight + c * weight_channel_stride;
    const scalar_t* grad_output_channel = grad_output + b * grad_output_batch_stride + c * grad_output_channel_stride;

    const scalar_t grad_val = grad_output_channel[h * width + w];
    
    // Early exit if gradient is zero
    if (grad_val == static_cast<scalar_t>(0))
        return;

    // Find the best kernel position for this output location
    scalar_t best_value = INITIAL_VAL;
    int best_kh = -1;
    int best_kw = -1;

    #pragma unroll
    for (int kh = 0; kh < kernel_height; kh++)
    {
        const int ph = h + kernel_height_half - kh;
        const bool h_valid = (ph >= 0 && ph < height);

        #pragma unroll
        for (int kw = 0; kw < kernel_width; kw++)
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

            const scalar_t val = Policy::combine(input_val, weight_channel[kh * kernel_width + kw]);

            if (Policy::is_better(val, best_value))
            {
                best_value = val;
                best_kh = kh;
                best_kw = kw;
            }
        }
    }

    // Backprop to input
    if (best_kh != -1)
    {
        const int ih = h + kernel_height_half - best_kh;
        const int iw = w + kernel_width_half - best_kw;

        if (ih >= 0 && ih < height && iw >= 0 && iw < width)
        {
            const int grad_input_idx = b * input_batch_stride + c * input_channel_stride + ih * width + iw;
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