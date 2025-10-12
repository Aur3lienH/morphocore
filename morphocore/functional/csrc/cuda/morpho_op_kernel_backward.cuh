#pragma once
#include "morpho_op.cuh"
#include "morpho_op_policy.cuh"
#include "morpho_merge_policy.cuh"
#include "../cpp/morpho_op_enum.hpp"

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
__global__ void morpho_operation_backward_sum_avg_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                                                         const scalar_t* grad_output, int batch_size, int in_channels, int out_channels, int height,
                                                         int width, int padded_height, int padded_width, int kernel_height, int kernel_width)
{
    using Policy = MorphoPolicyGPU<scalar_t, Op>;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= height || w >= width || b >= batch_size) return;

    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    for (int o = 0; o < out_channels; o++)
    {
        const int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
        const scalar_t grad_val = grad_output[out_idx];

        if (grad_val == static_cast<scalar_t>(0)) continue;

        const scalar_t scaled_grad = (merge == MORPHO_MERGE::AVERAGE) ? grad_val / static_cast<scalar_t>(in_channels) : grad_val;

        for (int c = 0; c < in_channels; c++)
        {
            const scalar_t* input_channel = input + b * in_channels * padded_height * padded_width + c * padded_height * padded_width;
            const scalar_t* weight_base = weight + o * in_channels * kernel_height * kernel_width + c * kernel_height * kernel_width;

            scalar_t best_value = Policy::initial_value();
            int best_kh = -1;
            int best_kw = -1;

            for (int kh = 0; kh < kernel_height; kh++)
            {
                for (int kw = 0; kw < kernel_width; kw++)
                {
                    const int ph = h + kernel_height_half - kh;
                    const int pw = w + kernel_width_half - kw;

                    const int padded_ph = ph + kernel_height_half;
                    const int padded_pw = pw + kernel_width_half;

                    const bool padded_h_valid = (padded_ph >= 0 && padded_ph < padded_height);
                    const bool padded_w_valid = (padded_pw >= 0 && padded_pw < padded_width);

                    const scalar_t input_val =
                        (padded_h_valid && padded_w_valid) ? input_channel[padded_ph * padded_width + padded_pw] : Policy::padding_value();

                    const scalar_t val = Policy::combine(input_val, weight_base[kh * kernel_width + kw]);

                    if (Policy::is_better(val, best_value))
                    {
                        best_value = val;
                        best_kh = kh;
                        best_kw = kw;
                    }
                }
            }

            if (best_kh != -1)
            {
                const int ih = h + kernel_height_half - best_kh;
                const int iw = w + kernel_width_half - best_kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                {
                    const int grad_input_idx = b * in_channels * height * width + c * height * width + ih * width + iw;
                    atomicAdd(&grad_input[grad_input_idx], scaled_grad);
                }

                const int grad_weight_idx =
                    o * in_channels * kernel_height * kernel_width + c * kernel_height * kernel_width + best_kh * kernel_width + best_kw;

                if constexpr (Op == MORPHO_OPERATION::DILATION)
                {
                    atomicAdd(&grad_weight[grad_weight_idx], scaled_grad);
                }
                else
                {
                    atomicAdd(&grad_weight[grad_weight_idx], -scaled_grad);
                }
            }
        }
    }
}

/// @brief CUDA kernel for backward pass with min/max merge policies
/// Finds the single best value across all channels and kernel positions
template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
__global__ void morpho_operation_backward_min_max_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                                                         const scalar_t* grad_output, int batch_size, int in_channels, int out_channels, int height,
                                                         int width, int padded_height, int padded_width, int kernel_height, int kernel_width)
{
    using Policy = MorphoPolicyGPU<scalar_t, Op>;
    using MergePolicy = MorphoMergePolicyGPU<scalar_t, merge>;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= height || w >= width || b >= batch_size) return;

    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    for (int o = 0; o < out_channels; o++)
    {
        const int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
        const scalar_t grad_val = grad_output[out_idx];

        if (grad_val == static_cast<scalar_t>(0)) continue;

        scalar_t best_merge_value = MergePolicy::get_init_value();
        int best_kh = -1;
        int best_kw = -1;
        int best_channel = -1;

        for (int c = 0; c < in_channels; c++)
        {
            const scalar_t* input_channel = input + b * in_channels * padded_height * padded_width + c * padded_height * padded_width;
            const scalar_t* weight_base = weight + o * in_channels * kernel_height * kernel_width + c * kernel_height * kernel_width;

            scalar_t best_channel_value = Policy::initial_value();
            int channel_best_kh = -1;
            int channel_best_kw = -1;

            for (int kh = 0; kh < kernel_height; kh++)
            {
                for (int kw = 0; kw < kernel_width; kw++)
                {
                    const int ph = h + kernel_height_half - kh;
                    const int pw = w + kernel_width_half - kw;

                    const int padded_ph = ph + kernel_height_half;
                    const int padded_pw = pw + kernel_width_half;

                    const bool padded_h_valid = (padded_ph >= 0 && padded_ph < padded_height);
                    const bool padded_w_valid = (padded_pw >= 0 && padded_pw < padded_width);

                    const scalar_t input_val = (padded_h_valid && padded_w_valid) ? input_channel[padded_ph * padded_width + padded_pw] : Policy::padding_value();

                    const scalar_t val = Policy::combine(input_val, weight_base[kh * kernel_width + kw]);

                    if (Policy::is_better(val, best_channel_value))
                    {
                        best_channel_value = val;
                        channel_best_kh = kh;
                        channel_best_kw = kw;
                    }
                }
            }

            if constexpr (merge == MORPHO_MERGE::MAX)
            {
                if (best_channel_value > best_merge_value)
                {
                    best_merge_value = best_channel_value;
                    best_kh = channel_best_kh;
                    best_kw = channel_best_kw;
                    best_channel = c;
                }
            }
            else if constexpr (merge == MORPHO_MERGE::MIN)
            {
                if (best_channel_value < best_merge_value)
                {
                    best_merge_value = best_channel_value;
                    best_kh = channel_best_kh;
                    best_kw = channel_best_kw;
                    best_channel = c;
                }
            }
        }

        const int ih = h + kernel_height_half - best_kh;
        const int iw = w + kernel_width_half - best_kw;

        if (ih >= 0 && ih < height && iw >= 0 && iw < width)
        {
            const int grad_input_idx = b * in_channels * height * width + best_channel * height * width + ih * width + iw;
            atomicAdd(&grad_input[grad_input_idx], grad_val);
        }

        const int grad_weight_idx =
            o * in_channels * kernel_height * kernel_width + best_channel * kernel_height * kernel_width + best_kh * kernel_width + best_kw;

        if constexpr (Op == MORPHO_OPERATION::DILATION)
        {
            atomicAdd(&grad_weight[grad_weight_idx], grad_val);
        }
        else
        {
            atomicAdd(&grad_weight[grad_weight_idx], -grad_val);
        }
    }
}

/// @brief Wrapper function to dispatch the appropriate kernel based on merge policy
template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
void morpho_operation_backward_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                                      const scalar_t* grad_output, int batch_size, int in_channels, int out_channels, int height, int width,
                                      int padded_height, int padded_width, int kernel_height, int kernel_width, dim3 blocks, dim3 threads)
{
    if constexpr (merge == MORPHO_MERGE::ADD || merge == MORPHO_MERGE::AVERAGE)
    {
        morpho_operation_backward_sum_avg_kernel<Op, merge, scalar_t><<<blocks, threads>>>(input, weight, grad_input, grad_weight, grad_output,
                                                                                           batch_size, in_channels, out_channels, height, width,
                                                                                           padded_height, padded_width, kernel_height, kernel_width);
    }
    else
    {
        morpho_operation_backward_min_max_kernel<Op, merge, scalar_t><<<blocks, threads>>>(input, weight, grad_input, grad_weight, grad_output,
                                                                                           batch_size, in_channels, out_channels, height, width,
                                                                                           padded_height, padded_width, kernel_height, kernel_width);
    }
}
