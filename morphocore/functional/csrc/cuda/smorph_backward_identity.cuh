#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include "morpho_merge_policy.cuh"
#include "atomic_ops.cuh"

/// @brief CUDA kernel for backward pass of soft morphological operations with IDENTITY merge.
/// Each channel's gradient is computed independently.
/// Constraint: in_channels == out_channels
template <typename scalar_t>
__global__ void smorph_backward_identity_kernel(
    const scalar_t* grad_output,
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    const scalar_t* alpha,
    const scalar_t* input,
    const scalar_t* weights,
    int batch_size, int channels,
    int height, int width, int kernel_height, int kernel_width,
    int kernel_height_half, int kernel_width_half)
{
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= height || w >= width)
        return;

    const int input_batch_offset = b * channels * height * width;

    scalar_t vals[121];
    scalar_t soft_weights[121];

    // Process each channel independently (IDENTITY: c_out == c_in)
    for (int c = 0; c < channels; ++c)
    {
        const int grad_out_idx = b * channels * height * width + 
                                 c * height * width + 
                                 h * width + w;
        const scalar_t grad_out = grad_output[grad_out_idx];
        
        if (grad_out == static_cast<scalar_t>(0))
            continue;

        const scalar_t alpha_val = alpha[c];
        const scalar_t abs_alpha = abs(alpha_val);
        const bool is_dilation_like = alpha_val >= static_cast<scalar_t>(0);
        
        const int input_channel_offset = input_batch_offset + c * height * width;
        const int weight_offset = c * kernel_height * kernel_width;

        scalar_t reference_value = is_dilation_like ? 
            -std::numeric_limits<scalar_t>::infinity() : 
             std::numeric_limits<scalar_t>::infinity();

        // First pass: compute values and find reference
        int k = 0;
        int num_valid = 0;
        
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            const int ph = h + kernel_height_half - kh;
            const bool h_valid = (ph >= 0 && ph < height);

            for (int kw = 0; kw < kernel_width; ++kw, ++k)
            {
                const int pw = w + kernel_width_half - kw;
                const bool w_valid = (pw >= 0 && pw < width);

                if (h_valid && w_valid)
                {
                    const scalar_t input_val = input[input_channel_offset + ph * width + pw];
                    const scalar_t weight_val = weights[weight_offset + k];
                    vals[k] = input_val + weight_val;
                    num_valid++;

                    reference_value = is_dilation_like ? 
                        max(vals[k], reference_value) : 
                        min(vals[k], reference_value);
                }
                else
                {
                    vals[k] = static_cast<scalar_t>(0);
                }
            }
        }

        if (num_valid == 0)
            continue;

        // Second pass: compute soft weights
        scalar_t sum = static_cast<scalar_t>(0);
        k = 0;
        
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            const int ph = h + kernel_height_half - kh;
            const bool h_valid = (ph >= 0 && ph < height);

            for (int kw = 0; kw < kernel_width; ++kw, ++k)
            {
                const int pw = w + kernel_width_half - kw;
                const bool w_valid = (pw >= 0 && pw < width);

                if (h_valid && w_valid)
                {
                    const scalar_t exp_term = is_dilation_like ? 
                        exp((vals[k] - reference_value) * abs_alpha) : 
                        exp((reference_value - vals[k]) * abs_alpha);
                    soft_weights[k] = exp_term;
                    sum += exp_term;
                }
                else
                {
                    soft_weights[k] = static_cast<scalar_t>(0);
                }
            }
        }

        if (sum <= static_cast<scalar_t>(1e-20))
            continue;

        // Third pass: compute smorph output
        scalar_t smorph_output = static_cast<scalar_t>(0);
        k = 0;
        
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            const int ph = h + kernel_height_half - kh;
            const bool h_valid = (ph >= 0 && ph < height);

            for (int kw = 0; kw < kernel_width; ++kw, ++k)
            {
                const int pw = w + kernel_width_half - kw;
                const bool w_valid = (pw >= 0 && pw < width);

                if (h_valid && w_valid)
                {
                    soft_weights[k] /= sum;
                    smorph_output += soft_weights[k] * vals[k];
                }
            }
        }

        // Fourth pass: compute gradients
        k = 0;
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            const int ph = h + kernel_height_half - kh;
            const bool h_valid = (ph >= 0 && ph < height);

            for (int kw = 0; kw < kernel_width; ++kw, ++k)
            {
                const int pw = w + kernel_width_half - kw;
                const bool w_valid = (pw >= 0 && pw < width);

                if (h_valid && w_valid)
                {
                    const scalar_t p = soft_weights[k];
                    const scalar_t val = vals[k];

                    const scalar_t grad_val = p * (static_cast<scalar_t>(1.0) + alpha_val * (val - smorph_output));
                    const scalar_t final_grad = grad_val * grad_out;
                    
                    atomicAdd(&grad_input[input_channel_offset + ph * width + pw], final_grad);
                    atomicAdd(&grad_weight[weight_offset + k], final_grad);

                    const scalar_t alpha_grad_contrib = p * (val - reference_value) * (val - smorph_output) * grad_out;
                    atomicAdd(&grad_alpha[c], alpha_grad_contrib);
                }
            }
        }
    }
}