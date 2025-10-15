#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
#include "morpho_merge_policy.cuh"
#include "morpho_op_policy.cuh"
#include "../cpp/morpho_op_enum.hpp"
#include "atomic_ops.cuh"

/// @brief Kernel backward complet pour smorph avec merge policy ADD
/// Utilise shared memory pour optimiser les accès mémoire
template <MORPHO_MERGE merge, typename scalar_t>
__global__ void smorph_backward_kernel(const scalar_t* grad_output,  // [batch, out_channels, height, width]
                                       scalar_t* grad_input,         // [batch, in_channels, height, width]
                                       scalar_t* grad_weight,        // [out_channels, in_channels, kernel_h, kernel_w]
                                       scalar_t* grad_alpha,         // [1]
                                       const scalar_t* alpha,        // [1]
                                       const scalar_t* input,        // [batch, in_channels, height, width]
                                       const scalar_t* weights,      // [out_channels, in_channels, kernel_h, kernel_w]
                                       int batch_size, int input_channels, int output_channels, int height, int width, int kernel_height,
                                       int kernel_width, int kernel_height_half, int kernel_width_half)
{
    extern __shared__ char shared_mem[];

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= height || w >= width) return;
    
    const int kernel_size = kernel_height * kernel_width;

    const int threads_per_block = blockDim.x * blockDim.y;
    scalar_t* shared_vals = (scalar_t*)shared_mem;
    scalar_t* shared_soft_weights = (scalar_t*)&shared_vals[threads_per_block * kernel_size];

    const int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    scalar_t* my_vals = &shared_vals[tid_in_block * kernel_size];
    scalar_t* my_soft_weights = &shared_soft_weights[tid_in_block * kernel_size];

    const scalar_t alpha_val = alpha[0];
    const scalar_t abs_alpha = abs(alpha_val);
    const bool is_dilation_like = alpha_val >= 0;

    const scalar_t scale_factor = (merge == MORPHO_MERGE::AVERAGE) ? 
        static_cast<scalar_t>(1.0) / input_channels : 
        static_cast<scalar_t>(1.0);

    for (int co = 0; co < output_channels; co++)
    {
        int grad_out_idx = b * output_channels * height * width + 
                           co * height * width + h * width + w;
        scalar_t grad_out = grad_output[grad_out_idx] * scale_factor;

        if (grad_out == static_cast<scalar_t>(0)) continue;

        for (int ci = 0; ci < input_channels; ci++)
        {
            scalar_t reference_value = is_dilation_like ? 
                -std::numeric_limits<scalar_t>::infinity() : 
                 std::numeric_limits<scalar_t>::infinity();

            for (int k = 0; k < kernel_size; ++k) {
                my_vals[k] = reference_value; 
            }

            int k = 0;
            for (int kh = 0; kh < kernel_height; kh++)
            {
                for (int kw = 0; kw < kernel_width; kw++, k++)
                {
                    int ph = h + kernel_height_half - kh;
                    int pw = w + kernel_width_half - kw;
                    if (ph < 0 || ph >= height || pw < 0 || pw >= width) continue;

                    if (ph >= 0 && ph < height && pw >= 0 && pw < width)
                    {
                        int input_idx = b * input_channels * height * width + 
                                        ci * height * width + ph * width + pw;
                        scalar_t input_val = input[input_idx];

                        int weight_idx = co * input_channels * kernel_height * kernel_width + 
                                         ci * kernel_height * kernel_width + 
                                         kh * kernel_width + kw;
                        scalar_t weight_val = weights[weight_idx];

                        my_vals[k] = input_val + weight_val;

                        if (is_dilation_like)
                        {
                            reference_value = max(my_vals[k], reference_value);
                        }
                        else
                        {
                            reference_value = min(my_vals[k], reference_value);
                        }
                    }
                }
            }

            scalar_t sum = static_cast<scalar_t>(0);

            for (k = 0; k < kernel_size; k++)
            {
                bool is_valid = is_dilation_like ? (my_vals[k] > -std::numeric_limits<scalar_t>::infinity()) : (my_vals[k] < std::numeric_limits<scalar_t>::infinity());
                if (is_valid)
                {
                    scalar_t exp_term = is_dilation_like ? 
                        exp((my_vals[k] - reference_value) * abs_alpha) : 
                        exp((reference_value - my_vals[k]) * abs_alpha);
                    my_soft_weights[k] = exp_term;
                    sum += exp_term;
                }
                else
                {
                    my_soft_weights[k] = static_cast<scalar_t>(0);
                }
            }
            
            if (sum > static_cast<scalar_t>(0))
            {
                scalar_t inv_sum = static_cast<scalar_t>(1.0) / sum;
                scalar_t weighted_result = static_cast<scalar_t>(0);
                for (k = 0; k < kernel_size; k++)
                {
                    if (my_soft_weights[k] > 0) {
                        my_soft_weights[k] *= inv_sum;
                        weighted_result += my_soft_weights[k] * my_vals[k];
                    }
                }

                k = 0;
                for (int kh = 0; kh < kernel_height; kh++)
                {
                    for (int kw = 0; kw < kernel_width; kw++, k++)
                    {
                        if (my_soft_weights[k] <= 0) continue;

                        int ph = h + kernel_height_half - kh;
                        int pw = w + kernel_width_half - kw;
                        if (ph < 0 || ph >= height || pw < 0 || pw >= width) continue;

                        scalar_t p = my_soft_weights[k];
                        scalar_t val = my_vals[k];

                        const scalar_t grad_val = p + alpha_val * p * (val - weighted_result);
                        scalar_t final_grad = grad_val * grad_out;
                        
                    

                        
                        
                        int input_idx = b * input_channels * height * width + 
                                       ci * height * width + ph * width + pw;
                        atomicAdd(&grad_input[input_idx], final_grad);

                        int weight_idx = co * input_channels * kernel_height * kernel_width + 
                                        ci * kernel_height * kernel_width + 
                                        kh * kernel_width + kw;
                        atomicAdd(&grad_weight[weight_idx], final_grad);

                        scalar_t grad_alpha_term = p * (val - reference_value) * (val - weighted_result) * grad_out;
                        atomicAdd(grad_alpha, grad_alpha_term);
                    }
                }
            }
        }
    }
}