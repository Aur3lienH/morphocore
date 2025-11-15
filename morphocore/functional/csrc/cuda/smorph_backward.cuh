#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include "morpho_merge_policy.cuh"
#include "atomic_ops.cuh"

template <MORPHO_MERGE merge, typename scalar_t>
__global__ void smorph_backward_kernel(
    const scalar_t* grad_output,
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    const scalar_t* alpha,
    const scalar_t* input,
    const scalar_t* weights,
    int batch_size, int input_channels, int output_channels,
    int height, int width, int kernel_height, int kernel_width,
    int kernel_height_half, int kernel_width_half)
{
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= height || w >= width)
        return;

    const int kernel_size = kernel_height * kernel_width;
    const scalar_t scale_factor = (merge == MORPHO_MERGE::AVERAGE) ? 
        static_cast<scalar_t>(1.0) / static_cast<scalar_t>(input_channels) : 
        static_cast<scalar_t>(1.0);

    extern __shared__ char shared_mem[];
    
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int vals_offset = thread_id * kernel_size;
    const int weights_offset = vals_offset + threads_per_block * kernel_size;
    
    scalar_t* vals = reinterpret_cast<scalar_t*>(shared_mem) + vals_offset;
    scalar_t* soft_weights = reinterpret_cast<scalar_t*>(shared_mem) + weights_offset;

    const int input_batch_offset = b * input_channels * height * width;

    for (int co = 0; co < output_channels; ++co)
    {
        const int grad_out_idx = b * output_channels * height * width + 
                                 co * height * width + 
                                 h * width + w;
        const scalar_t grad_out = grad_output[grad_out_idx] * scale_factor;
        
        if (grad_out == static_cast<scalar_t>(0))
            continue;

        const int weight_out_offset = co * input_channels * kernel_height * kernel_width;
        const int alpha_out_offset = co * input_channels;

        for (int ci = 0; ci < input_channels; ++ci)
        {
            const int alpha_index = alpha_out_offset + ci;
            const scalar_t alpha_val = alpha[alpha_index];
            const scalar_t abs_alpha = abs(alpha_val);
            const bool is_dilation_like = alpha_val >= static_cast<scalar_t>(0);
            
            scalar_t reference_value = is_dilation_like ? 
                -std::numeric_limits<scalar_t>::infinity() : 
                 std::numeric_limits<scalar_t>::infinity();

            const int input_channel_offset = input_batch_offset + ci * height * width;
            const int weight_in_offset = weight_out_offset + ci * kernel_height * kernel_width;

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
                        const scalar_t weight_val = weights[weight_in_offset + k];
                        vals[k] = input_val + weight_val;
                        num_valid++;

                        if (is_dilation_like)
                        {
                            reference_value = max(vals[k], reference_value);
                        }
                        else
                        {
                            reference_value = min(vals[k], reference_value);
                        }
                    }
                    else
                    {
                        vals[k] = static_cast<scalar_t>(0);
                    }
                }
            }

            if (num_valid == 0)
                continue;

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
                        
                        const int input_idx = input_channel_offset + ph * width + pw;
                        atomicAdd(&grad_input[input_idx], final_grad);

                        const int weight_idx = weight_in_offset + k;
                        atomicAdd(&grad_weight[weight_idx], final_grad);

                        const scalar_t sign_factor = is_dilation_like ? static_cast<scalar_t>(1.0) : static_cast<scalar_t>(-1.0);
                        const scalar_t alpha_grad_contrib = p * sign_factor * (val - reference_value) * (val - smorph_output) * grad_out;
                        atomicAdd(&grad_alpha[alpha_index], alpha_grad_contrib);
                    }
                }
            }
        }
    }
}

template <MORPHO_MERGE merge, typename scalar_t>
__global__ void smorph_backward_minmax_kernel(
    const scalar_t* grad_output,
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    const scalar_t* alpha,
    const scalar_t* input,
    const scalar_t* weights,
    int batch_size, int input_channels, int output_channels,
    int height, int width, int kernel_height, int kernel_width,
    int kernel_height_half, int kernel_width_half)
{
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= height || w >= width)
        return;

    constexpr scalar_t tolerance = static_cast<scalar_t>(1e-6);
    const int kernel_size = kernel_height * kernel_width;
    const int input_batch_offset = b * input_channels * height * width;

    extern __shared__ char shared_mem[];
    
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    
    scalar_t* thread_mem = reinterpret_cast<scalar_t*>(shared_mem) + 
                          thread_id * (2 * input_channels + 2 * input_channels * kernel_size);
    
    scalar_t* channel_outputs = thread_mem;
    scalar_t* channel_reference_values = thread_mem + input_channels;
    scalar_t* channel_vals = thread_mem + 2 * input_channels;
    scalar_t* channel_soft_weights = channel_vals + input_channels * kernel_size;

    for (int co = 0; co < output_channels; ++co)
    {
        const int grad_out_idx = b * output_channels * height * width + 
                                 co * height * width + 
                                 h * width + w;
        const scalar_t grad_out = grad_output[grad_out_idx];
        
        if (grad_out == static_cast<scalar_t>(0))
            continue;

        const int weight_out_offset = co * input_channels * kernel_height * kernel_width;
        const int alpha_out_offset = co * input_channels;

        for (int ci = 0; ci < input_channels; ++ci)
        {
            const int alpha_index = alpha_out_offset + ci;
            const scalar_t alpha_val = alpha[alpha_index];
            const scalar_t abs_alpha = abs(alpha_val);
            const bool is_dilation_like = alpha_val >= static_cast<scalar_t>(0);
            
            scalar_t reference_value = is_dilation_like ? 
                -std::numeric_limits<scalar_t>::infinity() : 
                 std::numeric_limits<scalar_t>::infinity();

            const int input_channel_offset = input_batch_offset + ci * height * width;
            const int weight_in_offset = weight_out_offset + ci * kernel_height * kernel_width;

            scalar_t* vals = channel_vals + ci * kernel_size;
            scalar_t* soft_weights = channel_soft_weights + ci * kernel_size;

            int k = 0;
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
                        const scalar_t weight_val = weights[weight_in_offset + k];
                        vals[k] = input_val + weight_val;

                        if (is_dilation_like)
                        {
                            reference_value = max(vals[k], reference_value);
                        }
                        else
                        {
                            reference_value = min(vals[k], reference_value);
                        }
                    }
                    else
                    {
                        vals[k] = static_cast<scalar_t>(0);
                    }
                }
            }

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

            scalar_t smorph_output = static_cast<scalar_t>(0);
            for (k = 0; k < kernel_size; ++k)
            {
                soft_weights[k] /= sum;
                smorph_output += soft_weights[k] * vals[k];
            }

            channel_outputs[ci] = smorph_output;
            channel_reference_values[ci] = reference_value;
        }

        scalar_t target_value;
        if constexpr (merge == MORPHO_MERGE::MIN)
        {
            target_value = std::numeric_limits<scalar_t>::infinity();
            for (int ci = 0; ci < input_channels; ++ci)
            {
                target_value = min(target_value, channel_outputs[ci]);
            }
        }
        else
        {
            target_value = -std::numeric_limits<scalar_t>::infinity();
            for (int ci = 0; ci < input_channels; ++ci)
            {
                target_value = max(target_value, channel_outputs[ci]);
            }
        }

        int num_winners = 0;
        for (int ci = 0; ci < input_channels; ++ci)
        {
            if (abs(channel_outputs[ci] - target_value) < tolerance)
            {
                num_winners++;
            }
        }

        const scalar_t winner_grad = grad_out / num_winners;

        for (int ci = 0; ci < input_channels; ++ci)
        {
            if (abs(channel_outputs[ci] - target_value) < tolerance)
            {
                const int alpha_index = alpha_out_offset + ci;
                const scalar_t alpha_val = alpha[alpha_index];
                const scalar_t smorph_output = channel_outputs[ci];
                const scalar_t reference_value = channel_reference_values[ci];
                const bool is_dilation_like = alpha_val >= static_cast<scalar_t>(0);

                const int input_channel_offset = input_batch_offset + ci * height * width;
                const int weight_in_offset = weight_out_offset + ci * kernel_height * kernel_width;

                scalar_t* vals = channel_vals + ci * kernel_size;
                scalar_t* soft_weights = channel_soft_weights + ci * kernel_size;

                int k = 0;
                for (int kh = 0; kh < kernel_height; ++kh)
                {
                    const int ph = h + kernel_height_half - kh;
                    const bool h_valid = (ph >= 0 && ph < height);

                    for (int kw = 0; kw < kernel_width; ++kw, ++k)
                    {
                        const int pw = w + kernel_width_half - kw;
                        const bool w_valid = (pw >= 0 && pw < width);

                        if (h_valid && w_valid && soft_weights[k] > static_cast<scalar_t>(0))
                        {
                            const scalar_t p = soft_weights[k];
                            const scalar_t val = vals[k];

                            const scalar_t grad_val = p * (static_cast<scalar_t>(1.0) + alpha_val * (val - smorph_output));
                            const scalar_t final_grad = grad_val * winner_grad;
                            
                            const int input_idx = input_channel_offset + ph * width + pw;
                            atomicAdd(&grad_input[input_idx], final_grad);

                            const int weight_idx = weight_in_offset + k;
                            atomicAdd(&grad_weight[weight_idx], final_grad);

                            const scalar_t sign_factor = is_dilation_like ? static_cast<scalar_t>(1.0) : static_cast<scalar_t>(-1.0);
                            const scalar_t alpha_grad_contrib = p * sign_factor * (val - reference_value) * (val - smorph_output) * winner_grad;
                            atomicAdd(&grad_alpha[alpha_index], alpha_grad_contrib);
                        }
                    }
                }
            }
        }
    }
}