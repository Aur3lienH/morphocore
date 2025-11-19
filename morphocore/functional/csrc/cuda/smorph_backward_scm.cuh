#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include "morpho_merge_policy.cuh"
#include "atomic_ops.cuh"

/// @brief CUDA kernel for SmorphSCM backward pass
/// Computes gradients through both alpha (spatial) and beta (channel) soft merging
template <typename scalar_t>
__global__ void smorph_scm_backward_kernel(
    const scalar_t* grad_output,
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    scalar_t* grad_beta,
    const scalar_t* alpha,
    const scalar_t* beta,
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
    const int input_batch_offset = b * input_channels * height * width;

    // Dynamic shared memory layout per thread:
    // [channel_results, channel_ref_values, channel_exp_terms, beta_soft_weights, 
    //  vals[input_channels][kernel_size], soft_weights[input_channels][kernel_size]]
    extern __shared__ char shared_mem[];
    
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    
    scalar_t* thread_mem = reinterpret_cast<scalar_t*>(shared_mem) + 
                          thread_id * (4 * input_channels + 2 * input_channels * kernel_size);
    
    scalar_t* channel_results = thread_mem;
    scalar_t* channel_ref_values = thread_mem + input_channels;
    scalar_t* channel_exp_terms = thread_mem + 2 * input_channels;
    scalar_t* beta_soft_weights = thread_mem + 3 * input_channels;
    scalar_t* all_vals = thread_mem + 4 * input_channels;
    scalar_t* all_soft_weights = all_vals + input_channels * kernel_size;

    for (int co = 0; co < output_channels; ++co)
    {
        const int grad_out_idx = b * output_channels * height * width + 
                                 co * height * width + 
                                 h * width + w;
        const scalar_t grad_out = grad_output[grad_out_idx];
        
        if (grad_out == static_cast<scalar_t>(0))
            continue;

        const scalar_t beta_val = beta[co];
        const scalar_t beta_abs = abs(beta_val);
        const bool beta_positive = beta_val >= static_cast<scalar_t>(0);

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

            scalar_t* vals = all_vals + ci * kernel_size;
            scalar_t* soft_weights = all_soft_weights + ci * kernel_size;

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

            channel_ref_values[ci] = reference_value;

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

            scalar_t weighted_result = static_cast<scalar_t>(0);
            for (k = 0; k < kernel_size; ++k)
            {
                soft_weights[k] /= sum;
                weighted_result += soft_weights[k] * vals[k];
            }

            channel_results[ci] = weighted_result;
        }

        scalar_t channel_out_ref = beta_positive ? 
            -std::numeric_limits<scalar_t>::infinity() : 
             std::numeric_limits<scalar_t>::infinity();
        
        for (int ci = 0; ci < input_channels; ++ci)
        {
            if (beta_positive)
            {
                channel_out_ref = max(channel_out_ref, channel_results[ci]);
            }
            else
            {
                channel_out_ref = min(channel_out_ref, channel_results[ci]);
            }
        }

        scalar_t beta_sum = static_cast<scalar_t>(0);
        
        for (int ci = 0; ci < input_channels; ++ci)
        {
            const scalar_t val = channel_results[ci];
            const scalar_t exp_term = beta_positive ? 
                exp(beta_abs * (val - channel_out_ref)) : 
                exp(beta_abs * (channel_out_ref - val));
            
            channel_exp_terms[ci] = exp_term;
            beta_sum += exp_term;
        }

        for (int ci = 0; ci < input_channels; ++ci)
        {
            beta_soft_weights[ci] = channel_exp_terms[ci] / beta_sum;
        }

        scalar_t final_value = static_cast<scalar_t>(0);
        for (int ci = 0; ci < input_channels; ++ci)
        {
            final_value += channel_results[ci] * beta_soft_weights[ci];
        }

        scalar_t beta_grad_accum = static_cast<scalar_t>(0);
        
        for (int ci = 0; ci < input_channels; ++ci)
        {
            const scalar_t p_ci = beta_soft_weights[ci];
            const scalar_t val_ci = channel_results[ci];
            const scalar_t sign_factor = beta_positive ? static_cast<scalar_t>(1.0) : static_cast<scalar_t>(-1.0);
            
            beta_grad_accum += p_ci * sign_factor * (val_ci - channel_out_ref) * (val_ci - final_value);
        }
        
        atomicAdd(&grad_beta[co], beta_grad_accum * grad_out);

        for (int ci = 0; ci < input_channels; ++ci)
        {
            const scalar_t p_ci = beta_soft_weights[ci];
            const scalar_t val_ci = channel_results[ci];
            
            const scalar_t grad_channel_result = p_ci * (static_cast<scalar_t>(1.0) + beta_val * (val_ci - final_value)) * grad_out;

            if (grad_channel_result == static_cast<scalar_t>(0))
                continue;

            const int alpha_index = alpha_out_offset + ci;
            const scalar_t alpha_val = alpha[alpha_index];
            const bool is_dilation_like = alpha_val >= static_cast<scalar_t>(0);
            const scalar_t reference_value = channel_ref_values[ci];

            const int input_channel_offset = input_batch_offset + ci * height * width;
            const int weight_in_offset = weight_out_offset + ci * kernel_height * kernel_width;

            scalar_t* vals = all_vals + ci * kernel_size;
            scalar_t* soft_weights = all_soft_weights + ci * kernel_size;
            const scalar_t smorph_output = channel_results[ci];

            scalar_t alpha_grad_contrib = static_cast<scalar_t>(0);

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
                        const scalar_t final_grad = grad_val * grad_channel_result;
                        
                        const int input_idx = input_channel_offset + ph * width + pw;
                        atomicAdd(&grad_input[input_idx], final_grad);

                        const int weight_idx = weight_in_offset + k;
                        atomicAdd(&grad_weight[weight_idx], final_grad);

                        const scalar_t sign_factor = is_dilation_like ? static_cast<scalar_t>(1.0) : static_cast<scalar_t>(-1.0);
                        alpha_grad_contrib += p * sign_factor * (val - reference_value) * (val - smorph_output);
                    }
                }
            }

            atomicAdd(&grad_alpha[alpha_index], alpha_grad_contrib * grad_channel_result);
        }
    }
}

template <typename scalar_t>
void morpho_smorph_scm_backward_kernel(
    const scalar_t* grad_output,
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    scalar_t* grad_beta,
    const scalar_t* alpha,
    const scalar_t* beta,
    const scalar_t* input,
    const scalar_t* weights,
    int batch_size, int input_channels, int output_channels,
    int height, int width, int kernel_height, int kernel_width,
    int kernel_height_half, int kernel_width_half)
{
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (width + TILE_WIDTH - 1) / TILE_WIDTH, 
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT, 
        batch_size
    );

    const int kernel_size = kernel_height * kernel_width;
    const int threads_per_block = TILE_WIDTH * TILE_HEIGHT;
    
    const size_t shared_mem_size = threads_per_block * 
        (4 * input_channels + 2 * input_channels * kernel_size) * sizeof(scalar_t);

    smorph_scm_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
        grad_output, grad_input, grad_weight, grad_alpha, grad_beta,
        alpha, beta, input, weights,
        batch_size, input_channels, output_channels, height, width,
        kernel_height, kernel_width, kernel_height_half, kernel_width_half
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error in smorph_scm_backward: %s\n", cudaGetErrorString(err));
    }
}