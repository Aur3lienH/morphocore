#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include "morpho_merge_policy.cuh"

/// @brief CUDA kernel for SmorphSCM forward pass (Soft Channel Merge)
/// Performs soft morphology per channel, then soft merge across channels using beta
template <typename scalar_t>
__global__ void smorph_scm_forward_kernel(
    const scalar_t* input,
    const scalar_t* weights,
    const scalar_t* alpha,
    const scalar_t* beta,
    scalar_t* output,
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
    // [channel_results, channel_exp_terms, vals, soft_weights]
    extern __shared__ char shared_mem[];
    
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    
    scalar_t* thread_mem = reinterpret_cast<scalar_t*>(shared_mem) + 
                          thread_id * (2 * input_channels + 2 * kernel_size);
    
    scalar_t* channel_results = thread_mem;
    scalar_t* channel_exp_terms = thread_mem + input_channels;
    scalar_t* vals = thread_mem + 2 * input_channels;
    scalar_t* soft_weights = vals + kernel_size;

    for (int co = 0; co < output_channels; ++co)
    {
        const scalar_t beta_val = beta[co];
        const scalar_t beta_abs = abs(beta_val);
        const bool beta_positive = beta_val >= static_cast<scalar_t>(0);

        const int weight_out_offset = co * input_channels * kernel_height * kernel_width;
        const int alpha_out_offset = co * input_channels;

        // Step 1: Compute soft morphology for each input channel
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

            // Pass 1a: Compute vals and find reference_value
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

            // Pass 1b: Compute exponentials and sum
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

            // Pass 1c: Normalize and compute weighted result
            scalar_t weighted_result = static_cast<scalar_t>(0);
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
                        const scalar_t soft_weight = soft_weights[k] / sum;
                        weighted_result += soft_weight * vals[k];
                    }
                }
            }

            channel_results[ci] = weighted_result;
        }

        // Step 2: Find reference value across channels for beta soft merge
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

        // Step 3: Compute beta soft weights
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

        // Step 4: Compute final weighted output
        scalar_t final_value = static_cast<scalar_t>(0);
        
        for (int ci = 0; ci < input_channels; ++ci)
        {
            const scalar_t beta_soft_weight = (beta_sum > std::numeric_limits<scalar_t>::epsilon()) ? 
                (channel_exp_terms[ci] / beta_sum) : static_cast<scalar_t>(0);
            
            final_value += channel_results[ci] * beta_soft_weight;
        }

        // Write output
        const int out_idx = b * output_channels * height * width + 
                           co * height * width + 
                           h * width + w;
        output[out_idx] = final_value;
    }
}

/// @brief Host function to launch SmorphSCM forward kernel
template <typename scalar_t>
void morpho_smorph_scm_forward_kernel(
    const scalar_t* input, const scalar_t* weight, const scalar_t* alpha, 
    const scalar_t* beta, scalar_t* output,
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
    
    // Memory per thread: channel_results + channel_exp_terms + vals + soft_weights
    const size_t shared_mem_size = threads_per_block * 
        (2 * input_channels + 2 * kernel_size) * sizeof(scalar_t);

    smorph_scm_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
        input, weight, alpha, beta, output,
        batch_size, input_channels, output_channels, height, width,
        kernel_height, kernel_width, kernel_height_half, kernel_width_half
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error in smorph_scm_forward: %s\n", cudaGetErrorString(err));
    }
}