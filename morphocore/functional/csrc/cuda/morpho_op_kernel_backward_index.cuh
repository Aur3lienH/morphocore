#pragma once

#include "morpho_op.cuh"
#include "morpho_op_policy.cuh"
#include "morpho_merge_policy.cuh"

template <MORPHO_OPERATION Op, typename scalar_t>
__global__ void morpho_operation_backward_kernel_grad_input(
    const int64_t* max_indexes,
    scalar_t* grad_input,
    const scalar_t* grad_output,
    int batch_size, int in_channels, int out_channels, int height,
    int width, int kernel_height, int kernel_width)
{
    // Launch grid is (width, height, batch_size * in_channels)
    const int c = blockIdx.z % in_channels;
    const int b = blockIdx.z / in_channels;
    const int ih = blockIdx.y * blockDim.y + threadIdx.y;
    const int iw = blockIdx.x * blockDim.x + threadIdx.x;

    if (ih >= height || iw >= width || b >= batch_size) return;

    scalar_t sum = static_cast<scalar_t>(0);
    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    for (int o = 0; o < out_channels; o++)
    {
        #pragma unroll
        for (int kh = 0; kh < kernel_height; kh++)
        {
            #pragma unroll
            for (int kw = 0; kw < kernel_width; kw++)
            {
                const int h = ih - kh + kernel_height_half;
                const int w = iw - kw + kernel_width_half;

                if (h >= 0 && h < height && w >= 0 && w < width)
                {
                    const size_t out_idx = static_cast<size_t>(b) * out_channels * height * width
                                         + static_cast<size_t>(o) * height * width
                                         + static_cast<size_t>(h) * width + w;

                    const int64_t packed_index = max_indexes[out_idx];
                    const int winning_channel = (packed_index >> 16) & 0xFFFF;
                    const int flat_kernel_index = packed_index & 0xFFFF;

                    const int best_kh = flat_kernel_index / kernel_width;
                    const int best_kw = flat_kernel_index % kernel_width;

                    if (winning_channel == c && best_kh == kh && best_kw == kw)
                    {
                        sum += grad_output[out_idx];
                    }
                }
            }
        }
    }

    const size_t grad_input_idx = static_cast<size_t>(b) * in_channels * height * width
                                + static_cast<size_t>(c) * height * width
                                + static_cast<size_t>(ih) * width + iw;

    if constexpr (Op == MORPHO_OPERATION::DILATION)
    {
        grad_input[grad_input_idx] = sum;
    }
    else // EROSION
    {
        grad_input[grad_input_idx] = -sum;
    }
}


template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
__global__ void morpho_operation_backward_kernel_grad_weight(
    const int64_t* max_indexes,
    scalar_t* grad_weight,
    const scalar_t* grad_output,
    int batch_size, int in_channels, int out_channels, int height,
    int width, int kernel_height, int kernel_width)
{
    static_assert(merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN, "Saved indices only work with MAX or MIN merge");

    // Launch grid is (width, height, batch_size)
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= height || w >= width || b >= batch_size) return;

    for (int o = 0; o < out_channels; o++)
    {
        const int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
        const scalar_t grad_val = grad_output[out_idx];

        //if (grad_val == static_cast<scalar_t>(0)) continue;

        const int64_t packed_index = max_indexes[out_idx];
        const int winning_channel = (packed_index >> 16) & 0xFFFF;
        const int flat_kernel_index = packed_index & 0xFFFF;

        const int best_kh = flat_kernel_index / kernel_width;
        const int best_kw = flat_kernel_index % kernel_width;

        const int grad_weight_idx =
            o * in_channels * kernel_height * kernel_width + winning_channel * kernel_height * kernel_width + best_kh * kernel_width + best_kw;

        if constexpr (Op == MORPHO_OPERATION::DILATION)
        {
            atomicAdd(&grad_weight[grad_weight_idx], grad_val);
        }
        else // EROSION
        {
            atomicAdd(&grad_weight[grad_weight_idx], -grad_val);
        }
    }
}


template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
__global__ void morpho_operation_backward_kernel_indices(
    const int64_t* __restrict__ max_indexes,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_weight,
    const scalar_t* __restrict__ grad_output,
    int batch_size, int in_channels, int out_channels, int height,
    int width, int kernel_height, int kernel_width)
{
    static_assert(merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN, "Saved indices only work with MAX or MIN merge");


    const int o = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= height || w >= width || b >= batch_size) return;
    

    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    const size_t out_idx = static_cast<size_t>(b) * out_channels * height * width
                         + static_cast<size_t>(o) * height * width
                         + static_cast<size_t>(h) * width + w;

    const scalar_t grad_val = grad_output[out_idx];

    const int64_t packed_index = max_indexes[out_idx];

    const int winning_channel = (packed_index >> 16) & 0xFFFF;
    const int flat_kernel_index = packed_index & 0xFFFF;
    const int best_kh = flat_kernel_index / kernel_width;
    const int best_kw = flat_kernel_index % kernel_width;

    const int ih = h - best_kh + kernel_height_half;
    const int iw = w - best_kw + kernel_width_half;

    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
    {
        const size_t grad_input_idx = static_cast<size_t>(b) * in_channels * height * width
                                    + static_cast<size_t>(winning_channel) * height * width
                                    + static_cast<size_t>(ih) * width + iw;
        
        atomicAdd(&grad_input[grad_input_idx], grad_val);
    }

    const size_t grad_weight_idx =
          static_cast<size_t>(o) * in_channels * kernel_height * kernel_width
        + static_cast<size_t>(winning_channel) * kernel_height * kernel_width
        + static_cast<size_t>(best_kh) * kernel_width + best_kw;

    atomicAdd(&grad_weight[grad_weight_idx], (Op == MORPHO_OPERATION::DILATION) ? grad_val : -grad_val);
}