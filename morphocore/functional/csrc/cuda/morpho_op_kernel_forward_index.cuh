#pragma once
#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "morpho_op_policy.cuh"
#include "morpho_merge_policy.cuh"

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
__global__ void morpho_operation_forward_kernel_indices(const scalar_t* padded_input, const scalar_t* weight, scalar_t* output, int64_t* max_indexes,
                                                        int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height,
                                                        int kernel_width)
{
    static_assert(merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN, "Index saving only works with MAX or MIN merge");

    using MergePolicy = MorphoMergePolicyGPU<scalar_t, merge>;
    using Policy = MorphoPolicyGPU<scalar_t, Op>;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int padded_height = height + (kernel_height / 2) * 2;
    const int padded_width = width + (kernel_width / 2) * 2;
    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    if (h < height && w < width && b < batch_size)
    {
        for (int o = 0; o < out_channels; o++)
        {
            scalar_t output_value = MergePolicy::get_init_value();
            int winning_channel = -1;
            int winning_flat_index = -1;

            for (int c = 0; c < in_channels; c++)
            {
                scalar_t best_value = Policy::initial_value();
                int max_flat_index = -1;

#pragma unroll
                for (int kh = 0; kh < kernel_height; kh++)
                {
#pragma unroll
                    for (int kw = 0; kw < kernel_width; kw++)
                    {
                        int padded_h = h - kh + 2 * kernel_height_half;
                        int padded_w = w - kw + 2 * kernel_width_half;

                        scalar_t input_val = padded_input[b * in_channels * padded_height * padded_width + c * padded_height * padded_width
                                                          + padded_h * padded_width + padded_w];

                        scalar_t weight_val =
                            weight[o * in_channels * kernel_height * kernel_width + c * kernel_height * kernel_width + kh * kernel_width + kw];

                        scalar_t val = Policy::combine(input_val, weight_val);

                        int flat_index = kh * kernel_width + kw;

                        if (Policy::is_better(val, best_value))
                        {
                            best_value = val;
                            max_flat_index = flat_index;
                        }
                    }
                }

                if (c == 0 || MergePolicy::is_better(best_value, output_value))
                {
                    output_value = best_value;
                    winning_channel = c;
                    winning_flat_index = max_flat_index;
                }
            }

            int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
            output[out_idx] = MergePolicy::finalize(output_value, in_channels);

            int64_t packed_index = (static_cast<int64_t>(winning_channel) << 16) | static_cast<int64_t>(winning_flat_index);
            max_indexes[out_idx] = packed_index;
        }
    }
}