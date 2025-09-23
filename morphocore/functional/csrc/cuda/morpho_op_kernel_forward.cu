#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "morpho_op_policy.cuh"
#include "morpho_merge_policy.cuh"

// template <MORPHO_OPERATION Op, MorphoMergePolicyGPU merge, typename scalar_t, int KH, int KW>
// __global__ void morpho_operation_forward_specialized_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* output, int*
// max_indexes,
//                                                             int batch_size, int in_channels, int out_channels, int height, int width)
// {

//     extern __shared__ char shared_mem[];

//     using MergePolicy = MorphoMergePolicyGPU<scalar_t, merge>;

//     const int b = blockIdx.z;
//     const int h = blockIdx.y * blockDim.y + threadIdx.y;
//     const int w = blockIdx.x * blockDim.x + threadIdx.x;
//     const int padded_height = height + (KH / 2) * 2;
//     const int padded_width = width + (KW / 2) * 2;
//     const int kernel_height_half = KH / 2;
//     const int kernel_width_half = KW / 2;

//     using Policy = MorphoPolicy<scalar_t, Op>;

//     if (h < height && w < width && b < batch_size)
//     {

//         for (int o = 0; o < out_channels; o++)
//         {

//             scalar_t output_value = MergePolicy::get_init_value();

//             for (int c = 0; c < in_channels; c++)
//             {

//                 scalar_t best_value = Policy::initial_value();
//                 int max_flat_index = -1;

// #pragma unroll
//                 for (int kh = 0; kh < KH; kh++)
//                 {
// #pragma unroll
//                     for (int kw = 0; kw < KW; kw++)
//                     {
//                         int padded_h = h - kh + 2 * kernel_height_half;
//                         int padded_w = w - kw + 2 * kernel_width_half;

//                         scalar_t input_val = padded_input[b * in_channels * padded_height * padded_width + c * padded_height * padded_width
//                                                           + padded_h * padded_width + padded_w];

//                         scalar_t weight_val = weight[o * in_channels * KH * KW + c * KH * KW + kh * KW + kw];

//                         scalar_t val = Policy::combine(input_val, weight_val);

//                         int flat_index = kh * KW + kw;

//                         if (Policy::is_better(val, best_value))
//                         {
//                             best_value = val;
//                             max_flat_index = flat_index;
//                         }
//                     }
//                 }

//                 output_value = MergePolicy::merge(output_value, best_value);

//                 int max_idx_index =
//                     b * out_channels * in_channels * height * width + o * in_channels * height * width + c * height * width + h * width + w;
//                 max_indexes[max_idx_index] = max_flat_index;
//             }

//             int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
//             output[out_idx] = MergePolicy::finalize(output_value, in_channels);
//         }
//     }
// }

template <MORPHO_OPERATION Op, MorphoMergePolicyGPU merge, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_forward_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* output, int* max_indexes,
                                                int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height,
                                                int kernel_width)
{

    extern __shared__ char shared_mem[];

    using MergePolicy = MorphoMergePolicyGPU<scalar_t, merge>;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int padded_height = height + (kernel_height / 2) * 2;
    const int padded_width = width + (kernel_width / 2) * 2;
    const int kernel_height_half = kernel_height / 2;
    const int kernel_width_half = kernel_width / 2;

    using Policy = MorphoPolicy<scalar_t, Op>;

    if (h < height && w < width && b < batch_size)
    {

        for (int o = 0; o < out_channels; o++)
        {

            scalar_t output_value = MergePolicy::get_init_value();

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

                output_value = MergePolicy::merge(output_value, best_value);

                int max_idx_index =
                    b * out_channels * in_channels * height * width + o * in_channels * height * width + c * height * width + h * width + w;
                max_indexes[max_idx_index] = max_flat_index;
            }

            int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
            output[out_idx] = MergePolicy::finalize(output_value, in_channels);
        }
    }
}
