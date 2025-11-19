#pragma once
#include <memory>
#include <vector>
#include <limits>

#include <immintrin.h>
#include <omp.h>
#include <torch/torch.h>

#include "morpho_op.hpp"
#include "morpho_op_enum.hpp"
#include "morpho_merge_policy.hpp"
#include "morpho_op_policy.hpp"
#include "morpho_op_cpu_forward_linear.hxx"

// TODO: Dispatch for the kernel sizes 3x3, 5x5, 7x7 for better performance
/// @brief Dispatches to the most optimized forward morphological operation implementation based on kernel size.
///        Uses specialized templates for common kernel sizes (3x3, 5x5, 7x7) for better performance; falls back to generic implementation
///        otherwise.
/// @tparam op Morphological operation type (e.g., dilation, erosion).
/// @tparam merge Merge policy for combining results across channels.
/// @tparam T Data type of tensor elements.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position for each output element.
/// @param input Input tensor of shape [batch, channels, height, width].
/// @param weight Morphological kernel tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
/// @return Output tensor, or tuple (output, indices) if save_index is true.cc
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
auto morpho_operation_forward_cpu_dispatcher(torch::Tensor input, torch::Tensor weight)
{
    const auto weight_shape = weight.sizes();
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];

    // if (kernel_height == 3 && kernel_width == 3)
    // {
    //     return morpho_operation_forward_cpu_template<op, merge, T, save_index, 3>(input, weight);
    // }
    // if (kernel_height == 5 && kernel_width == 5)
    // {
    //     return morpho_operation_forward_cpu_template<op, merge, T, save_index, 5>(input, weight);
    // }
    // if (kernel_height == 7 && kernel_width == 7)
    // {
    //     return morpho_operation_forward_cpu_template<op, merge, T, save_index, 7>(input, weight);
    // }

    if (merge == MORPHO_MERGE::IDENTITY)
    {
        return morpho_operation_forward_identity_cpu<op, T, save_index>(input, weight);
    }

    return morpho_operation_forward_cpu<op, merge, T, save_index>(input, weight);
}

// TODO: Handle the save_index case in backward pass and with multiple channels.
/// @brief Perform the forward standard morphological operation on CPU.
/// @tparam T Data type of tensor elements.
/// @tparam op Morphological operation type (e.g., dilation, erosion).
/// @tparam merge Merge policy for combining results across channels.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position for each output element.
/// @param input Input tensor of shape [batch, channels, height, width].
/// @param weight Morphological kernel tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
/// @return Output tensor, or tuple (output, indices) if save_index is true.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
auto morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight)
{
    // Policy for choosing between erosion and dilation.
    using Policy = MorphoPolicyCPU<T, op>;

    const auto input_shape = input.sizes();
    const auto weight_shape = weight.sizes();
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];

    const int64_t batch_size = input_shape[0];
    const int64_t in_channels = input_shape[1];
    const int64_t out_channels = weight_shape[0];
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];

    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    torch::Tensor output = torch::empty({batch_size, out_channels, height, width}, input_contig.options());

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();

    // Optimized strides for contiguous tensors
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;

    torch::Tensor max_indices;
    int64_t* __restrict__ indices_ptr = nullptr;

    if constexpr (save_index)
    {
        max_indices = torch::empty({batch_size, out_channels, in_channels, height, width}, torch::kLong);
        indices_ptr = max_indices.template data_ptr<int64_t>();
    }

    constexpr T PADDING_VAL = Policy::padding_value();
    constexpr T INITIAL_VAL = Policy::initial_value();

    // Smaller tiles for better cache utilization
    constexpr int64_t TILE_H = 16;
    constexpr int64_t TILE_W = 16;

#pragma omp parallel
    {
#pragma omp for schedule(guided) collapse(2)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                for (int64_t h_tile = 0; h_tile < height; h_tile += TILE_H)
                {
                    const int64_t h_end = std::min(h_tile + TILE_H, height);

                    const T* input_batch = input_ptr + b * input_batch_stride;
                    const T* weight_out = weight_ptr + o * weight_out_stride;
                    T* output_base = output_ptr + b * output_batch_stride + o * output_channel_stride;

                    int64_t* indices_base = nullptr;
                    if constexpr (save_index)
                    {
                        indices_base = indices_ptr + b * (out_channels * in_channels * height * width) + o * (in_channels * height * width);
                    }

                    for (int64_t w_tile = 0; w_tile < width; w_tile += TILE_W)
                    {
                        const int64_t w_end = std::min(w_tile + TILE_W, width);

                        for (int64_t h = h_tile; h < h_end; h++)
                        {
                            for (int64_t w = w_tile; w < w_end; w++)
                            {
                                T channel_out_val = MorphoMergePolicy<T, merge>::get_init_value();

                                for (int64_t i = 0; i < in_channels; i++)
                                {
                                    const T* input_channel = input_batch + i * input_channel_stride;
                                    const T* weight_in = weight_out + i * weight_in_stride;

                                    T best_value = INITIAL_VAL;
                                    int64_t best_k = 0;
                                    int64_t k = 0;

                                    for (int64_t kh = 0; kh < kernel_height; kh++)
                                    {
                                        const int64_t ph = h + kernel_height_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);

                                        for (int64_t kw = 0; kw < kernel_width; kw++, k++)
                                        {
                                            const int64_t pw = w + kernel_width_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);

                                            const T input_val = (h_valid && w_valid) ? input_channel[ph * width + pw] : PADDING_VAL;

                                            const T val = Policy::combine(input_val, weight_in[k]);

                                            if (Policy::is_better(val, best_value))
                                            {
                                                best_value = val;
                                                if constexpr (save_index)
                                                {
                                                    best_k = k;
                                                }
                                            }
                                        }
                                    }

                                    channel_out_val = MorphoMergePolicy<T, merge>::merge(channel_out_val, best_value);

                                    if constexpr (save_index)
                                    {
                                        indices_base[i * height * width + h * width + w] = best_k;
                                    }
                                }

                                output_base[h * width + w] = MorphoMergePolicy<T, merge>::finalize(channel_out_val, in_channels);
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (save_index)
    {
        return std::make_tuple(output, max_indices);
    }
    else
    {
        return output;
    }
}