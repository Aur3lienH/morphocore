#pragma once
#include <torch/torch.h>
#include <omp.h>

#include "morpho_op.hpp"
#include "morpho_op_enum.hpp"
#include "morpho_merge_policy.hpp"
#include "morpho_op_policy.hpp"

/// @brief Forward pass for morphological operations with IDENTITY channel merge.
/// Each output channel processes only its corresponding input channel.
/// Constraint: in_channels == out_channels
/// Weight shape: [channels, 1, kernel_height, kernel_width]
/// @tparam op Morphological operation type (DILATION or EROSION).
/// @tparam T Data type of tensor elements.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position.
/// @param input Input tensor of shape [batch, channels, height, width].
/// @param weight Morphological kernel tensor of shape [channels, 1, kernel_height, kernel_width].
/// @return Output tensor, or tuple (output, indices) if save_index is true.
template <MORPHO_OPERATION op, MORPHO_PADDING padding, typename T, bool save_index>
auto morpho_operation_forward_identity_cpu(torch::Tensor input, torch::Tensor weight)
{
    using Policy = MorphoPolicyCPU<T, op>;

    const auto input_shape = input.sizes();
    const auto weight_shape = weight.sizes();

    const int64_t batch_size = input_shape[0];
    const int64_t channels = input_shape[1];  // in_channels == out_channels
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];

    // Verify constraint
    TORCH_CHECK(weight_shape[0] == channels, "Identity mode requires weight.shape[0] == input.shape[1]");
    TORCH_CHECK(weight_shape[1] == 1, "Identity mode requires weight.shape[1] == 1");

    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    torch::Tensor output = torch::empty({batch_size, channels, height, width}, input_contig.options());

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();

    // Strides
    const int64_t input_batch_stride = channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_channel_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = channels * height * width;
    const int64_t output_channel_stride = height * width;

    torch::Tensor max_indices;
    int64_t* __restrict__ indices_ptr = nullptr;

    if constexpr (save_index)
    {
        max_indices = torch::empty({batch_size, channels, height, width}, torch::kLong);
        indices_ptr = max_indices.template data_ptr<int64_t>();
    }

    constexpr T PADDING_VAL = Policy::padding_value();
    constexpr T INITIAL_VAL = Policy::initial_value();

    constexpr int64_t TILE_H = 16;
    constexpr int64_t TILE_W = 16;

#pragma omp parallel for schedule(guided) collapse(3)
    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t c = 0; c < channels; c++)
        {
            for (int64_t h_tile = 0; h_tile < height; h_tile += TILE_H)
            {
                const int64_t h_end = std::min(h_tile + TILE_H, height);

                const T* input_channel = input_ptr + b * input_batch_stride + c * input_channel_stride;
                const T* weight_channel = weight_ptr + c * weight_channel_stride;
                T* output_channel = output_ptr + b * output_batch_stride + c * output_channel_stride;

                int64_t* indices_channel = nullptr;
                if constexpr (save_index)
                {
                    indices_channel = indices_ptr + b * (channels * height * width) + c * (height * width);
                }

                for (int64_t w_tile = 0; w_tile < width; w_tile += TILE_W)
                {
                    const int64_t w_end = std::min(w_tile + TILE_W, width);

                    for (int64_t h = h_tile; h < h_end; h++)
                    {
                        for (int64_t w = w_tile; w < w_end; w++)
                        {
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

                                    T input_val;
                                    if constexpr (padding == MORPHO_PADDING::GEODESIC)
                                    {
                                        input_val = (h_valid && w_valid) ? input_channel[ph * width + pw] : PADDING_VAL;
                                    }
                                    else if constexpr (padding == MORPHO_PADDING::REPLICATE)
                                    {
                                        const int64_t ph_rep = std::min(std::max(ph, int64_t(0)), height - 1);
                                        const int64_t pw_rep = std::min(std::max(pw, int64_t(0)), width - 1);
                                        input_val = input_channel[ph_rep * width + pw_rep];
                                    }
                                    else
                                    {
                                        throw std::runtime_error("morpho_op_cpu_forward: Unsupported padding mode.");
                                    }
                                    const T val = Policy::combine(input_val, weight_channel[k]);

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

                            output_channel[h * width + w] = best_value;

                            if constexpr (save_index)
                            {
                                indices_channel[h * width + w] = best_k;
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