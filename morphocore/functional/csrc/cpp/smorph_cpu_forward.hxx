#pragma once
#include <torch/torch.h>
#include "morpho_merge_policy.hpp"

/// @brief Forward CPU pass for the smorph operation (Smooth approximation of erosion or dilation).
/// @tparam T Data type (float or double).
/// @tparam merge Merge channel policy.
/// @param input Input tensor.
/// @param weight Weight tensor.
/// @param alpha Control parameter tensor of shape [out_channels, in_channels] for choosing between dilation (alpha -> +inf) and erosion (alpha ->
/// -inf) for each operation.
/// @return Output tensor.
template <MORPHO_MERGE merge, typename T>
auto smorph_forward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha)
{
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
    auto alpha_contig = alpha.contiguous();

    torch::Tensor output = torch::empty({batch_size, out_channels, height, width}, input_contig.options());

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ alpha_ptr = alpha_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();

    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;
    const int64_t alpha_out_stride = in_channels;

    constexpr int64_t TILE_H = 16;
    constexpr int64_t TILE_W = 16;

#pragma omp parallel
    {
#pragma omp for schedule(guided) collapse(3)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                for (int64_t h_tile = 0; h_tile < height; h_tile += TILE_H)
                {
                    const int64_t h_end = std::min(h_tile + TILE_H, height);

                    const T* input_batch = input_ptr + b * input_batch_stride;
                    const T* weight_out = weight_ptr + o * weight_out_stride;
                    const T* alpha_out = alpha_ptr + o * alpha_out_stride;
                    T* output_base = output_ptr + b * output_batch_stride + o * output_channel_stride;

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
                                    const T alpha_val = alpha_out[i];
                                    const T abs_alpha = std::abs(alpha_val);
                                    const bool is_dilation_like = alpha_val >= 0;

                                    const T* input_channel = input_batch + i * input_channel_stride;
                                    const T* weight_in = weight_out + i * weight_in_stride;

                                    T reference_value = is_dilation_like ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();

                                    int64_t k = 0;

                                    for (int64_t kh = 0; kh < kernel_height; kh++)
                                    {
                                        const int64_t ph = h + kernel_height_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);

                                        for (int64_t kw = 0; kw < kernel_width; kw++, k++)
                                        {
                                            const int64_t pw = w + kernel_width_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);

                                            if (h_valid && w_valid)
                                            {
                                                const T input_val = input_channel[ph * width + pw];
                                                const T val = input_val + weight_in[k];

                                                if (is_dilation_like)
                                                {
                                                    reference_value = std::max(val, reference_value);
                                                }
                                                else
                                                {
                                                    reference_value = std::min(val, reference_value);
                                                }
                                            }
                                        }
                                    }

                                    T sum = 0;
                                    T weighted_result = 0;
                                    k = 0;

                                    for (int64_t kh = 0; kh < kernel_height; kh++)
                                    {
                                        const int64_t ph = h + kernel_height_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);

                                        for (int64_t kw = 0; kw < kernel_width; kw++, k++)
                                        {
                                            const int64_t pw = w + kernel_width_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);

                                            if (h_valid && w_valid)
                                            {
                                                const T input_val = input_channel[ph * width + pw];
                                                const T val = input_val + weight_in[k];

                                                const T exp_term = is_dilation_like ? std::exp((val - reference_value) * abs_alpha)
                                                                                    : std::exp((reference_value - val) * abs_alpha);
                                                sum += exp_term;
                                            }
                                        }
                                    }

                                    k = 0;
                                    for (int64_t kh = 0; kh < kernel_height; kh++)
                                    {
                                        const int64_t ph = h + kernel_height_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);

                                        for (int64_t kw = 0; kw < kernel_width; kw++, k++)
                                        {
                                            const int64_t pw = w + kernel_width_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);

                                            if (h_valid && w_valid)
                                            {
                                                const T input_val = input_channel[ph * width + pw];
                                                const T val = input_val + weight_in[k];

                                                const T exp_term = is_dilation_like ? std::exp((val - reference_value) * abs_alpha)
                                                                                    : std::exp((reference_value - val) * abs_alpha);

                                                const T soft_weight = exp_term / sum;

                                                weighted_result += soft_weight * val;
                                            }
                                        }
                                    }

                                    channel_out_val = MorphoMergePolicy<T, merge>::merge(channel_out_val, weighted_result);
                                }

                                output_base[h * width + w] = MorphoMergePolicy<T, merge>::finalize(channel_out_val, in_channels);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}