#pragma once
#include <torch/torch.h>
#include "morpho_merge_policy.hpp"

/// @brief Forward CPU pass for smorph with IDENTITY merge (no channel merging).
/// Each output channel processes only its corresponding input channel.
/// Constraint: in_channels == out_channels
/// Weight shape: [channels, 1, kernel_height, kernel_width]
/// Alpha shape: [channels]
/// Output shape: [batch_size, channels, height, width]
template <typename T>
auto smorph_forward_cpu_identity(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha)
{
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
    TORCH_CHECK(weight_shape[1] == 1, "Identity mode requires weight.shape[1] == 1, got ", weight_shape[1]);
    TORCH_CHECK(alpha.size(0) == channels, "Alpha must have shape [channels]");

    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto alpha_contig = alpha.contiguous();

    // Output: [batch, channels, height, width]
    torch::Tensor output = torch::empty({batch_size, channels, height, width}, input_contig.options());

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ alpha_ptr = alpha_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();

    const int64_t input_batch_stride = channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_channel_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = channels * height * width;
    const int64_t output_channel_stride = height * width;

#pragma omp parallel for schedule(guided) collapse(3)
    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t c = 0; c < channels; c++)
        {
            for (int64_t h = 0; h < height; h++)
            {
                const T* input_channel = input_ptr + b * input_batch_stride + c * input_channel_stride;
                const T* weight_channel = weight_ptr + c * weight_channel_stride;
                T* output_channel = output_ptr + b * output_batch_stride + c * output_channel_stride;

                const T alpha_val = alpha_ptr[c];
                const T abs_alpha = std::abs(alpha_val);
                const bool is_dilation_like = alpha_val >= 0;

                for (int64_t w = 0; w < width; w++)
                {
                    // Find reference value (max for dilation, min for erosion)
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
                                const T val = input_val + weight_channel[k];

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

                    // Compute softmax weights and weighted sum
                    T sum = 0;
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
                                const T val = input_val + weight_channel[k];

                                const T exp_term =
                                    is_dilation_like ? std::exp((val - reference_value) * abs_alpha) : std::exp((reference_value - val) * abs_alpha);
                                sum += exp_term;
                            }
                        }
                    }

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
                                const T val = input_val + weight_channel[k];

                                const T exp_term =
                                    is_dilation_like ? std::exp((val - reference_value) * abs_alpha) : std::exp((reference_value - val) * abs_alpha);

                                const T soft_weight = exp_term / sum;
                                weighted_result += soft_weight * val;
                            }
                        }
                    }

                    output_channel[h * width + w] = weighted_result;
                }
            }
        }
    }

    return output;
}