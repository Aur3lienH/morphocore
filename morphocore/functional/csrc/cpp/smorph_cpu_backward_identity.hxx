#pragma once
#include <torch/torch.h>
#include "morpho_merge_policy.hpp"

/// @brief Backward pass for smorph with IDENTITY merge.
/// grad_output shape: [batch_size, out_channels, in_channels, height, width]
template <typename T>
void smorph_backward_cpu_identity(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor grad_output, torch::Tensor grad_input,
                                  torch::Tensor grad_weight, torch::Tensor grad_alpha)
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
    auto grad_output_contig = grad_output.contiguous();

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ alpha_ptr = alpha_contig.template data_ptr<T>();
    const T* __restrict__ grad_output_ptr = grad_output_contig.template data_ptr<T>();

    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t grad_output_batch_stride = out_channels * in_channels * height * width;
    const int64_t grad_output_outchan_stride = in_channels * height * width;
    const int64_t grad_output_inchan_stride = height * width;
    const int64_t alpha_out_stride = in_channels;

    // Thread-local storage
    const int num_threads = omp_get_max_threads();
    std::vector<torch::Tensor> local_grad_inputs(num_threads);
    std::vector<torch::Tensor> local_grad_weights(num_threads);
    std::vector<torch::Tensor> local_grad_alphas(num_threads);

    for (int tid = 0; tid < num_threads; tid++)
    {
        local_grad_inputs[tid] = torch::zeros_like(grad_input);
        local_grad_weights[tid] = torch::zeros_like(grad_weight);
        local_grad_alphas[tid] = torch::zeros_like(grad_alpha);
    }

    grad_input.zero_();
    grad_weight.zero_();
    grad_alpha.zero_();

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        T* __restrict__ local_grad_input_ptr = local_grad_inputs[tid].template data_ptr<T>();
        T* __restrict__ local_grad_weight_ptr = local_grad_weights[tid].template data_ptr<T>();
        T* __restrict__ local_grad_alpha_ptr = local_grad_alphas[tid].template data_ptr<T>();

#pragma omp for schedule(guided) collapse(4)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                for (int64_t i = 0; i < in_channels; i++)
                {
                    for (int64_t h = 0; h < height; h++)
                    {
                        const T alpha = alpha_ptr[o * alpha_out_stride + i];
                        const T abs_alpha = std::abs(alpha);
                        const bool is_dilation_like = alpha >= 0;

                        const T* input_channel = input_ptr + b * input_batch_stride + i * input_channel_stride;
                        const T* weight_in = weight_ptr + o * weight_out_stride + i * weight_in_stride;
                        const T* grad_output_base =
                            grad_output_ptr + b * grad_output_batch_stride + o * grad_output_outchan_stride + i * grad_output_inchan_stride;

                        for (int64_t w = 0; w < width; w++)
                        {
                            const T grad_out = grad_output_base[h * width + w];
                            if (grad_out == static_cast<T>(0)) continue;

                            // Recompute forward pass values
                            T reference_value = is_dilation_like ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();

                            std::vector<T> vals(kernel_height * kernel_width);
                            std::vector<bool> valid_positions(kernel_height * kernel_width, false);
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
                                        vals[k] = input_val + weight_in[k];
                                        valid_positions[k] = true;

                                        if (is_dilation_like)
                                        {
                                            reference_value = std::max(vals[k], reference_value);
                                        }
                                        else
                                        {
                                            reference_value = std::min(vals[k], reference_value);
                                        }
                                    }
                                }
                            }

                            // Compute soft weights
                            T sum = static_cast<T>(0);
                            std::vector<T> soft_weights(kernel_height * kernel_width, static_cast<T>(0));

                            for (k = 0; k < kernel_height * kernel_width; k++)
                            {
                                if (valid_positions[k])
                                {
                                    const T exp_term = is_dilation_like ? std::exp((vals[k] - reference_value) * abs_alpha)
                                                                        : std::exp((reference_value - vals[k]) * abs_alpha);
                                    soft_weights[k] = exp_term;
                                    sum += exp_term;
                                }
                            }

                            T smorph_output = static_cast<T>(0);
                            for (k = 0; k < kernel_height * kernel_width; k++)
                            {
                                if (valid_positions[k])
                                {
                                    soft_weights[k] /= sum;
                                    smorph_output += soft_weights[k] * vals[k];
                                }
                            }

                            // Backpropagate gradients
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
                                        const T p = soft_weights[k];
                                        const T val = vals[k];

                                        // Gradient of SMorph w.r.t. val: p + alpha * p * (val - smorph_output)
                                        const T grad_val = p + alpha * p * (val - smorph_output);
                                        const T final_grad = grad_val * grad_out;

                                        // Accumulate gradients
                                        const int64_t input_idx = b * input_batch_stride + i * input_channel_stride + ph * width + pw;
                                        local_grad_input_ptr[input_idx] += final_grad;

                                        const int64_t weight_idx = o * weight_out_stride + i * weight_in_stride + kh * kernel_width + kw;
                                        local_grad_weight_ptr[weight_idx] += final_grad;

                                        // Gradient w.r.t. alpha
                                        const int64_t alpha_idx = o * alpha_out_stride + i;
                                        local_grad_alpha_ptr[alpha_idx] += p * (val - reference_value) * (val - smorph_output) * grad_out;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-local gradients
    T* __restrict__ grad_input_ptr = grad_input.template data_ptr<T>();
    T* __restrict__ grad_weight_ptr = grad_weight.template data_ptr<T>();
    T* __restrict__ grad_alpha_ptr = grad_alpha.template data_ptr<T>();

#pragma omp parallel
    {
#pragma omp for
        for (int64_t i = 0; i < grad_input.numel(); i++)
        {
            T sum = static_cast<T>(0);
            for (const auto& local_tensor : local_grad_inputs)
            {
                sum += local_tensor.template data_ptr<T>()[i];
            }
            grad_input_ptr[i] = sum;
        }

#pragma omp for
        for (int64_t i = 0; i < grad_weight.numel(); i++)
        {
            T sum = static_cast<T>(0);
            for (const auto& local_tensor : local_grad_weights)
            {
                sum += local_tensor.template data_ptr<T>()[i];
            }
            grad_weight_ptr[i] = sum;
        }

#pragma omp for
        for (int64_t i = 0; i < grad_alpha.numel(); i++)
        {
            T sum = static_cast<T>(0);
            for (const auto& local_tensor : local_grad_alphas)
            {
                sum += local_tensor.template data_ptr<T>()[i];
            }
            grad_alpha_ptr[i] = sum;
        }
    }
}