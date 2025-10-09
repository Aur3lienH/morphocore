#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include "morpho_merge_policy.hpp"

/// @brief Core backward implementation for sum/average merge policies for SMorph operations.
/// @tparam merge Merge policy for combining results across channels.
/// @tparam T Data type of tensor elements.
/// @param b Batch index.
/// @param o Output channel index.
/// @param in_channels The number of input channels.
/// @param height The height of the input/output tensors.
/// @param width The width of the input/output tensors.
/// @param kernel_height The height of the kernel.
/// @param kernel_width The width of the kernel.
/// @param kernel_height_half Half of the kernel height.
/// @param kernel_width_half Half of the kernel width.
/// @param alpha The alpha parameter for SMorph.
/// @param input_ptr The pointer to the input tensor data.
/// @param weight_ptr The pointer to the weight tensor data.
/// @param grad_output_ptr The pointer to the gradient output tensor data.
/// @param local_grad_input_ptr The pointer to the local gradient input tensor data.
/// @param local_grad_weight_ptr The pointer to the local gradient weight tensor data.
/// @param local_alpha_grad Reference to thread-local alpha gradient accumulator.
/// @param input_batch_stride The stride of the input tensor in the batch dimension.
/// @param input_channel_stride The stride of the input tensor in the channel dimension.
/// @param weight_out_stride The stride of the weight tensor in the output channel dimension.
/// @param weight_in_stride The stride of the weight tensor in the input channel dimension.
/// @param grad_output_batch_stride The stride of the gradient output tensor in the batch dimension.
/// @param grad_output_channel_stride The stride of the gradient output tensor in the channel dimension.
template <MORPHO_MERGE merge, typename T>
void smorph_backward_sum_avg(int64_t b, int64_t o, int64_t in_channels, int64_t height, int64_t width, int64_t kernel_height, int64_t kernel_width,
                             int64_t kernel_height_half, int64_t kernel_width_half, T alpha, const T* __restrict__ input_ptr,
                             const T* __restrict__ weight_ptr, const T* __restrict__ grad_output_ptr, T* __restrict__ local_grad_input_ptr,
                             T* __restrict__ local_grad_weight_ptr, T& local_alpha_grad, int64_t input_batch_stride, int64_t input_channel_stride,
                             int64_t weight_out_stride, int64_t weight_in_stride, int64_t grad_output_batch_stride,
                             int64_t grad_output_channel_stride)
{

    const T* grad_output_base = grad_output_ptr + b * grad_output_batch_stride + o * grad_output_channel_stride;
    const T abs_alpha = std::abs(alpha);
    const bool is_dilation_like = alpha >= 0;

    // Scale factor for averaging
    const T scale_factor = (merge == MORPHO_MERGE::AVERAGE) ? static_cast<T>(1.0) / in_channels : static_cast<T>(1.0);

    for (int64_t h = 0; h < height; h++)
    {
        for (int64_t w = 0; w < width; w++)
        {
            const T grad_out = grad_output_base[h * width + w] * scale_factor;
            if (grad_out == static_cast<T>(0)) continue;

            for (int64_t i = 0; i < in_channels; i++)
            {
                const T* input_channel = input_ptr + b * input_batch_stride + i * input_channel_stride;
                const T* weight_in = weight_ptr + o * weight_out_stride + i * weight_in_stride;

                T reference_value = is_dilation_like ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();

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

                T sum = static_cast<T>(0);
                std::vector<T> soft_weights(kernel_height * kernel_width, static_cast<T>(0));

                for (k = 0; k < kernel_height * kernel_width; k++)
                {
                    if (valid_positions[k])
                    {
                        const T exp_term =
                            is_dilation_like ? std::exp((vals[k] - reference_value) * abs_alpha) : std::exp((reference_value - vals[k]) * abs_alpha);
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

                            const T grad_val = p + alpha * p * (val - smorph_output);
                            const T final_grad = grad_val * grad_out;

                            const int64_t input_idx = b * input_batch_stride + i * input_channel_stride + ph * width + pw;
                            local_grad_input_ptr[input_idx] += final_grad;

                            const int64_t weight_idx = o * weight_out_stride + i * weight_in_stride + kh * kernel_width + kw;
                            local_grad_weight_ptr[weight_idx] += final_grad;

                            local_alpha_grad += p * (val - reference_value) * (val - smorph_output) * grad_out;
                        }
                    }
                }
            }
        }
    }
}

/// @brief Core backward implementation for min/max merge policies for SMorph operations.
/// @tparam merge Merge policy for combining results across channels.
/// @tparam T Data type of tensor elements.
/// @param b Batch index.
/// @param o Output channel index.
/// @param in_channels The number of input channels.
/// @param height The height of the input/output tensors.
/// @param width The width of the input/output tensors.
/// @param kernel_height The height of the kernel.
/// @param kernel_width The width of the kernel.
/// @param kernel_height_half Half of the kernel height.
/// @param kernel_width_half Half of the kernel width.
/// @param alpha The alpha parameter for SMorph.
/// @param input_ptr The pointer to the input tensor data.
/// @param weight_ptr The pointer to the weight tensor data.
/// @param grad_output_ptr The pointer to the gradient output tensor data.
/// @param local_grad_input_ptr The pointer to the local gradient input tensor data.
/// @param local_grad_weight_ptr The pointer to the local gradient weight tensor data.
/// @param local_alpha_grad Reference to thread-local alpha gradient accumulator.
/// @param input_batch_stride The stride of the input tensor in the batch dimension.
/// @param input_channel_stride The stride of the input tensor in the channel dimension.
/// @param weight_out_stride The stride of the weight tensor in the output channel dimension.
/// @param weight_in_stride The stride of the weight tensor in the input channel dimension.
/// @param grad_output_batch_stride The stride of the gradient output tensor in the batch dimension.
/// @param grad_output_channel_stride The stride of the gradient output tensor in the channel dimension.
template <MORPHO_MERGE merge, typename T>
void smorph_backward_min_max(int64_t b, int64_t o, int64_t in_channels, int64_t height, int64_t width, int64_t kernel_height, int64_t kernel_width,
                             int64_t kernel_height_half, int64_t kernel_width_half, T alpha, const T* __restrict__ input_ptr,
                             const T* __restrict__ weight_ptr, const T* __restrict__ grad_output_ptr, T* __restrict__ local_grad_input_ptr,
                             T* __restrict__ local_grad_weight_ptr, T& local_alpha_grad, int64_t input_batch_stride, int64_t input_channel_stride,
                             int64_t weight_out_stride, int64_t weight_in_stride, int64_t grad_output_batch_stride,
                             int64_t grad_output_channel_stride)
{
    const T* grad_output_base = grad_output_ptr + b * grad_output_batch_stride + o * grad_output_channel_stride;
    const T abs_alpha = std::abs(alpha);
    const bool is_dilation_like = alpha >= 0;
    std::vector<T> channel_reference_values(in_channels);

    for (int64_t h = 0; h < height; h++)
    {
        for (int64_t w = 0; w < width; w++)
        {
            const T grad_out = grad_output_base[h * width + w];
            if (grad_out == static_cast<T>(0)) continue;

            std::vector<T> channel_outputs(in_channels);
            std::vector<std::vector<T>> channel_vals(in_channels);
            std::vector<std::vector<T>> channel_soft_weights(in_channels);

            for (int64_t i = 0; i < in_channels; i++)
            {
                const T* input_channel = input_ptr + b * input_batch_stride + i * input_channel_stride;
                const T* weight_in = weight_ptr + o * weight_out_stride + i * weight_in_stride;

                T reference_value = is_dilation_like ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();

                channel_vals[i].resize(kernel_height * kernel_width);
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
                            channel_vals[i][k] = input_val + weight_in[k];
                            valid_positions[k] = true;

                            if (is_dilation_like)
                            {
                                reference_value = std::max(channel_vals[i][k], reference_value);
                            }
                            else
                            {
                                reference_value = std::min(channel_vals[i][k], reference_value);
                            }
                        }
                    }
                }
                channel_reference_values[i] = reference_value;

                T sum = static_cast<T>(0);
                channel_soft_weights[i].resize(kernel_height * kernel_width, static_cast<T>(0));

                for (k = 0; k < kernel_height * kernel_width; k++)
                {
                    if (valid_positions[k])
                    {
                        const T exp_term = is_dilation_like ? std::exp((channel_vals[i][k] - reference_value) * abs_alpha)
                                                            : std::exp((reference_value - channel_vals[i][k]) * abs_alpha);
                        channel_soft_weights[i][k] = exp_term;
                        sum += exp_term;
                    }
                }

                channel_outputs[i] = static_cast<T>(0);
                for (k = 0; k < kernel_height * kernel_width; k++)
                {
                    if (valid_positions[k])
                    {
                        channel_soft_weights[i][k] /= sum;
                        channel_outputs[i] += channel_soft_weights[i][k] * channel_vals[i][k];
                    }
                }
            }

            T target_value;
            if constexpr (merge == MORPHO_MERGE::MAX)
            {
                target_value = *std::max_element(channel_outputs.begin(), channel_outputs.end());
            }
            else
            {
                target_value = *std::min_element(channel_outputs.begin(), channel_outputs.end());
            }

            int64_t winner_count = 0;
            const T tolerance = static_cast<T>(1e-7);
            for (const T& val : channel_outputs)
            {
                if (std::abs(val - target_value) < tolerance) winner_count++;
            }

            const T winner_grad = grad_out / static_cast<T>(winner_count);

            for (int64_t i = 0; i < in_channels; i++)
            {
                if (std::abs(channel_outputs[i] - target_value) < tolerance)
                {
                    const T smorph_output = channel_outputs[i];
                    const T reference_value = channel_reference_values[i];
                    int64_t k = 0;

                    for (int64_t kh = 0; kh < kernel_height; kh++)
                    {
                        const int64_t ph = h + kernel_height_half - kh;
                        const bool h_valid = (ph >= 0 && ph < height);

                        for (int64_t kw = 0; kw < kernel_width; kw++, k++)
                        {
                            const int64_t pw = w + kernel_width_half - kw;
                            const bool w_valid = (pw >= 0 && pw < width);

                            if (h_valid && w_valid && channel_soft_weights[i][k] > static_cast<T>(0))
                            {
                                const T p = channel_soft_weights[i][k];
                                const T val = channel_vals[i][k];

                                const T grad_val = p + alpha * p * (val - smorph_output);
                                const T final_grad = grad_val * winner_grad;

                                const int64_t input_idx = b * input_batch_stride + i * input_channel_stride + ph * width + pw;
                                local_grad_input_ptr[input_idx] += final_grad;

                                const int64_t weight_idx = o * weight_out_stride + i * weight_in_stride + kh * kernel_width + kw;
                                local_grad_weight_ptr[weight_idx] += final_grad;

                                local_alpha_grad += p * (val - reference_value) * (val - smorph_output) * winner_grad;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// @brief Main backward pass dispatcher for SMorph operations on CPU.
/// @tparam merge The merge strategy for combining gradients.
/// @tparam T The data type of the tensor elements.
/// @param input The input tensor.
/// @param weight The weight tensor.
/// @param alpha The alpha parameter.
/// @param grad_output The gradient output tensor.
/// @param grad_input The gradient input tensor.
/// @param grad_weight The gradient weight tensor.
/// @param grad_alpha The gradient alpha tensor.
template <MORPHO_MERGE merge, typename T>
void smorph_backward_cpu(torch::Tensor input, torch::Tensor weight, T alpha, torch::Tensor grad_output, torch::Tensor grad_input,
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

    // Ensure contiguous memory layout for better cache performance
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto grad_output_contig = grad_output.contiguous();

    // Use raw pointers for better performance
    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ grad_output_ptr = grad_output_contig.template data_ptr<T>();
    T* __restrict__ grad_alpha_ptr = grad_alpha.template data_ptr<T>();

    // Pre-computed strides for contiguous tensors
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;

    // Thread-local storage to eliminate atomic operations
    const int num_threads = omp_get_max_threads();
    std::vector<torch::Tensor> local_grad_inputs(num_threads);
    std::vector<torch::Tensor> local_grad_weights(num_threads);
    std::vector<T> local_alpha_grads(num_threads, static_cast<T>(0));

    for (int tid = 0; tid < num_threads; tid++)
    {
        local_grad_inputs[tid] = torch::zeros_like(grad_input);
        local_grad_weights[tid] = torch::zeros_like(grad_weight);
    }

    // Initialize gradients to zero
    grad_input.zero_();
    grad_weight.zero_();
    grad_alpha.zero_();

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        T* __restrict__ local_grad_input_ptr = local_grad_inputs[tid].template data_ptr<T>();
        T* __restrict__ local_grad_weight_ptr = local_grad_weights[tid].template data_ptr<T>();
        T& local_alpha_grad = local_alpha_grads[tid];

#pragma omp for schedule(guided) collapse(2)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                if constexpr (merge == MORPHO_MERGE::ADD || merge == MORPHO_MERGE::AVERAGE)
                {
                    smorph_backward_sum_avg<merge, T>(b, o, in_channels, height, width, kernel_height, kernel_width, kernel_height_half,
                                                      kernel_width_half, alpha, input_ptr, weight_ptr, grad_output_ptr, local_grad_input_ptr,
                                                      local_grad_weight_ptr, local_alpha_grad, input_batch_stride, input_channel_stride,
                                                      weight_out_stride, weight_in_stride, output_batch_stride, output_channel_stride);
                }
                else
                {
                    smorph_backward_min_max<merge, T>(b, o, in_channels, height, width, kernel_height, kernel_width, kernel_height_half,
                                                      kernel_width_half, alpha, input_ptr, weight_ptr, grad_output_ptr, local_grad_input_ptr,
                                                      local_grad_weight_ptr, local_alpha_grad, input_batch_stride, input_channel_stride,
                                                      weight_out_stride, weight_in_stride, output_batch_stride, output_channel_stride);
                }
            }
        }
    }

    T* __restrict__ grad_input_ptr = grad_input.template data_ptr<T>();
    T* __restrict__ grad_weight_ptr = grad_weight.template data_ptr<T>();

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

#pragma omp single
        {
            for (const T& local_alpha_grad : local_alpha_grads)
            {
                grad_alpha_ptr[0] += local_alpha_grad;
            }
        }
    }
}