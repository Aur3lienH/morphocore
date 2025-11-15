#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <immintrin.h>  // For AVX, etc. (though not explicitly used in this logic, good to keep)
#include <omp.h>        // For OpenMP
#include <cmath>        // For std::abs, std::exp, std::max, std::min
#include <limits>       // For std::numeric_limits
#include "morpho_merge_policy.hpp"

// Forward declaration of the non-beta backward pass (if it's in a different file)
// This isn't strictly necessary for this file, but good practice if they are co-located.
// We will omit it for a clean, self-contained file.

/// @brief Core backward implementation for beta-softmax merge policies for SMorph operations.
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
/// @param alpha_out Pointer to alpha values for this output channel.
/// @param beta_val The beta value for this output channel.
/// @param input_ptr The pointer to the input tensor data.
/// @param weight_ptr The pointer to the weight tensor data.
/// @param grad_output_ptr The pointer to the gradient output tensor data.
/// @param local_grad_input_ptr The pointer to the local gradient input tensor data.
/// @param local_grad_weight_ptr The pointer to the local gradient weight tensor data.
/// @param local_alpha_grad_ptr Pointer to thread-local alpha gradient accumulator for this output channel.
/// @param local_grad_beta_ptr Pointer to thread-local beta gradient accumulator for this output channel.
/// @param input_batch_stride The stride of the input tensor in the batch dimension.
/// @param input_channel_stride The stride of the input tensor in the channel dimension.
/// @param weight_out_stride The stride of the weight tensor in the output channel dimension.
/// @param weight_in_stride The stride of the weight tensor in the input channel dimension.
/// @param grad_output_batch_stride The stride of the gradient output tensor in the batch dimension.
/// @param grad_output_channel_stride The stride of the gradient output tensor in the channel dimension.
template <typename T>
void smorph_scm_backward_beta(int64_t b, int64_t o, int64_t in_channels, int64_t height, int64_t width, int64_t kernel_height, int64_t kernel_width,
                              int64_t kernel_height_half, int64_t kernel_width_half, const T* __restrict__ alpha_out, const T beta_val,
                              const T* __restrict__ input_ptr, const T* __restrict__ weight_ptr, const T* __restrict__ grad_output_ptr,
                              T* __restrict__ local_grad_input_ptr, T* __restrict__ local_grad_weight_ptr, T* __restrict__ local_alpha_grad_ptr,
                              T* __restrict__ local_grad_beta_ptr, int64_t input_batch_stride, int64_t input_channel_stride,
                              int64_t weight_out_stride, int64_t weight_in_stride, int64_t grad_output_batch_stride,
                              int64_t grad_output_channel_stride)
{
    const T* grad_output_base = grad_output_ptr + b * grad_output_batch_stride + o * grad_output_channel_stride;
    const T beta_abs = std::abs(beta_val);
    const bool beta_positive = beta_val >= 0;

    for (int64_t h = 0; h < height; h++)
    {
        for (int64_t w = 0; w < width; w++)
        {
            const T grad_out = grad_output_base[h * width + w];
            if (grad_out == static_cast<T>(0)) continue;

            // --- 1. Re-compute Alpha-Smorph (Inner) Forward Pass ---
            // We must re-calculate the forward pass for each channel to get intermediate values.
            std::vector<T> channel_outputs(in_channels);
            std::vector<T> channel_reference_values(in_channels);
            std::vector<std::vector<T>> channel_vals(in_channels, std::vector<T>(kernel_height * kernel_width));
            std::vector<std::vector<T>> channel_soft_weights(in_channels, std::vector<T>(kernel_height * kernel_width, static_cast<T>(0)));
            std::vector<std::vector<bool>> channel_valid_positions(in_channels, std::vector<bool>(kernel_height * kernel_width, false));

            for (int64_t i = 0; i < in_channels; i++)
            {
                const T alpha = alpha_out[i];
                const T abs_alpha = std::abs(alpha);
                const bool is_dilation_like = alpha >= 0;

                const T* input_channel = input_ptr + b * input_batch_stride + i * input_channel_stride;
                const T* weight_in = weight_ptr + o * weight_out_stride + i * weight_in_stride;

                T reference_value = is_dilation_like ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
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
                            const T val = input_channel[ph * width + pw] + weight_in[k];
                            channel_vals[i][k] = val;
                            channel_valid_positions[i][k] = true;
                            reference_value = is_dilation_like ? std::max(val, reference_value) : std::min(val, reference_value);
                        }
                    }
                }

                T sum = static_cast<T>(0);
                for (k = 0; k < kernel_height * kernel_width; k++)
                {
                    if (channel_valid_positions[i][k])
                    {
                        const T val = channel_vals[i][k];
                        const T exp_term =
                            is_dilation_like ? std::exp((val - reference_value) * abs_alpha) : std::exp((reference_value - val) * abs_alpha);
                        channel_soft_weights[i][k] = exp_term;
                        sum += exp_term;
                    }
                }

                T smorph_output = static_cast<T>(0);
                if (sum > std::numeric_limits<T>::epsilon())
                {
                    for (k = 0; k < kernel_height * kernel_width; k++)
                    {
                        if (channel_valid_positions[i][k])
                        {
                            channel_soft_weights[i][k] /= sum;
                            smorph_output += channel_soft_weights[i][k] * channel_vals[i][k];
                        }
                    }
                }
                channel_outputs[i] = smorph_output;
                channel_reference_values[i] = reference_value;
            }

            // --- 2. Re-compute Beta-Smorph (Outer) Forward Pass ---
            T beta_ref = beta_positive ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
            for (int64_t i = 0; i < in_channels; i++)
            {
                beta_ref = beta_positive ? std::max(channel_outputs[i], beta_ref) : std::min(channel_outputs[i], beta_ref);
            }

            T beta_sum = static_cast<T>(0);
            std::vector<T> beta_soft_weights(in_channels);
            for (int64_t i = 0; i < in_channels; i++)
            {
                const T c_i = channel_outputs[i];
                const T exp_term = beta_positive ? std::exp(beta_abs * (c_i - beta_ref)) : std::exp(beta_abs * (beta_ref - c_i));
                beta_soft_weights[i] = exp_term;
                beta_sum += exp_term;
            }

            T final_output = static_cast<T>(0);
            if (beta_sum > std::numeric_limits<T>::epsilon())
            {
                for (int64_t i = 0; i < in_channels; i++)
                {
                    beta_soft_weights[i] /= beta_sum;
                    final_output += beta_soft_weights[i] * channel_outputs[i];
                }
            }

            // --- 3. Compute Gradients (Backward Pass) ---

            // 3a. Gradient w.r.t. Beta
            T grad_beta_pixel = static_cast<T>(0);
            for (int64_t i = 0; i < in_channels; i++)
            {
                const T p_i = beta_soft_weights[i];
                const T c_i = channel_outputs[i];
                // d(exp_arg) / d(beta) = (c_i - beta_ref) * sign(beta_val)
                // d(beta_val * (c_i - beta_ref)) / d(beta_val) = c_i - beta_ref
                // But since beta_abs is used, it's (c_i - beta_ref) * sign(beta_val)
                // And d(softmax) / d(beta) = ... p_i * (c_i - beta_ref) * (c_i - final_output) * sign(beta_val)
                // Let's assume beta is always positive for simplicity or that d(abs(beta)) / d(beta) = sign(beta)
                // The derivative of softmax(z_i * alpha) w.r.t alpha is: sum(p_i * z_i * (z_i - output))
                // Here z_i = (c_i - beta_ref) [or (beta_ref - c_i)]
                // Let's use d(beta_abs * (c_i - beta_ref)) / d(beta) = sign(beta) * (c_i - beta_ref)
                T exp_arg_deriv = beta_positive ? (c_i - beta_ref) : -(beta_ref - c_i);  // This is (c_i - beta_ref)

                // d(abs(beta))/d(beta) = sign(beta).
                // Since d(exp_arg)/d(beta) = sign(beta) * (c_i - beta_ref), and alpha derivative is p * (z-ref) * (z-out),
                // we get:
                grad_beta_pixel += p_i * exp_arg_deriv * (c_i - final_output);
            }
            // We multiply by sign(beta) again if beta is not guaranteed positive
            // d(beta_val)/d(beta_val) = 1. d(abs(beta_val))/d(beta_val) = sign(beta_val)
            // The grad_beta_pixel expression is correct if beta_val is the parameter.
            // If beta_abs is the parameter, it's different.
            // Assuming beta_val is the parameter:
            local_grad_beta_ptr[0] += grad_beta_pixel * grad_out;

            // 3b. Gradient w.r.t. Channel Outputs (c_i) and propagate down
            for (int64_t i = 0; i < in_channels; i++)
            {
                const T p_i = beta_soft_weights[i];
                const T c_i = channel_outputs[i];

                // Grad w.r.t. c_i
                const T grad_c_i = p_i + beta_val * p_i * (c_i - final_output);
                const T new_grad_out = grad_c_i * grad_out;  // This is the grad to pass to the alpha-smorph

                if (new_grad_out == static_cast<T>(0)) continue;

                // --- 3c. Propagate 'new_grad_out' through the Alpha-Smorph (Inner) ---
                const T alpha = alpha_out[i];
                const T reference_value = channel_reference_values[i];  // Alpha-ref
                const T smorph_output = channel_outputs[i];             // Alpha-output (c_i)

                // 3c.i. Gradient w.r.t. Alpha
                T grad_alpha_pixel = static_cast<T>(0);
                for (int64_t k = 0; k < kernel_height * kernel_width; k++)
                {
                    if (channel_valid_positions[i][k])
                    {
                        const T p_k = channel_soft_weights[i][k];
                        const T val = channel_vals[i][k];
                        // d(alpha * (val-ref)) / d(alpha) = (val-ref)
                        T exp_arg_deriv_alpha = alpha >= 0 ? (val - reference_value) : -(reference_value - val);  // (val - reference_value)
                        grad_alpha_pixel += p_k * exp_arg_deriv_alpha * (val - smorph_output);
                    }
                }
                local_alpha_grad_ptr[i] += grad_alpha_pixel * new_grad_out;

                // 3c.ii. Gradient w.r.t. Input and Weight
                int64_t k = 0;
                for (int64_t kh = 0; kh < kernel_height; kh++)
                {
                    const int64_t ph = h + kernel_height_half - kh;
                    const bool h_valid = (ph >= 0 && ph < height);
                    for (int64_t kw = 0; kw < kernel_width; kw++, k++)
                    {
                        const int64_t pw = w + kernel_width_half - kw;
                        const bool w_valid = (pw >= 0 && pw < width);

                        if (h_valid && w_valid && channel_valid_positions[i][k])
                        {
                            const T p_k = channel_soft_weights[i][k];
                            const T val = channel_vals[i][k];

                            const T grad_val = p_k + alpha * p_k * (val - smorph_output);
                            const T final_grad = grad_val * new_grad_out;

                            const int64_t input_idx = b * input_batch_stride + i * input_channel_stride + ph * width + pw;
                            local_grad_input_ptr[input_idx] += final_grad;

                            const int64_t weight_idx = o * weight_out_stride + i * weight_in_stride + kh * kernel_width + kw;
                            local_grad_weight_ptr[weight_idx] += final_grad;
                        }
                    }
                }
            }
        }
    }
}

/// @brief Main backward pass dispatcher for BETA SMorph operations.
/// @tparam merge The merge strategy (must be BETA).
/// @tparam T The data type of the tensor elements.
/// @param input The input tensor.
/// @param weight The weight tensor.
/// @param alpha The alpha parameter tensor of shape [out_channels, in_channels].
/// @param beta The beta parameter tensor of shape [out_channels].
/// @param grad_output The gradient output tensor.
/// @param grad_input The gradient input tensor.
/// @param grad_weight The gradient weight tensor.
/// @param grad_alpha The gradient alpha tensor of shape [out_channels, in_channels].
/// @param grad_beta The gradient beta tensor of shape [out_channels].
template <typename T>
void smorph_scm_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor beta, torch::Tensor grad_output,
                             torch::Tensor grad_input, torch::Tensor grad_weight, torch::Tensor grad_alpha, torch::Tensor grad_beta)
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

    // Ensure contiguous memory layout
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto alpha_contig = alpha.contiguous();
    auto beta_contig = beta.contiguous();
    auto grad_output_contig = grad_output.contiguous();

    // Use raw pointers for better performance
    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ alpha_ptr = alpha_contig.template data_ptr<T>();
    const T* __restrict__ beta_ptr = beta_contig.template data_ptr<T>();
    const T* __restrict__ grad_output_ptr = grad_output_contig.template data_ptr<T>();

    // Pre-computed strides
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;
    const int64_t alpha_out_stride = in_channels;

    // Thread-local storage
    const int num_threads = omp_get_max_threads();
    std::vector<torch::Tensor> local_grad_inputs(num_threads);
    std::vector<torch::Tensor> local_grad_weights(num_threads);
    std::vector<torch::Tensor> local_grad_alphas(num_threads);
    std::vector<torch::Tensor> local_grad_betas(num_threads);

    for (int tid = 0; tid < num_threads; tid++)
    {
        local_grad_inputs[tid] = torch::zeros_like(grad_input);
        local_grad_weights[tid] = torch::zeros_like(grad_weight);
        local_grad_alphas[tid] = torch::zeros_like(grad_alpha);
        local_grad_betas[tid] = torch::zeros_like(grad_beta);
    }

    // Initialize gradients to zero
    grad_input.zero_();
    grad_weight.zero_();
    grad_alpha.zero_();
    grad_beta.zero_();

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        T* __restrict__ local_grad_input_ptr = local_grad_inputs[tid].template data_ptr<T>();
        T* __restrict__ local_grad_weight_ptr = local_grad_weights[tid].template data_ptr<T>();
        T* __restrict__ local_grad_alpha_ptr = local_grad_alphas[tid].template data_ptr<T>();
        T* __restrict__ local_grad_beta_ptr = local_grad_betas[tid].template data_ptr<T>();

#pragma omp for schedule(guided) collapse(2)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                const T* alpha_out = alpha_ptr + o * alpha_out_stride;
                T* local_alpha_grad_out = local_grad_alpha_ptr + o * alpha_out_stride;

                // Pointers for beta
                const T beta_val = beta_ptr[o];
                T* local_grad_beta_out = local_grad_beta_ptr + o;

                // We know it's BETA, so we call the helper directly
                smorph_scm_backward_beta<T>(b, o, in_channels, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half,
                                            alpha_out, beta_val, input_ptr, weight_ptr, grad_output_ptr, local_grad_input_ptr, local_grad_weight_ptr,
                                            local_alpha_grad_out, local_grad_beta_out, input_batch_stride, input_channel_stride, weight_out_stride,
                                            weight_in_stride, output_batch_stride, output_channel_stride);
            }
        }
    }

    // --- Final Reduction (With Beta) ---
    T* __restrict__ grad_input_ptr = grad_input.template data_ptr<T>();
    T* __restrict__ grad_weight_ptr = grad_weight.template data_ptr<T>();
    T* __restrict__ grad_alpha_ptr = grad_alpha.template data_ptr<T>();
    T* __restrict__ grad_beta_ptr = grad_beta.template data_ptr<T>();

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

#pragma omp for
        for (int64_t i = 0; i < grad_beta.numel(); i++)
        {
            T sum = static_cast<T>(0);
            for (const auto& local_tensor : local_grad_betas)
            {
                sum += local_tensor.template data_ptr<T>()[i];
            }
            grad_beta_ptr[i] = sum;
        }
    }
}