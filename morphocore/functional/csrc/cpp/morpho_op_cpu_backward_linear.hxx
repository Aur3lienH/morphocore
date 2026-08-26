#pragma once
#include <torch/torch.h>
#include <omp.h>
#include "morpho_op_enum.hpp"
#include "morpho_op_policy.hpp"
#include "morpho_padding_policy.hpp"

/// @brief Backward pass for morphological operations with IDENTITY channel merge.
/// Each channel's gradient is computed independently.
/// @tparam op Morphological operation type (DILATION or EROSION).
/// @tparam kind Padding family used for out of bounds taps.
/// @tparam T Data type of tensor elements.
/// @tparam save_index Whether to use precomputed indices.
/// @param input Input tensor of shape [batch, channels, height, width].
/// @param weight Weight tensor of shape [channels, 1, kernel_height, kernel_width].
/// @param grad_output Gradient of output tensor.
/// @param max_indices Precomputed indices (if save_index is true).
/// @param fill Border fill value, see morpho_fill_value. Unused when kind is REPLICATE.
/// @return Gradients for input and weight.
template <MORPHO_OPERATION op, MORPHO_PADDING_KIND kind, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_identity_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output,
                                                                    torch::Tensor max_indices, T fill)
{
    using Policy = MorphoPolicyCPU<T, op>;

    const auto input_shape = input.sizes();
    const auto weight_shape = weight.sizes();

    const int64_t batch_size = input_shape[0];
    const int64_t channels = input_shape[1];
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];
    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;

    // Ensure contiguous memory
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto grad_output_contig = grad_output.contiguous();

    auto grad_input = torch::zeros_like(input_contig);
    auto grad_weight = torch::zeros_like(weight_contig);

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ grad_output_ptr = grad_output_contig.template data_ptr<T>();
    T* __restrict__ grad_input_ptr = grad_input.template data_ptr<T>();

    // Strides
    const int64_t input_batch_stride = channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_channel_stride = kernel_height * kernel_width;
    const int64_t grad_output_batch_stride = channels * height * width;
    const int64_t grad_output_channel_stride = height * width;

    // Thread-local storage
    const int num_threads = omp_get_max_threads();
    std::vector<torch::Tensor> local_grad_weights(num_threads);

    for (int tid = 0; tid < num_threads; tid++)
    {
        local_grad_weights[tid] = torch::zeros_like(grad_weight);
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        T* __restrict__ local_grad_weight_ptr = local_grad_weights[tid].template data_ptr<T>();

#pragma omp for schedule(guided) collapse(2)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t c = 0; c < channels; c++)
            {
                const T* input_channel = input_ptr + b * input_batch_stride + c * input_channel_stride;
                const T* weight_channel = weight_ptr + c * weight_channel_stride;
                const T* grad_output_channel = grad_output_ptr + b * grad_output_batch_stride + c * grad_output_channel_stride;

                for (int64_t h = 0; h < height; h++)
                {
                    for (int64_t w = 0; w < width; w++)
                    {
                        const T grad_val = grad_output_channel[h * width + w];
                        if (grad_val == static_cast<T>(0)) continue;

                        // Find the best kernel position for this output location
                        T best_value = Policy::initial_value();
                        int64_t best_kh = -1;
                        int64_t best_kw = -1;

                        for (int64_t kh = 0; kh < kernel_height; kh++)
                        {
                            const int64_t ph = h + kernel_height_half - kh;
                            const T* input_row = morpho_row<kind>(input_channel, ph, height, width);

                            for (int64_t kw = 0; kw < kernel_width; kw++)
                            {
                                const int64_t pw = w + kernel_width_half - kw;

                                const T input_val = morpho_load_row<kind>(input_row, pw, width, fill);
                                const T val = Policy::combine(input_val, weight_channel[kh * kernel_width + kw]);

                                if (Policy::is_better(val, best_value))
                                {
                                    best_value = val;
                                    best_kh = kh;
                                    best_kw = kw;
                                }
                            }
                        }

                        // Backprop to input. A winning tap on the border read padding rather than a
                        // pixel, so it carries no input gradient, REPLICATE included.
                        const int64_t ih = h + kernel_height_half - best_kh;
                        const int64_t iw = w + kernel_width_half - best_kw;

                        if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                        {
                            const int64_t grad_input_idx = b * input_batch_stride + c * input_channel_stride + ih * width + iw;
#pragma omp atomic
                            grad_input_ptr[grad_input_idx] += grad_val;
                        }

                        // Backprop to weight
                        const int64_t grad_weight_idx = c * weight_channel_stride + best_kh * kernel_width + best_kw;

                        if constexpr (op == MORPHO_OPERATION::DILATION)
                        {
                            local_grad_weight_ptr[grad_weight_idx] += grad_val;
                        }
                        else  // EROSION
                        {
                            local_grad_weight_ptr[grad_weight_idx] -= grad_val;
                        }
                    }
                }
            }
        }
    }

    // Accumulate thread-local weight gradients
#pragma omp parallel for
    for (int64_t i = 0; i < grad_weight.numel(); i++)
    {
        T sum = static_cast<T>(0);
        for (const auto& local_tensor : local_grad_weights)
        {
            sum += local_tensor.template data_ptr<T>()[i];
        }
        grad_weight.template data_ptr<T>()[i] = sum;
    }

    return {grad_input, grad_weight};
}