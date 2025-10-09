#pragma once
#include <memory>
#include <vector>

#include <immintrin.h>
#include <omp.h>

#include "morpho_op_enum.hpp"
#include "morpho_op_policy.hpp"

// This function should be inlined because of the number of arguments and it is in a hot loop !
/// @brief Core backward implementation for sum/average merge policies on CPU for morphological operations.
/// @tparam T Data type of tensor elements.
/// @tparam op Morphological operation type (e.g., dilation, erosion).
/// @tparam merge Merge policy for combining results across channels.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position for each output element.
/// @tparam outside_in_channels If true, processes outside in channels.
/// @param b Batch index.
/// @param o Output channel index.
/// @param in_channels The number of input channels.
/// @param height The height of the input/output tensors.
/// @param width The width of the input/output tensors.
/// @param kernel_height The height of the kernel.
/// @param kernel_width The width of the kernel.
/// @param kernel_height_half Half of the kernel height.
/// @param kernel_width_half Half of the kernel width.
/// @param input_ptr The pointer to the input tensor data.
/// @param weight_ptr The pointer to the weight tensor data.
/// @param grad_output_ptr The pointer to the gradient output tensor data.
/// @param grad_input_ptr The pointer to the gradient input tensor data.
/// @param local_grad_weight_ptr The pointer to the local gradient weight tensor data.
/// @param input_batch_stride The stride of the input tensor in the batch dimension.
/// @param input_channel_stride The stride of the input tensor in the channel dimension.
/// @param weight_out_stride The stride of the weight tensor in the output channel dimension.
/// @param weight_in_stride The stride of the weight tensor in the input channel dimension.
/// @param grad_output_batch_stride The stride of the gradient output tensor in the batch dimension.
/// @param grad_output_channel_stride The stride of the gradient output tensor in the channel dimension.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index, bool outside_in_channels>
void morpho_backward_sum_avg(int64_t b, int64_t o, int64_t in_channels, int64_t height, int64_t width, int64_t kernel_height, int64_t kernel_width,
                             int64_t kernel_height_half, int64_t kernel_width_half, const T* __restrict__ input_ptr, const T* __restrict__ weight_ptr,
                             const T* __restrict__ grad_output_ptr, T* __restrict__ grad_input_ptr, T* __restrict__ local_grad_weight_ptr,
                             int64_t input_batch_stride, int64_t input_channel_stride, int64_t weight_out_stride, int64_t weight_in_stride,
                             int64_t grad_output_batch_stride, int64_t grad_output_channel_stride)
{
    using Policy = MorphoPolicyCPU<T, op>;

    const T* grad_output_base = grad_output_ptr + b * grad_output_batch_stride + o * grad_output_channel_stride;

    const T scale_factor = (merge == MORPHO_MERGE::AVERAGE) ? static_cast<T>(1.0) / in_channels : static_cast<T>(1.0);

    for (int64_t h = 0; h < height; h++)
    {
        for (int64_t w = 0; w < width; w++)
        {
            const T grad_val = grad_output_base[h * width + w] * scale_factor;
            if (grad_val == static_cast<T>(0)) continue;

            for (int64_t i = 0; i < in_channels; i++)
            {
                const T* input_channel = input_ptr + b * input_batch_stride + i * input_channel_stride;
                const T* weight_base = weight_ptr + o * weight_out_stride + i * weight_in_stride;

                T best_value = Policy::initial_value();
                int64_t best_kh = -1;
                int64_t best_kw = -1;

                for (int64_t kh = 0; kh < kernel_height; kh++)
                {
                    for (int64_t kw = 0; kw < kernel_width; kw++)
                    {
                        const int64_t ph = h + kernel_height_half - kh;
                        const int64_t pw = w + kernel_width_half - kw;

                        const bool h_valid = (ph >= 0 && ph < height);
                        const bool w_valid = (pw >= 0 && pw < width);

                        const T input_val = (h_valid && w_valid) ? input_channel[ph * width + pw] : Policy::padding_value();

                        const T val = Policy::combine(input_val, weight_base[kh * kernel_width + kw]);

                        if (Policy::is_better(val, best_value))
                        {
                            best_value = val;
                            best_kh = kh;
                            best_kw = kw;
                        }
                    }
                }

                const int64_t ih = h + kernel_height_half - best_kh;
                const int64_t iw = w + kernel_width_half - best_kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                {
                    const int64_t grad_input_idx = b * input_batch_stride + i * input_channel_stride + ih * width + iw;
                    grad_input_ptr[grad_input_idx] += grad_val;
                }

                const int64_t grad_weight_idx = o * weight_out_stride + i * weight_in_stride + best_kh * kernel_width + best_kw;

                if constexpr (op == MORPHO_OPERATION::DILATION)
                {
                    local_grad_weight_ptr[grad_weight_idx] += grad_val;
                }
                else
                {
                    local_grad_weight_ptr[grad_weight_idx] -= grad_val;
                }
            }
        }
    }
}

/// @brief Core backward implementation for min/max merge policies on CPU for morphological operations.
/// @tparam T Data type of tensor elements.
/// @tparam op Morphological operation type (e.g., dilation, erosion).
/// @tparam merge Merge policy for combining results across channels.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position
/// @tparam outside_in_channels If true, processes outside in channels.
/// @param b Batch index.
/// @param o Output channel index.
/// @param in_channels The number of input channels.
/// @param height The height of the input/output tensors.
/// @param width The width of the input/output tensors.
/// @param kernel_height The height of the kernel.
/// @param kernel_width The width of the kernel.
/// @param kernel_height_half Half of the kernel height.
/// @param kernel_width_half Half of the kernel width.
/// @param input_ptr The pointer to the input tensor data.
/// @param weight_ptr The pointer to the weight tensor data.
/// @param grad_output_ptr The pointer to the gradient output tensor data.
/// @param grad_input_ptr The pointer to the gradient input tensor data.
/// @param local_grad_weight_ptr The pointer to the local gradient weight tensor data.
/// @param input_batch_stride The stride of the input tensor in the batch dimension.
/// @param input_channel_stride The stride of the input tensor in the channel dimension.
/// @param weight_out_stride The stride of the weight tensor in the output channel dimension.
/// @param weight_in_stride The stride of the weight tensor in the input channel dimension.
/// @param grad_output_batch_stride The stride of the gradient output tensor in the batch dimension.
/// @param grad_output_channel_stride The stride of the gradient output tensor in the channel dimension.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index, bool outside_in_channels>
void morpho_backward_min_max(int64_t b, int64_t o, int64_t in_channels, int64_t height, int64_t width, int64_t kernel_height, int64_t kernel_width,
                             int64_t kernel_height_half, int64_t kernel_width_half, const T* __restrict__ input_ptr, const T* __restrict__ weight_ptr,
                             const T* __restrict__ grad_output_ptr, T* __restrict__ grad_input_ptr, T* __restrict__ local_grad_weight_ptr,
                             int64_t input_batch_stride, int64_t input_channel_stride, int64_t weight_out_stride, int64_t weight_in_stride,
                             int64_t grad_output_batch_stride, int64_t grad_output_channel_stride)

{

    // Policy for choosing between erosion and dilation.
    using Policy = MorphoPolicyCPU<T, op>;
    using MergePolicy = MorphoMergePolicy<T, merge>;

    const T* grad_output_base = grad_output_ptr + b * grad_output_batch_stride + o * grad_output_channel_stride;

    for (int64_t h = 0; h < height; h++)
    {
        for (int64_t w = 0; w < width; w++)
        {
            const T grad_val = grad_output_base[h * width + w];
            if (grad_val == static_cast<T>(0)) continue;

            T best_value = MergePolicy::get_init_value();
            int64_t best_kh = -1;
            int64_t best_kw = -1;
            int64_t best_channel = -1;

            for (int64_t i = 0; i < in_channels; i++)
            {
                T best_value_channel = Policy::initial_value();
                int64_t best_kh_channel = -1;
                int64_t best_kw_channel = -1;
                const T* input_channel = input_ptr + b * input_batch_stride + i * input_channel_stride;
                const T* weight_base = weight_ptr + o * weight_out_stride + i * weight_in_stride;

                for (int64_t kh = 0; kh < kernel_height; kh++)
                {
                    for (int64_t kw = 0; kw < kernel_width; kw++)
                    {
                        const int64_t pw = w + kernel_width_half - kw;
                        const int64_t ph = h + kernel_height_half - kh;

                        const bool h_valid = (ph >= 0 && ph < height);
                        const bool w_valid = (pw >= 0 && pw < width);

                        const T input_val = (h_valid && w_valid) ? input_channel[ph * width + pw] : Policy::padding_value();

                        const T val = Policy::combine(input_val, weight_base[kh * kernel_width + kw]);

                        if (Policy::is_better(val, best_value_channel))
                        {
                            best_value_channel = val;
                            best_kh_channel = kh;
                            best_kw_channel = kw;
                        }
                    }
                }
                if (MergePolicy::is_better(best_value_channel, best_value))
                {
                    best_value = best_value_channel;
                    best_kh = best_kh_channel;
                    best_kw = best_kw_channel;
                    best_channel = i;
                }
            }

            const int64_t ih = h + kernel_height_half - best_kh;
            const int64_t iw = w + kernel_width_half - best_kw;

            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
            {
                const int64_t grad_input_idx = b * input_batch_stride + best_channel * input_channel_stride + ih * width + iw;
#pragma omp atomic
                grad_input_ptr[grad_input_idx] += grad_val;
            }

            const int64_t grad_weight_idx = o * weight_out_stride + best_channel * weight_in_stride + best_kh * kernel_width + best_kw;

            if constexpr (op == MORPHO_OPERATION::DILATION)
            {
                local_grad_weight_ptr[grad_weight_idx] += grad_val;
            }
            else
            {
                local_grad_weight_ptr[grad_weight_idx] -= grad_val;
            }
        }
    }
}

/// @brief Backward pass for the morphological operation on CPU.
/// @tparam T The data type of the tensor elements.
/// @tparam op The morphological operation type (e.g., dilation, erosion).
/// @tparam merge The merge strategy for combining gradients.
/// @tparam save_index Whether to save the indices of the max values.
/// @param input The input tensor.
/// @param weight The weight tensor.
/// @param grad_output The gradient output tensor.
/// @param max_indices The indices of the max values.
/// @return The gradient tensors for the input and weight.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output,
                                                           torch::Tensor max_indices)
{
    const auto input_shape = input.sizes();
    const auto weight_shape = weight.sizes();

    const int64_t batch_size = input_shape[0];
    const int64_t in_channels = input_shape[1];
    const int64_t out_channels = weight_shape[0];
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];
    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;

    // Ensure contiguous memory layout for better cache performance
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto grad_output_contig = grad_output.contiguous();
    auto max_indices_contig = max_indices.contiguous();

    auto grad_input = torch::zeros_like(input_contig);
    auto grad_weight = torch::zeros_like(weight_contig);

    // Use raw pointers for better performance
    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ grad_output_ptr = grad_output_contig.template data_ptr<T>();
    T* __restrict__ grad_input_ptr = grad_input.template data_ptr<T>();

    // Pre-computed strides for contiguous tensors
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t grad_output_batch_stride = out_channels * height * width;
    const int64_t grad_output_channel_stride = height * width;

    const int64_t* __restrict__ indices_ptr = nullptr;
    int64_t indices_batch_stride = 0;
    int64_t indices_out_stride = 0;
    int64_t indices_in_stride = 0;

    if constexpr (save_index)
    {
        indices_ptr = max_indices_contig.template data_ptr<int64_t>();
        indices_batch_stride = out_channels * in_channels * height * width;
        indices_out_stride = in_channels * height * width;
        indices_in_stride = height * width;
    }

    // Thread-local storage to eliminate atomic operations
    const int num_threads = omp_get_max_threads();
    std::vector<torch::Tensor> local_grad_inputs(num_threads);
    std::vector<torch::Tensor> local_grad_weights(num_threads);

    for (int tid = 0; tid < num_threads; tid++)
    {
        local_grad_inputs[tid] = torch::zeros_like(grad_input);
        local_grad_weights[tid] = torch::zeros_like(grad_weight);
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        T* __restrict__ local_grad_input_ptr = local_grad_inputs[tid].template data_ptr<T>();
        T* __restrict__ local_grad_weight_ptr = local_grad_weights[tid].template data_ptr<T>();
#pragma omp for collapse(2)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                if constexpr (merge == MORPHO_MERGE::ADD || merge == MORPHO_MERGE::AVERAGE)
                {
                    morpho_backward_sum_avg<op, merge, T, save_index, false>(
                        b, o, in_channels, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half, input_ptr, weight_ptr,
                        grad_output_ptr, local_grad_input_ptr, local_grad_weight_ptr, input_batch_stride, input_channel_stride, weight_out_stride,
                        weight_in_stride, grad_output_batch_stride, grad_output_channel_stride);
                }
                else
                {
                    morpho_backward_min_max<op, merge, T, save_index, false>(
                        b, o, in_channels, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half, input_ptr, weight_ptr,
                        grad_output_ptr, local_grad_input_ptr, local_grad_weight_ptr, input_batch_stride, input_channel_stride, weight_out_stride,
                        weight_in_stride, grad_output_batch_stride, grad_output_channel_stride);
                }
            }
        }
    }

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
            grad_weight.template data_ptr<T>()[i] = sum;
        }
    }
    return {grad_input, grad_weight};
}