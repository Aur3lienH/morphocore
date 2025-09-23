#pragma once
#include <torch/torch.h>

#include "../cuda/morpho_op.cuh"
#include "morpho_op_enum.hpp"
#include "reshaper.hpp"

#if WITH_CUDA
    #include <cuda_runtime.h>
#endif

inline bool save_indices(torch::Tensor input, torch::Tensor weight)
{
    return false;
}

template <MORPHO_OPERATION Op>
torch::Scalar get_padding_value(torch::ScalarType dtype)
{
    if constexpr (Op == MORPHO_OPERATION::DILATION)
    {
        // For dilation, use negative infinity as padding
        if (dtype == torch::kFloat)
        {
            return torch::Scalar(-std::numeric_limits<float>::infinity());
        }
        else if (dtype == torch::kDouble)
        {
            return torch::Scalar(-std::numeric_limits<double>::infinity());
        }
    }
    else
    {  // EROSION
        // For erosion, use positive infinity as padding
        if (dtype == torch::kFloat)
        {
            return torch::Scalar(std::numeric_limits<float>::infinity());
        }
        else if (dtype == torch::kDouble)
        {
            return torch::Scalar(std::numeric_limits<double>::infinity());
        }
    }

    TORCH_CHECK(false, "Unsupported dtype for morphological operation");
    return torch::Scalar(0);
}

// CORRECTED FORWARD PASS:
template <MORPHO_OPERATION Op, MORPHO_MERGE merge>
torch::Tensor MorphoOperation<Op, merge>::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight)
{
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);

    auto input_shape = input_correct_dims.sizes();
    auto weight_shape = weight_correct_dims.sizes();
    auto batch_size = input_shape[0];
    auto in_channels = input_shape[1];
    auto out_channels = weight_shape[0];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;

    // Validation checks
    // TORCH_CHECK(input.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    // TORCH_CHECK(weight.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");
    // TORCH_CHECK(input_shape[1] == weight_shape[1], "Input channels must match weight input channels");

    torch::Tensor output;
    torch::Tensor max_indices;

    if (input.is_cuda())
    {
#ifdef WITH_CUDA

        output = torch::zeros({batch_size, out_channels, height, width}, input_correct_dims.options());
        max_indices = torch::zeros({batch_size, out_channels, in_channels, height, width}, input.options().dtype(torch::kLong));

        auto padding_value = get_padding_value<Op>(input_correct_dims.scalar_type());
        auto padded_input = torch::constant_pad_nd(
            input_correct_dims, torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, padding_value);

        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;

        constexpr const char* operation_name = (Op == MORPHO_OPERATION::DILATION) ? "morpho_dilation_forward" : "morpho_erosion_forward";

        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), operation_name, [&] {
            morpho_forward_kernel<Op, scalar_t>(padded_input.template data_ptr<scalar_t>(), weight_correct_dims.template data_ptr<scalar_t>(),
                                                output.template data_ptr<scalar_t>(), max_indices.template data_ptr<int64_t>(), batch_size,
                                                in_channels, out_channels, height, width, padded_height, padded_width, kernel_height, kernel_width);
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }

#else
        TORCH_CHECK(false, "CUDA not available for this build");
#endif
    }
    else
    {
        constexpr const char* cpu_operation_name = "morpho_operation_forward_cpu";

        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), cpu_operation_name, [&] {
            if (save_indices(input, weight))
            {
                auto result = morpho_operation_forward_cpu_dispatcher<Op, merge, scalar_t, true>(input_correct_dims, weight_correct_dims);
                output = std::get<0>(result);
                max_indices = std::get<1>(result);
            }
            else
            {
                output = morpho_operation_forward_cpu_dispatcher<Op, merge, scalar_t, false>(input_correct_dims, weight_correct_dims);
                max_indices = torch::empty({0}, torch::kLong);
            }
        });
    }

    output = input_dim_manager.ToOriginalTensor(output);
    ctx->save_for_backward({input_correct_dims, weight_correct_dims, max_indices});
    return output;
}

template <MORPHO_OPERATION Op, MORPHO_MERGE merge>
torch::autograd::tensor_list MorphoOperation<Op, merge>::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto max_indices = saved[2];
    auto grad_output = grad_outputs[0];

    torch::Tensor grad_input;
    torch::Tensor grad_weight;
    bool indices_were_saved = max_indices.numel() > 0;

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        auto input_shape = input.sizes();
        auto weight_shape = weight.sizes();
        auto batch_size = input_shape[0];
        auto in_channels = input_shape[1];
        auto out_channels = weight_shape[0];
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;

        // Create gradient tensors
        grad_input = torch::zeros_like(input);
        grad_weight = torch::zeros_like(weight);

        // Recreate padded input for backward pass
        auto padding_value = get_padding_value<Op>(input.scalar_type());
        auto padded_input = torch::constant_pad_nd(
            input, torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, padding_value);

        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;

        constexpr const char* operation_name = "morpho_backward_kernel";

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), operation_name, [&] {
            morpho_backward_kernel<Op, scalar_t>(padded_input.template data_ptr<scalar_t>(), weight.template data_ptr<scalar_t>(),
                                                 grad_input.template data_ptr<scalar_t>(), grad_weight.template data_ptr<scalar_t>(),
                                                 grad_output.template data_ptr<scalar_t>(), max_indices.template data_ptr<int64_t>(), batch_size,
                                                 in_channels, out_channels, height, width, padded_height, padded_width, kernel_height, kernel_width);
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }

#else
        TORCH_CHECK(false, "CUDA not available for this build");
#endif
    }
    else
    {
        constexpr const char* cpu_operation_name =
            (Op == MORPHO_OPERATION::DILATION) ? "morpho_dilation_backward_cpu" : "morpho_erosion_backward_cpu";

        torch::autograd::tensor_list result;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), cpu_operation_name, [&] {
            if (indices_were_saved)
            {
                result = morpho_operation_backward_cpu<Op, merge, scalar_t, true>(input, weight, grad_output, max_indices);
            }
            else
            {
                result = morpho_operation_backward_cpu<Op, merge, scalar_t, false>(input, weight, grad_output, max_indices);
            }
        });

        grad_input = result[0];
        grad_weight = result[1];
    }

    return {grad_input, grad_weight};
}