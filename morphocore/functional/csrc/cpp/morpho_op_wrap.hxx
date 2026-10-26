#pragma once
#include <torch/all.h>

#include "../cuda/morpho_op.cuh"
#include "morpho_op_enum.hpp"
#include "reshaper.hpp"

#ifdef WITH_CUDA
    #include <c10/cuda/CUDAGuard.h>
#endif

/// @brief Forward wrapper for morphological operations. (Choose between CPU and CUDA)
/// @tparam Op Morphological operation (dilation or erosion).
/// @tparam merge Merge channel policy.
/// @tparam kind Padding family that implements `padding`.
/// @param ctx context to save tensors for backward pass.
/// @param input Input tensor of shape [batch, in_channels, height, width].
/// @param weight Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
/// @param padding Padding mode requested by the caller.
/// @param padding_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes.
/// @return Output tensor of shape [batch, out_channels, height, width].
template <MORPHO_OPERATION Op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind>
torch::Tensor MorphoOperation<Op, merge, kind>::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight,
                                                        bool save_indices, MORPHO_PADDING padding, double padding_value)
{
    TORCH_CHECK(padding_kind(padding) == kind, "Padding mode does not belong to the instantiated padding family");
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);

    auto input_shape = input_correct_dims.sizes();
    auto weight_shape = weight_correct_dims.sizes();
    auto batch_size = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t out_channels = weight_shape[0];

    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;

    // Validation checks
    TORCH_CHECK(input_correct_dims.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    TORCH_CHECK(weight_correct_dims.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");
    TORCH_CHECK(input_shape[1] == weight_shape[1] || merge == MORPHO_MERGE::IDENTITY, "Input channels must match weight input channels");

    torch::Tensor output;
    torch::Tensor max_indices;
    const int MAX_GRID_DIM = 65535;
    bool should_save_indices = save_indices && (merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN) && out_channels * batch_size < MAX_GRID_DIM;

    if (input.is_cuda())
    {
#ifdef WITH_CUDA

        const c10::cuda::CUDAGuard device_guard(input.device());
        if constexpr (merge == MORPHO_MERGE::IDENTITY)
        {
            TORCH_CHECK(out_channels == in_channels, "Identity mode requires weight.shape[0] == input.shape[1]");
            TORCH_CHECK(weight_shape[1] == 1, "Identity mode requires weight.shape[1] == 1");
        }
        input_correct_dims = input_correct_dims.contiguous();
        weight_correct_dims = weight_correct_dims.contiguous();

        // Every output element is written by the kernel, and the border is resolved in the kernel too.
        output = torch::empty({batch_size, out_channels, height, width}, input_correct_dims.options());
        // Left undefined when not saved: an empty tensor would still cost an allocation per call.
        if (should_save_indices)
        {
            max_indices = torch::empty({batch_size, out_channels, height, width}, input.options().dtype(torch::kInt));
        }

        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "morpho_operation_forward_cuda", [&] {
            morpho_forward_cuda<Op, scalar_t>(input_correct_dims, weight_correct_dims, output, max_indices, merge,
                                              kind == MORPHO_PADDING_KIND::REPLICATE, morpho_fill_value<Op, scalar_t>(padding, padding_value));
        });

#else
        TORCH_CHECK(false, "CUDA not available for this build");
#endif
    }
    else
    {
        constexpr const char* cpu_operation_name = "morpho_operation_forward_cpu";

        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), cpu_operation_name, [&] {
            const scalar_t fill = morpho_fill_value<Op, scalar_t>(padding, padding_value);
            constexpr bool merge_supports_indices = (merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN);

            if constexpr (merge_supports_indices)
            {
                if (should_save_indices)
                {
                    std::tie(output, max_indices) =
                        morpho_operation_forward_cpu<Op, merge, kind, scalar_t, true>(input_correct_dims, weight_correct_dims, fill);
                    return;
                }
            }
            output = std::get<0>(morpho_operation_forward_cpu<Op, merge, kind, scalar_t, false>(input_correct_dims, weight_correct_dims, fill));
            max_indices = torch::empty({0}, torch::kInt);
        });
    }

    output = input_dim_manager.ToOriginalTensor(output);
    ctx->save_for_backward({input_correct_dims, weight_correct_dims, max_indices});
    // The padding mode is not a tensor, so it travels to backward through saved_data.
    ctx->saved_data["padding"] = static_cast<int64_t>(padding);
    ctx->saved_data["padding_value"] = padding_value;
    return output;
}

/// @brief Backward wrapper for morphological operations. (Choose between CPU and CUDA)
/// @tparam Op Morphological operation (dilation or erosion).
/// @tparam merge Merge channel policy.
/// @param ctx context to retrieve saved tensors from forward pass.
/// @param grad_outputs Gradient tensors from the next layer.
/// @return Gradient tensors for the input and weight.
template <MORPHO_OPERATION Op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind>
torch::autograd::tensor_list MorphoOperation<Op, merge, kind>::backward(torch::autograd::AutogradContext* ctx,
                                                                        torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto max_indices = saved[2];
    const auto padding = static_cast<MORPHO_PADDING>(ctx->saved_data["padding"].toInt());
    const double padding_value = ctx->saved_data["padding_value"].toDouble();
    auto grad_output = grad_outputs[0];
    grad_output = grad_output.contiguous();
    torch::Tensor grad_input;
    torch::Tensor grad_weight;
    bool indices_were_saved = max_indices.defined() && max_indices.numel() > 0;

    std::vector<int64_t> expected_grad_shape = input.sizes().vec();
    expected_grad_shape[1] = weight.sizes()[0];
    TORCH_CHECK(grad_output.sizes() == expected_grad_shape, "grad_output shape mismatch: expected [", expected_grad_shape, "] but got [",
                grad_output.sizes(), "]");

    TORCH_CHECK(grad_output.device() == input.device(), "grad_output must be on the same device as input");

    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), "grad_output must have the same dtype as input");

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        const c10::cuda::CUDAGuard device_guard(input.device());
        // Zeroed by morpho_backward_cuda, into which the kernels accumulate.
        grad_input = torch::empty_like(input);
        grad_weight = torch::empty_like(weight);

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_operation_backward_cuda", [&] {
            morpho_backward_cuda<Op, scalar_t>(input, weight, indices_were_saved ? max_indices : torch::Tensor(), grad_output, grad_input,
                                               grad_weight, merge, kind == MORPHO_PADDING_KIND::REPLICATE,
                                               morpho_fill_value<Op, scalar_t>(padding, padding_value));
        });

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
            const scalar_t fill = morpho_fill_value<Op, scalar_t>(padding, padding_value);
            constexpr bool merge_supports_indices = (merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN);

            if constexpr (merge_supports_indices)
            {
                if (indices_were_saved)
                {
                    result = morpho_operation_backward_cpu<Op, merge, kind, scalar_t, true>(input, weight, grad_output, max_indices, fill);
                    return;
                }
            }
            result = morpho_operation_backward_cpu<Op, merge, kind, scalar_t, false>(input, weight, grad_output, max_indices, fill);
        });

        grad_input = result[0];
        grad_weight = result[1];
    }

    // One undefined gradient per non differentiable forward argument: save_indices, padding, padding_value.
    return {grad_input, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}