#pragma once
#include <torch/torch.h>

#include "../cuda/morpho_op.cuh"
#include "morpho_op_enum.hpp"
#include "reshaper.hpp"

#if WITH_CUDA
    #include <cuda_runtime.h>
#endif

/// @brief Get the padding value for a given morphological operation and data type.
/// @tparam Op Morphological operation type (e.g., dilation, erosion).
/// @param dtype Data type of the tensor elements.
/// @return Padding value as a scalar.
template <MORPHO_OPERATION Op>
torch::Scalar get_padding_value(torch::ScalarType dtype)
{
    // Using big value and not infinity to avoid overflow
    if constexpr (Op == MORPHO_OPERATION::DILATION)
    {
        if (dtype == torch::kFloat)
        {
            return torch::Scalar(-1e10f);
        }
        else if (dtype == torch::kDouble)
        {
            return torch::Scalar(-1e10);
        }
    }
    else
    {
        if (dtype == torch::kFloat)
        {
            return torch::Scalar(1e10f);
        }
        else if (dtype == torch::kDouble)
        {
            return torch::Scalar(1e10);
        }
    }

    TORCH_CHECK(false, "Unsupported dtype for morphological operation");
    return torch::Scalar(0);
}

/// @brief Border value the CUDA path materialises around the input, mirroring morpho_fill_value.
/// @tparam Op Morphological operation, only used to pick the geodesic neutral element.
/// @param padding Padding mode requested by the caller.
/// @param constant_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes.
/// @param dtype Data type of the tensor elements.
template <MORPHO_OPERATION Op>
torch::Scalar morpho_fill_scalar(MORPHO_PADDING padding, double constant_value, torch::ScalarType dtype)
{
    switch (padding)
    {
        case MORPHO_PADDING::ZERO: return torch::Scalar(0.0);
        case MORPHO_PADDING::CONSTANT: return torch::Scalar(constant_value);
        // GEODESIC, and the value REPLICATE never reads.
        default: return get_padding_value<Op>(dtype);
    }
}

/// @brief Materialise the border so the CUDA kernels can read a plain padded buffer and stay
///        completely unaware of the padding mode.
/// @tparam kind Padding family.
/// @param pad_h Rows added on the top and on the bottom.
/// @param pad_w Columns added on the left and on the right.
/// @param fill Fill value from morpho_fill_scalar, unused when kind is REPLICATE.
template <MORPHO_PADDING_KIND kind>
torch::Tensor morpho_pad_input(const torch::Tensor& input, int64_t pad_h, int64_t pad_w, const torch::Scalar& fill)
{
    if constexpr (kind == MORPHO_PADDING_KIND::REPLICATE)
    {
        const int64_t height = input.size(2);
        const int64_t width = input.size(3);

        // replication_pad2d is a single fused kernel but rejects pads larger than the image, which
        // happens for structuring elements wider than the input. A clamped gather covers that case.
        if (pad_h < height && pad_w < width)
        {
            return at::replication_pad2d(input, {pad_w, pad_w, pad_h, pad_h});
        }

        const auto index_options = input.options().dtype(torch::kLong);
        auto rows = torch::arange(-pad_h, height + pad_h, index_options).clamp_(0, height - 1);
        auto cols = torch::arange(-pad_w, width + pad_w, index_options).clamp_(0, width - 1);
        return input.index_select(2, rows).index_select(3, cols);
    }

    return torch::constant_pad_nd(input, torch::IntList{pad_w, pad_w, pad_h, pad_h}, fill);
}

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

        input_correct_dims = input_correct_dims.contiguous();
        weight_correct_dims = weight_correct_dims.contiguous();

        output = torch::zeros({batch_size, out_channels, height, width}, input_correct_dims.options());

        if (should_save_indices)
        {
            max_indices = torch::zeros({batch_size, out_channels, height, width}, input.options().dtype(torch::kLong));
        }
        else
        {
            max_indices = torch::empty({0}, input.options().dtype(torch::kLong));
        }

        auto fill = morpho_fill_scalar<Op>(padding, padding_value, input_correct_dims.scalar_type());
        auto padded_input = morpho_pad_input<kind>(input_correct_dims, kernel_height_half, kernel_width_half, fill);

        padded_input = padded_input.contiguous();

        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;

        constexpr const char* operation_name = (Op == MORPHO_OPERATION::DILATION) ? "morpho_dilation_forward" : "morpho_erosion_forward";

        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), operation_name, [&] {
            constexpr bool merge_supports_indices = (merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN);

            if constexpr (merge_supports_indices)
            {
                if (should_save_indices)
                {
                    morpho_forward_kernel<Op, merge, scalar_t, true>(
                        padded_input.template data_ptr<scalar_t>(), weight_correct_dims.template data_ptr<scalar_t>(),
                        output.template data_ptr<scalar_t>(), max_indices.template data_ptr<int64_t>(), batch_size, in_channels, out_channels, height,
                        width, padded_height, padded_width, kernel_height, kernel_width);
                }
                else
                {
                    morpho_forward_kernel<Op, merge, scalar_t, false>(
                        padded_input.template data_ptr<scalar_t>(), weight_correct_dims.template data_ptr<scalar_t>(),
                        output.template data_ptr<scalar_t>(), nullptr, batch_size, in_channels, out_channels, height, width, padded_height,
                        padded_width, kernel_height, kernel_width);
                }
            }
            else
            {
                morpho_forward_kernel<Op, merge, scalar_t, false>(
                    padded_input.template data_ptr<scalar_t>(), weight_correct_dims.template data_ptr<scalar_t>(),
                    output.template data_ptr<scalar_t>(), nullptr, batch_size, in_channels, out_channels, height, width, padded_height, padded_width,
                    kernel_height, kernel_width);
            }
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
            const scalar_t fill = morpho_fill_value<Op, scalar_t>(padding, padding_value);

            if (should_save_indices)
            {
                auto result =
                    morpho_operation_forward_cpu_dispatcher<Op, merge, kind, scalar_t, true>(input_correct_dims, weight_correct_dims, fill);
                output = std::get<0>(result);
                max_indices = std::get<1>(result);
            }
            else
            {
                output = morpho_operation_forward_cpu_dispatcher<Op, merge, kind, scalar_t, false>(input_correct_dims, weight_correct_dims, fill);
                max_indices = torch::empty({0}, torch::kLong);
            }
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
    bool indices_were_saved = max_indices.numel() > 0;

    std::vector<int64_t> expected_grad_shape = input.sizes().vec();
    expected_grad_shape[1] = weight.sizes()[0];
    TORCH_CHECK(grad_output.sizes() == expected_grad_shape, "grad_output shape mismatch: expected [", expected_grad_shape, "] but got [",
                grad_output.sizes(), "]");

    TORCH_CHECK(grad_output.device() == input.device(), "grad_output must be on the same device as input");

    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), "grad_output must have the same dtype as input");

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
        grad_input.contiguous();
        grad_weight.contiguous();

        // Recreate padded input for backward pass
        auto fill = morpho_fill_scalar<Op>(padding, padding_value, input.scalar_type());
        auto padded_input = morpho_pad_input<kind>(input, kernel_height_half, kernel_width_half, fill);

        // Ensure padded input is contiguous
        padded_input = padded_input.contiguous();

        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;

        constexpr const char* operation_name = "morpho_backward_kernel";

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), operation_name, [&] {
            // Calculate grid dimensions
            dim3 threads(16, 16);
            dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, batch_size);
            constexpr bool merge_supports_indices = (merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN);

            if constexpr (merge_supports_indices)
            {
                if (indices_were_saved)
                {
                    morpho_backward_kernel<Op, merge, scalar_t, true>(
                        padded_input.template data_ptr<scalar_t>(), weight.template data_ptr<scalar_t>(), grad_input.template data_ptr<scalar_t>(),
                        grad_weight.template data_ptr<scalar_t>(), grad_output.template data_ptr<scalar_t>(), batch_size, in_channels, out_channels,
                        height, width, padded_height, padded_width, kernel_height, kernel_width, max_indices.template data_ptr<int64_t>());
                }
                else
                {
                    morpho_backward_kernel<Op, merge, scalar_t, false>(
                        padded_input.template data_ptr<scalar_t>(), weight.template data_ptr<scalar_t>(), grad_input.template data_ptr<scalar_t>(),
                        grad_weight.template data_ptr<scalar_t>(), grad_output.template data_ptr<scalar_t>(), batch_size, in_channels, out_channels,
                        height, width, padded_height, padded_width, kernel_height, kernel_width, nullptr);
                }
            }
            else
            {
                morpho_backward_kernel<Op, merge, scalar_t, false>(
                    padded_input.template data_ptr<scalar_t>(), weight.template data_ptr<scalar_t>(), grad_input.template data_ptr<scalar_t>(),
                    grad_weight.template data_ptr<scalar_t>(), grad_output.template data_ptr<scalar_t>(), batch_size, in_channels, out_channels,
                    height, width, padded_height, padded_width, kernel_height, kernel_width, nullptr);
            }
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
            const scalar_t fill = morpho_fill_value<Op, scalar_t>(padding, padding_value);

            if (indices_were_saved)
            {
                result = morpho_operation_backward_cpu_dispatch<Op, merge, kind, scalar_t, true>(input, weight, grad_output, max_indices, fill);
            }
            else
            {
                result = morpho_operation_backward_cpu_dispatch<Op, merge, kind, scalar_t, false>(input, weight, grad_output, max_indices, fill);
            }
        });

        grad_input = result[0];
        grad_weight = result[1];
    }

    // One undefined gradient per non differentiable forward argument: save_indices, padding, padding_value.
    return {grad_input, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}