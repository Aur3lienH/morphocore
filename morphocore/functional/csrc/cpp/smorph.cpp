#include "smorph.hpp"
#include "smorph_cpu.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include <torch/all.h>
#include "reshaper.hpp"

#ifdef WITH_CUDA
    #include <c10/cuda/CUDAGuard.h>

    #include "../cuda/smorph.cuh"
#endif

/// @brief Forward pass of the Soft Morphological layer (dispatcher between CPU and CUDA).
/// @tparam merge Merge channel strategy to use.
/// @param ctx Autograd context to save tensors for backward pass
/// @param input Input tensor of shape (batch_size, in_channels, height, width)
/// @param weight Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
/// @param alpha Learnable parameter controlling the operation either dilation or erosion or something between the two
/// @return Output tensor of shape (batch_size, out_channels, height, width)
template <MORPHO_MERGE merge>
torch::Tensor Smorph<merge>::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor alpha)
{
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);
    [[maybe_unused]] auto input_shape = input_correct_dims.sizes();
    [[maybe_unused]] auto weight_shape = weight_correct_dims.sizes();

    TORCH_CHECK(input_correct_dims.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    TORCH_CHECK(weight_correct_dims.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        const c10::cuda::CUDAGuard device_guard(input.device());
        auto input_contig = input_correct_dims.contiguous();
        auto weight_contig = weight_correct_dims.contiguous();
        if constexpr (merge == MORPHO_MERGE::IDENTITY)
        {
            TORCH_CHECK(weight_shape[0] == input_shape[1], "Identity mode requires weight.shape[0] == input.shape[1]");
            TORCH_CHECK(weight_shape[1] == 1, "Identity mode requires weight.shape[1] == 1, got ", weight_shape[1]);
        }
        const auto alpha_pairs = smorph_pair_alphas<merge>(alpha, MorphoGeometryCPU::make<float>(input_contig, weight_contig));

        torch::Tensor output = torch::empty({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smorph_forward_cuda",
                                   [&] { smorph_forward_cuda<scalar_t>(input_contig, weight_contig, alpha_pairs, torch::Tensor(), output, merge); });
        // The backward pass reads the 4D output: the merge target of MAX and MIN.
        ctx->save_for_backward({output, torch::Tensor(), torch::Tensor(), alpha, input_contig, weight_contig});
        return input_dim_manager.ToOriginalTensor(output);
#endif
    }
    else
    {
        torch::Tensor output;
        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "smorph_forward_cpu",
                                   [&] { output = smorph_forward_cpu<merge, scalar_t>(input_correct_dims, weight_correct_dims, alpha); });
        output = input_dim_manager.ToOriginalTensor(output);
        // The CPU backward recomputes everything from the input: the two intermediate slots stay empty.
        ctx->save_for_backward({output, torch::Tensor(), torch::Tensor(), alpha, input_correct_dims, weight_correct_dims});
        return output;
    }
    return torch::Tensor();
}

/// @brief Backward pass of the Soft Morphological layer
/// @tparam merge Merge strategy to use
/// @param ctx Autograd context to retrieve saved tensors from forward pass
/// @param grad_outputs Gradient of the output tensor
/// @return Gradients of the input, weight and alpha tensors
template <MORPHO_MERGE merge>
torch::autograd::tensor_list Smorph<merge>::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs)
{

    auto saved = ctx->get_saved_variables();
    auto output = saved[0];
    auto values = saved[1];
    auto probabilities = saved[2];
    auto alpha = saved[3];
    auto input = saved[4];
    auto weight = saved[5];
    torch::Tensor grad_input = torch::zeros_like(input, grad_outputs[0].options()).contiguous();
    torch::Tensor grad_weight = torch::zeros_like(weight, grad_outputs[0].options()).contiguous();
    torch::Tensor grad_alpha = torch::zeros_like(alpha, grad_outputs[0].options()).contiguous();
    if (grad_outputs[0].is_cuda())
    {
#ifdef WITH_CUDA
        const c10::cuda::CUDAGuard device_guard(input.device());
        const auto alpha_pairs = smorph_pair_alphas<merge>(alpha, MorphoGeometryCPU::make<float>(input, weight));
        torch::Tensor grad_alpha_pairs = torch::zeros_like(alpha_pairs);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smorph_backward_cuda", [&] {
            smorph_backward_cuda<scalar_t>(input, weight, alpha_pairs, torch::Tensor(), output, grad_outputs[0].contiguous(), grad_input, grad_weight,
                                           grad_alpha_pairs, torch::Tensor(), merge);
        });
        // A single alpha shared by every pair receives the gradient of all of them.
        grad_alpha =
            grad_alpha.numel() == grad_alpha_pairs.numel() ? grad_alpha_pairs.view_as(grad_alpha) : grad_alpha_pairs.sum().view_as(grad_alpha);
        return {grad_input, grad_weight, grad_alpha};
#endif
    }

    else
    {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_backward_cpu", [&] {
            smorph_backward_cpu<merge, scalar_t>(input, weight, alpha, grad_outputs[0], grad_input, grad_weight, grad_alpha);
        });
        return {grad_input, grad_weight, grad_alpha};
    }
    return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

template class Smorph<MORPHO_MERGE::MIN>;
template class Smorph<MORPHO_MERGE::MAX>;
template class Smorph<MORPHO_MERGE::AVERAGE>;
template class Smorph<MORPHO_MERGE::ADD>;
template class Smorph<MORPHO_MERGE::IDENTITY>;
