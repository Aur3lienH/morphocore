#include "smorph_scm.hpp"
#include "smorph_scm_cpu.hpp"
#include <torch/torch.h>
#include "reshaper.hpp"
#include "../cpp/morpho_op_enum.hpp"

#ifdef WITH_CUDA
    #include <c10/cuda/CUDAGuard.h>

    #include "../cuda/smorph.cuh"
#endif

torch::Tensor SmorphSCM::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor alpha,
                                 torch::Tensor beta)
{
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);
    [[maybe_unused]] auto input_shape = input_correct_dims.sizes();
    [[maybe_unused]] auto weight_shape = weight_correct_dims.sizes();

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        const c10::cuda::CUDAGuard device_guard(input.device());
        auto input_contig = input_correct_dims.contiguous();
        auto weight_contig = weight_correct_dims.contiguous();
        const auto alpha_pairs = smorph_pair_alphas<MORPHO_MERGE::SMOOTH>(alpha, MorphoGeometryCPU::make<float>(input_contig, weight_contig));
        const auto beta_contig = beta.contiguous();
        TORCH_CHECK(beta_contig.numel() == weight_shape[0], "Beta must have shape [out_channels]");

        torch::Tensor output = torch::empty({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smorph_scm_forward_cuda", [&] {
            smorph_forward_cuda<scalar_t>(input_contig, weight_contig, alpha_pairs, beta_contig, output, MORPHO_MERGE::SMOOTH);
        });
        // The backward pass reads the 4D output: the final value of the soft merge.
        ctx->save_for_backward({output, torch::Tensor(), torch::Tensor(), alpha, beta_contig, input_contig, weight_contig});
        return input_dim_manager.ToOriginalTensor(output);
#endif
    }
    else
    {
        torch::Tensor output;
        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "smorph_forward_cpu", [&] {
            output = smorph_forward_cpu<MORPHO_MERGE::SMOOTH, scalar_t>(input_correct_dims, weight_correct_dims, alpha, beta);
        });
        output = input_dim_manager.ToOriginalTensor(output);
        // The CPU backward recomputes everything from the input: the two intermediate slots stay empty.
        ctx->save_for_backward({output, torch::Tensor(), torch::Tensor(), alpha, beta, input_correct_dims, weight_correct_dims});
        return output;
    }
    return torch::Tensor();
}

torch::autograd::tensor_list SmorphSCM::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs)
{
    auto saved = ctx->get_saved_variables();
    auto output = saved[0];
    auto values = saved[1];
    auto probabilities = saved[2];
    auto alpha = saved[3];
    auto beta = saved[4];
    auto input = saved[5];
    auto weight = saved[6];
    torch::Tensor grad_input = torch::zeros_like(input, grad_outputs[0].options()).contiguous();
    torch::Tensor grad_weight = torch::zeros_like(weight, grad_outputs[0].options()).contiguous();
    torch::Tensor grad_alpha = torch::zeros_like(alpha, grad_outputs[0].options()).contiguous();
    torch::Tensor grad_beta = torch::zeros_like(beta, grad_outputs[0].options()).contiguous();

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        const c10::cuda::CUDAGuard device_guard(input.device());
        const auto alpha_pairs = smorph_pair_alphas<MORPHO_MERGE::SMOOTH>(alpha, MorphoGeometryCPU::make<float>(input, weight));
        torch::Tensor grad_alpha_pairs = torch::zeros_like(alpha_pairs);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smorph_scm_backward_cuda", [&] {
            smorph_backward_cuda<scalar_t>(input, weight, alpha_pairs, beta, output, grad_outputs[0].contiguous(), grad_input, grad_weight,
                                           grad_alpha_pairs, grad_beta, MORPHO_MERGE::SMOOTH);
        });
        // A single alpha shared by every pair receives the gradient of all of them.
        grad_alpha =
            grad_alpha.numel() == grad_alpha_pairs.numel() ? grad_alpha_pairs.view_as(grad_alpha) : grad_alpha_pairs.sum().view_as(grad_alpha);
        return {grad_input, grad_weight, grad_alpha, grad_beta};
#endif
    }
    else
    {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_backward_cpu", [&] {
            smorph_backward_cpu<MORPHO_MERGE::SMOOTH, scalar_t>(input, weight, alpha, grad_outputs[0], grad_input, grad_weight, grad_alpha, beta,
                                                                grad_beta);
        });
        return {grad_input, grad_weight, grad_alpha, grad_beta};
    }
    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}
