#include "smorph_scm.hpp"
#include "smorph_scm_cpu.hpp"
#include <torch/torch.h>
#include "reshaper.hpp"
#include "../cpp/morpho_op_enum.hpp"

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
        // TODO: CUDA implementation
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
        // TODO: CUDA implementation
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
