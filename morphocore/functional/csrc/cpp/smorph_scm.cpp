#include "smorph_scm.hpp"
#include "smorph_scm_cpu.hpp"
#include <torch/torch.h>
#include "reshaper.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "smorph_cpu_forward_soft_merge.hxx"
#include "smorph_cpu_backward_soft_merge.hxx"

torch::Tensor SmorphSCM::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor alpha,
                                 torch::Tensor beta)
{
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);
    auto input_shape = input_correct_dims.sizes();
    auto weight_shape = weight_correct_dims.sizes();

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        // TODO: CUDA implementation
#endif
    }
    else
    {
        torch::Tensor output = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        torch::Tensor reference_values = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        torch::Tensor max_values = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "smorph_forward_cpu",
                                   [&] { output = smorph_scm_forward_cpu<scalar_t>(input_correct_dims, weight_correct_dims, alpha, beta); });
        output = input_dim_manager.ToOriginalTensor(output);
        ctx->save_for_backward({output, reference_values, max_values, alpha, beta, input_correct_dims, weight_correct_dims});
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
    torch::Tensor grad_input = torch::zeros_like(input, grad_outputs[0].options());
    torch::Tensor grad_weight = torch::zeros_like(weight, grad_outputs[0].options());
    torch::Tensor grad_alpha = torch::zeros_like(alpha, grad_outputs[0].options());
    torch::Tensor grad_beta = torch::zeros_like(beta, grad_outputs[0].options());

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        // TODO: CUDA implementation
#endif
    }
    else
    {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_backward_cpu", [&] {
            smorph_scm_backward_cpu<scalar_t>(input, weight, alpha, beta, grad_outputs[0], grad_input, grad_weight, grad_alpha, grad_beta);
        });
        return {grad_input, grad_weight, grad_alpha, grad_beta};
    }
    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}
