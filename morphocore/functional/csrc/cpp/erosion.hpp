#pragma once 
#include <torch/torch.h>

class MorphoErosion : public torch::autograd::Function<MorphoErosion> 
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight
    );

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    );
};

torch::Tensor morpho_dilaton_forward_cpu(torch::Tensor input, torch::Tensor weight);

torch::autograd::tensor_list morpho_dilaton_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor min_indexes,
    torch::Tensor grad_output);

