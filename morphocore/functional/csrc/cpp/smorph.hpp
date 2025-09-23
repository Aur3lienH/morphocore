#pragma once
#include <torch/torch.h>
#include "morpho_op_enum.hpp"


template <MORPHO_MERGE merge>
class Smorph : public torch::autograd::Function<Smorph<merge>> 
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor alpha
    );

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    );
};
