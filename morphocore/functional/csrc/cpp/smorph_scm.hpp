#pragma once
#include <torch/torch.h>
#include "morpho_op_enum.hpp"

/// @brief Autograd Function for the Soft Morphological layer with Smooth Channel Merge.
class SmorphSCM : public torch::autograd::Function<SmorphSCM>
{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor alpha,
                                 torch::Tensor beta);

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs);

    static std::string name() { return "SmorphSCM"; }
};
