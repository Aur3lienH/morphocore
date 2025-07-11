#pragma once
#include <torch/torch.h>

#define TILE_WIDTH 32
#define TILE_HEIGHT 8

class Smorph : public torch::autograd::Function<Smorph> 
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
