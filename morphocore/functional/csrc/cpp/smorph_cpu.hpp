#pragma once
#include <torch/torch.h>
#include "morpho_op_enum.hpp"

template <MORPHO_MERGE merge, typename T>
auto smorph_forward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha);

template <MORPHO_MERGE merge, typename T>
void smorph_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor grad_output, torch::Tensor grad_input,
                         torch::Tensor grad_weight, torch::Tensor grad_alpha);

#include "smorph_cpu_backward.hxx"
#include "smorph_cpu_forward.hxx"