#pragma once
#include <torch/torch.h>

template <typename T>
auto smorph_scm_forward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor beta);

template <typename T>
void smorph_scm_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor beta, torch::Tensor grad_output,
                             torch::Tensor grad_input, torch::Tensor grad_weight, torch::Tensor grad_alpha, torch::Tensor grad_beta);

#include "smorph_cpu_forward_soft_merge.hxx"
#include "smorph_cpu_backward_soft_merge.hxx"
