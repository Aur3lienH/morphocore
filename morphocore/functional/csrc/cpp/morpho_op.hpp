#pragma once
#include <iostream>
#include <limits>

#include <torch/torch.h>

#include "morpho_op_enum.hpp"

template <MORPHO_OPERATION op, MORPHO_MERGE merge>
class MorphoOperation : public torch::autograd::Function<MorphoOperation<op, merge>>
{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight);

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs);
};

template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
auto morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index, int KERNEL_SIZE>
auto morpho_operation_forward_cpu_template(torch::Tensor input, torch::Tensor weight);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output,
                                                           torch::Tensor max_indices);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
auto morpho_operation_forward_cpu_dispatcher(torch::Tensor input, torch::Tensor weight);

#include "morpho_op_cpu_forward.hxx"
#include "morpho_op_cpu_backward.hxx"
#include "morpho_op_specialized.hxx"
#include "morpho_op_wrap.hxx"