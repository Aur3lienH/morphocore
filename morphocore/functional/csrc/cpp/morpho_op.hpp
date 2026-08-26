#pragma once
#include <iostream>
#include <limits>

#include <torch/torch.h>

#include "morpho_op_enum.hpp"

/// @brief Morphological operation class implementing forward and backward passes (dispatcher).
/// @tparam op Morphological operation (dilation or erosion).
/// @tparam merge Merge channel policy.
/// @tparam kind Padding family that implements the requested mode, see padding_kind().
/// @note The exact mode and its constant are runtime arguments so that every value based padding
///       shares a single instantiation of this class.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind>
class MorphoOperation : public torch::autograd::Function<MorphoOperation<op, merge, kind>>
{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight, bool save_indices,
                                 MORPHO_PADDING padding, double padding_value);

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs);

    static std::string name()
    {
        if (op == MORPHO_OPERATION::DILATION)
            return "Dilation";
        else
            return "Erosion";
    }
};

template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
auto morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight, T fill);

// Not wired up yet, see morpho_op_specialized.hxx; it has no padding support.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index, int KERNEL_SIZE>
auto morpho_operation_forward_cpu_template(torch::Tensor input, torch::Tensor weight);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output,
                                                           torch::Tensor max_indices, T fill);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
auto morpho_operation_forward_cpu_dispatcher(torch::Tensor input, torch::Tensor weight, T fill);

template <MORPHO_OPERATION op, MORPHO_PADDING_KIND kind, typename T, bool save_index>
auto morpho_operation_forward_identity_cpu(torch::Tensor input, torch::Tensor weight, T fill);

template <MORPHO_OPERATION op, MORPHO_PADDING_KIND kind, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_identity_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output,
                                                                    torch::Tensor max_indices, T fill);

#include "morpho_padding_policy.hpp"
#include "morpho_op_cpu_forward.hxx"
#include "morpho_op_cpu_backward.hxx"
#include "morpho_op_cpu_forward_linear.hxx"
#include "morpho_op_cpu_backward_linear.hxx"
#include "morpho_op_specialized.hxx"
#include "morpho_op_wrap.hxx"