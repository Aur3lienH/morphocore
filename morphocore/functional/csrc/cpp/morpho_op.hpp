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

// The CUDA sources only need the declarations above. The CPU kernels use GCC vector extensions that
// the nvcc front end cannot parse, and the dispatching wrapper is compiled in morpho_op.cpp.
#ifndef __CUDACC__
    #include "morpho_padding_policy.hpp"
    #include "morpho_op_cpu_forward.hxx"
    #include "morpho_op_cpu_backward.hxx"
    #include "morpho_op_wrap.hxx"
#endif