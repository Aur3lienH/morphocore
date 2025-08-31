#pragma once 
#include <iostream>
#include <limits>
#include <torch/torch.h>
#include "morpho_op_enum.hpp"


template<typename T, MORPHO_OPERATION op>
struct MorphoPolicyCPU {
    static inline __attribute__((always_inline)) constexpr T padding_value() {
        if constexpr (op == MORPHO_OPERATION::DILATION) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::infinity();
        }
    }
    
    static inline __attribute__((always_inline)) constexpr T initial_value() {
        if constexpr (op == MORPHO_OPERATION::DILATION) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::infinity();
        }
    }
    
    static inline __attribute__((always_inline)) T combine(T input_val, T weight_val) {
        if constexpr (op == MORPHO_OPERATION::DILATION) {
            return input_val + weight_val;
        } else {
            return input_val - weight_val;
        }
    }
    
    static inline __attribute__((always_inline)) bool is_better(T val, T best) {
        if constexpr (op == MORPHO_OPERATION::DILATION) {
            return val > best;
        } else {
            return val < best;
        }
    }
};


template<MORPHO_OPERATION op, typename T, bool save_index>
auto morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight);

template<MORPHO_OPERATION op, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_output,
    torch::Tensor max_indices
);


template <MORPHO_OPERATION op>
class MorphoOperation : public torch::autograd::Function<MorphoOperation<op>> 
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


#include "morpho_op_cpu.hxx"
#include "morpho_op_wrap.hxx"