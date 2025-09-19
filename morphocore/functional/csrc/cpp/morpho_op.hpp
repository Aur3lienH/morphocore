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

// Create an always false value for handling errors
template<auto V>
constexpr bool always_false = false;

template<typename T, MORPHO_MERGE merge_op>
struct MorphoMergePolicy 
{
    static inline __attribute__((__always_inline)) T get_init_value()
    {
        if constexpr (merge_op == MORPHO_MERGE::ADD)
        {
            return static_cast<T>(0);
        }
        else if constexpr (merge_op == MORPHO_MERGE::MAX)
        {
            return std::numeric_limits<T>::lowest();
        }
        else if constexpr (merge_op == MORPHO_MERGE::MIN)
        {
            return std::numeric_limits<T>::max();
        }
        else if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return static_cast<T>(0);
        }
        else
        {
            static_assert(always_false<merge_op>, "Invalid merge operation");
        }
    }

    static inline __attribute__((__always_inline)) T merge(T a, T b)
    {
        if constexpr (merge_op == MORPHO_MERGE::ADD)
        {
            return a + b;
        }
        else if constexpr (merge_op == MORPHO_MERGE::MAX)
        {
            return std::max(a, b);
        }
        else if constexpr (merge_op == MORPHO_MERGE::MIN)
        {
            return std::min(a, b);
        }
        else if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return a + b;
        }
        else
        {
            static_assert(always_false<merge_op>, "Invalid merge operation");
        }
    }

    static inline __attribute__((__always_inline)) T finalize(T a, int in_channels)
    {
        if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return a / static_cast<T>(in_channels);
        }
        else
        {
            return a;
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