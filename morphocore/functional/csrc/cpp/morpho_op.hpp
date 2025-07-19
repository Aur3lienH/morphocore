#pragma once 
#include <iostream>
#include <limits>
#include <torch/torch.h>
#include "morpho_op_enum.hpp"

template<typename scalar_t, MORPHO_OPERATION Op>
struct MorphoPolicyCPU;


template<typename scalar_t>
struct MorphoPolicyCPU<scalar_t, MORPHO_OPERATION::EROSION> {
    inline static scalar_t initial_value() 
    {
        return 10000;
    }
    
    inline static scalar_t combine(scalar_t input, scalar_t weight) 
    {
        return input - weight;
    }
    
    inline static bool is_better(scalar_t new_val, scalar_t current_best) 
    {
        return new_val < current_best;
    }

    inline static scalar_t padding_value() 
    {
        return std::numeric_limits<scalar_t>::infinity();
    }

    inline static int weight_gradient_sign() 
    {
        return -1;  
    }
};


template<typename scalar_t>
struct MorphoPolicyCPU<scalar_t, MORPHO_OPERATION::DILATION> {
    inline static scalar_t initial_value() 
    {
        return -1000;
    }
    
    inline static scalar_t combine(scalar_t input, scalar_t weight) 
    {
        return input + weight;
    }
    
    inline static bool is_better(scalar_t new_val, scalar_t current_best) 
    {
        return new_val > current_best;
    }

    inline static scalar_t padding_value() 
    {
        return -1000;
    }

    inline static int weight_gradient_sign() 
    {
        return 1;  
    }
};

template<MORPHO_OPERATION op, typename T>
torch::Tensor morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight);

template<MORPHO_OPERATION op, typename T>
torch::autograd::tensor_list morpho_operation_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_output
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