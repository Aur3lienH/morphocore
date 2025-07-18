#pragma once 



enum class MORPHO_OPERATION {
    DILATION,
    EROSION
};


template<typename scalar_t>
struct MorphoPolicyCPU<scalar_t, MorphoOp::EROSION> {
    __forceinline__ static scalar_t initial_value() 
    {
        return std::numeric_limits<scalar_t>::max();
    }
    
    __forceinline__ static scalar_t combine(scalar_t input, scalar_t weight) 
    {
        return input - weight;
    }
    
    __forceinline__ static bool is_better(scalar_t new_val, scalar_t current_best) 
    {
        return new_val < current_best;
    }

    __forceinline__ static scalar_t padding_value() 
    {
        return -std::numeric_limits<scalar_t>::infinity();
    }

    __forceinline__ static int weight_gradient_sign() 
    {
        return 1;  
    }
};

template<MORPHO_OPERATION op, typename T>
torch::Tensor morpho_dilaton_forward_cpu(torch::Tensor input, torch::Tensor weight);

template<MORPHO_OPERATION op, typename T>
torch::autograd::tensor_list morpho_dilaton_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_output
);


template <MORPHO_OPERATION op>
class MorphoOpertion : public torch::autograd::Function<MorphoDilation<op>> 
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


#include "morpho_op.hxx"