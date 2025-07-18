#pragma once 
#include <torch/torch.h>


template<typename scalar_t, MORPHO_OPERATION Op>
struct MorphoPolicy;

template<typename scalar_t, MORPHO_OPERATION::DILATION> 
{
    __device__ __forceinline__ static scalar_t initial_value() 
    {
        return std::numeric_limits<scalar_t>::lowest();
    }
    
    __device__ __forceinline__ static scalar_t combine(scalar_t input, scalar_t weight) 
    {
        return input + weight;
    }
    
    __device__ __forceinline__ static bool is_better(scalar_t new_val, scalar_t current_best) 
    {
        return new_val > current_best;
    }
}

template <MorphoOp Op, typename scalar_t>
void morpho_forward_kernel(const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int64_t* max_indexes,
    int batch_size,
    int in_channels,        
    int out_channels,       
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width);


template <MorphoOp Op, typename scalar_t>
void morpho_backward_kernel(const scalar_t* padded_input,
    const scalar_t* weight, 
    scalar_t* grad_input,
    scalar_t* grad_weight,
    const scalar_t* grad_output,
    const int64_t* max_indexes,
    int batch_size,
    int in_channels,               
    int out_channels,              
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width);