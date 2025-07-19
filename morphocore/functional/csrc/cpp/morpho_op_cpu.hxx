#pragma once
#include "morpho_op.hpp"


template<MORPHO_OPERATION op, typename T>
torch::Tensor morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight) 
{

    using Policy = MorphoPolicyCPU<T, op>;


    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    
    auto batch_size = input_shape[0];
    auto in_channels = input_shape[1];
    auto out_channels = weight_shape[0];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;
    torch::Tensor output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    auto padded_input = torch::constant_pad_nd(
        input, 
        torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, 
        Policy::padding_value()
    );
    
    // Get tensor accessors for efficient access
    auto output_accessor = output.template accessor<T, 4>();
    auto padded_input_accessor = padded_input.template accessor<T, 4>();
    auto weight_accessor = weight.template accessor<T, 4>();
    
    for (int64_t b = 0; b < batch_size; b++) 
    {
        for(int64_t o = 0; o < out_channels; o++)
        {
            for(int64_t i = 0; i < in_channels; i++)
            {
                //#pragma omp parallel for simd
                for (int64_t h = 0; h < height; h++) 
                {
                    for (int64_t w = 0; w < width; w++) 
                    {

                        T best_value = Policy::initial_value();
                        int64_t max_flat_index = 0;

                        for (int64_t kh = 0; kh < kernel_height; kh++) 
                        {
                            for (int64_t kw = 0; kw < kernel_width; kw++) 
                            {
                                int64_t ph = h - kh + 2 * kernel_height_half;
                                int64_t pw = w - kw + 2 * kernel_width_half;
                                
                                T val = Policy::combine(padded_input_accessor[b][i][ph][pw], weight_accessor[o][i][kh][kw]);
                                
                                int64_t flat_index = kh * kernel_width + kw;
                                if (Policy::is_better(val, best_value)) 
                                {
                                    max_flat_index = flat_index;
                                    best_value = val;
                                }
                            }
                        }
                        
                        output_accessor[b][o][h][w] += best_value;
                    }
                }
            }
            
        }
        
    }
    
    return output;
}


template<MORPHO_OPERATION op ,typename T>
torch::autograd::tensor_list morpho_operation_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_output) {
    
    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    
    auto batch_size = input_shape[0];
    auto in_channels = input_shape[1]; 
    auto out_channels = weight_shape[0];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;
    
    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    
    auto padded_input = torch::constant_pad_nd(
        input, 
        torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, 
        MorphoPolicyCPU<T, op>::padding_value()
    );
    
    auto grad_input_accessor = grad_input.template accessor<T, 4>();
    auto grad_weight_accessor = grad_weight.template accessor<T, 4>();
    auto padded_input_accessor = padded_input.template accessor<T, 4>();
    auto weight_accessor = weight.template accessor<T, 4>();
    auto grad_output_accessor = grad_output.template accessor<T, 4>();
    
    for (int64_t b = 0; b < batch_size; b++) {
        auto local_grad_weight = torch::zeros_like(weight);
        auto local_grad_weight_accessor = local_grad_weight.template accessor<T, 4>();
        
        for(int64_t o = 0; o < out_channels; o++) {          
            for(int64_t i = 0; i < in_channels; i++) {       
                //#pragma omp parallel for simd
                for (int64_t h = 0; h < height; h++) {
                    for (int64_t w = 0; w < width; w++) {
                    
                        T grad_val = grad_output_accessor[b][o][h][w];
                        
                        T max_value = -std::numeric_limits<T>::infinity();
                        int64_t max_kh = 0, max_kw = 0;
                        
                        for (int64_t kh = 0; kh < kernel_height; kh++) {
                            for (int64_t kw = 0; kw < kernel_width; kw++) {
                                int64_t ph = h - kh + 2 * kernel_height_half;
                                int64_t pw = w - kw + 2 * kernel_width_half;
                                
                                T val = padded_input_accessor[b][i][ph][pw] + weight_accessor[o][i][kh][kw];
                                
                                if (val > max_value) {
                                    max_value = val;
                                    max_kh = kh;
                                    max_kw = kw;
                                }
                            }
                        }
                        
                        int64_t ih = h - max_kh + kernel_height_half;
                        int64_t iw = w - max_kw + kernel_width_half;
                        
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) 
                        {
                            grad_input_accessor[b][i][ih][iw] += grad_val;  
                        }
                        
                        if constexpr (op == MORPHO_OPERATION::DILATION) 
                        {
                            local_grad_weight_accessor[o][i][max_kh][max_kw] += grad_val;
                        } else 
                        { 
                            local_grad_weight_accessor[o][i][max_kh][max_kw] -= grad_val;
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            grad_weight += local_grad_weight;
        }
    }
    
    return {grad_input, grad_weight};
}