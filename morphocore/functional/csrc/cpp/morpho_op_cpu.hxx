#pragma once
#include "morpho_op.hpp"
#include <omp.h>
#include <vector>
#include <immintrin.h>
#include <memory>


inline thread_local struct ThreadLocalStorage {
    std::vector<float> temp_buffer;
    std::vector<int64_t> indices_buffer;
    bool initialized = false;
    
    void ensure_capacity(size_t size) {
        if (!initialized || temp_buffer.size() < size) {
            temp_buffer.resize(size);
            indices_buffer.resize(size);
            initialized = true;
        }
    }
} tls_storage;




template<MORPHO_OPERATION op, typename T, bool save_index>
auto morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight) 
{
    using Policy = MorphoPolicyCPU<T, op>;

    const auto input_shape = input.sizes();
    const auto weight_shape = weight.sizes();
    
    const int64_t batch_size = input_shape[0];
    const int64_t in_channels = input_shape[1];
    const int64_t out_channels = weight_shape[0];
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];
    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;
    
    // Ensure tensors are contiguous for optimal memory access
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    
    torch::Tensor output = torch::zeros({batch_size, out_channels, height, width}, 
                                       input_contig.options());
    
    const auto padded_input = torch::constant_pad_nd(
        input_contig, 
        torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, 
        Policy::padding_value()
    );
    
    // Get raw data pointers for maximum performance
    const T* input_ptr = padded_input.template data_ptr<T>();
    const T* weight_ptr = weight_contig.template data_ptr<T>();
    T* output_ptr = output.template data_ptr<T>();
    
    // Tensor dimensions for pointer arithmetic
    const int64_t input_stride_b = padded_input.stride(0);
    const int64_t input_stride_c = padded_input.stride(1);
    const int64_t input_stride_h = padded_input.stride(2);
    const int64_t input_stride_w = padded_input.stride(3);
    
    const int64_t weight_stride_o = weight_contig.stride(0);
    const int64_t weight_stride_i = weight_contig.stride(1);
    const int64_t weight_stride_kh = weight_contig.stride(2);
    const int64_t weight_stride_kw = weight_contig.stride(3);
    
    const int64_t output_stride_b = output.stride(0);
    const int64_t output_stride_o = output.stride(1);
    const int64_t output_stride_h = output.stride(2);
    const int64_t output_stride_w = output.stride(3);

    torch::Tensor max_indices;
    int64_t* indices_ptr = nullptr;
    int64_t indices_stride_b = 0, indices_stride_o = 0, indices_stride_i = 0, 
            indices_stride_h = 0, indices_stride_w = 0;
    
    if constexpr (save_index) {
        max_indices = torch::zeros({batch_size, out_channels, in_channels, height, width}, 
                                 torch::kLong);
        indices_ptr = max_indices.template data_ptr<int64_t>();
        indices_stride_b = max_indices.stride(0);
        indices_stride_o = max_indices.stride(1);
        indices_stride_i = max_indices.stride(2);
        indices_stride_h = max_indices.stride(3);
        indices_stride_w = max_indices.stride(4);
    }

    // Cache-friendly blocking parameters
    constexpr int64_t BLOCK_H = 32;
    constexpr int64_t BLOCK_W = 32;
    constexpr int64_t BLOCK_OC = 8;
    
    // Optimized parallel loop with better work distribution
    #pragma omp parallel
    {
        // Thread-local temporary storage
        tls_storage.ensure_capacity(BLOCK_H * BLOCK_W * BLOCK_OC);
        
        #pragma omp for schedule(dynamic) collapse(2)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t h_block = 0; h_block < height; h_block += BLOCK_H) {
                const int64_t h_end = std::min(h_block + BLOCK_H, height);
                
                for (int64_t w_block = 0; w_block < width; w_block += BLOCK_W) {
                    const int64_t w_end = std::min(w_block + BLOCK_W, width);
                    
                    for (int64_t o_block = 0; o_block < out_channels; o_block += BLOCK_OC) {
                        const int64_t o_end = std::min(o_block + BLOCK_OC, out_channels);
                        
                        // Process block
                        for (int64_t h = h_block; h < h_end; h++) {
                            for (int64_t w = w_block; w < w_end; w++) {
                                for (int64_t o = o_block; o < o_end; o++) {
                                    T channel_sum = static_cast<T>(0);
                                    
                                    T* output_pos = output_ptr + b * output_stride_b + 
                                                         o * output_stride_o + h * output_stride_h + w;
                                    
                                    for (int64_t i = 0; i < in_channels; i++) {
                                        T best_value = Policy::initial_value();
                                        int64_t best_k = 0;
                                        
                                        const T* input_base = input_ptr + b * input_stride_b + 
                                                             i * input_stride_c;
                                        const T* weight_base = weight_ptr + o * weight_stride_o + 
                                                              i * weight_stride_i;
                                        
                                        // Optimized kernel loop - linear memory access
                                        for (int64_t kh = 0; kh < kernel_height; kh++) {
                                            const int64_t ph = h + kernel_height_half * 2 - kh;
                                            const T* input_row = input_base + ph * input_stride_h;
                                            const T* weight_row = weight_base + kh * weight_stride_kh;
                                            
                                            for (int64_t kw = 0; kw < kernel_width; kw++) {
                                                const int64_t pw = w + kernel_width_half * 2 - kw;
                                                
                                                const T val = Policy::combine(
                                                    input_row[pw * input_stride_w],
                                                    weight_row[kw * weight_stride_kw]
                                                );
                                                
                                                if (Policy::is_better(val, best_value)) {
                                                    best_value = val;
                                                    if constexpr (save_index) {
                                                        best_k = kh * kernel_width + kw;
                                                    }
                                                }
                                            }
                                        }
                                        
                                        channel_sum += best_value;
                                        
                                        if constexpr (save_index) {
                                            int64_t* indices_pos = indices_ptr + 
                                                b * indices_stride_b + o * indices_stride_o + 
                                                i * indices_stride_i + h * indices_stride_h + w;
                                            *indices_pos = best_k;
                                        }
                                    }
                                    
                                    *output_pos = channel_sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (save_index) {
        return std::make_tuple(output, max_indices);
    } else {
        return output;
    }
}

template<MORPHO_OPERATION op, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_output,
    torch::Tensor max_indices
) {

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
    
    auto max_indices_accessor = [&]() {
        if constexpr (save_index) {
            return max_indices.template accessor<int64_t, 5>();
        } else {
            static thread_local torch::Tensor dummy = torch::zeros({1,1,1,1,1}, torch::kLong);
            return dummy.accessor<int64_t, 5>();
        }
    }();
    
    std::vector<torch::Tensor> local_grad_weights(omp_get_max_threads());
    for (auto& lgw : local_grad_weights) {
        lgw = torch::zeros_like(weight);
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto local_grad_weight_accessor = local_grad_weights[tid].template accessor<T, 4>();
        
        #pragma omp for collapse(2)
        for (int64_t b = 0; b < batch_size; b++) {
            for(int64_t o = 0; o < out_channels; o++) {          
                for(int64_t i = 0; i < in_channels; i++) {       
                    for (int64_t h = 0; h < height; h++) {
                        for (int64_t w = 0; w < width; w++) {
                            T grad_val = grad_output_accessor[b][o][h][w];
                            int64_t max_kh = 0, max_kw = 0;

                            if constexpr (save_index) {
                                int64_t val = max_indices_accessor[b][o][i][h][w];
                                max_kh = val / kernel_width;  
                                max_kw = val % kernel_width;
                            } else {
                                T max_value = Policy::initial_value();
                                
                                for (int64_t kh = 0; kh < kernel_height; kh++) {
                                    for (int64_t kw = 0; kw < kernel_width; kw++) {
                                        int64_t ph = h + kernel_height_half * 2 - kh;
                                        int64_t pw = w + kernel_width_half * 2 - kw;
                                        
                                        T val = Policy::combine(padded_input_accessor[b][i][ph][pw], 
                                                              weight_accessor[o][i][kh][kw]);
                                        
                                        if (Policy::is_better(val, max_value)) 
                                        {  
                                            max_value = val;
                                            max_kh = kh;
                                            max_kw = kw;
                                        }
                                    }
                                }
                            }
                            
                            int64_t ih = h + kernel_height_half - max_kh;
                            int64_t iw = w + kernel_width_half - max_kw;
                            
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                #pragma omp atomic
                                grad_input_accessor[b][i][ih][iw] += grad_val;  
                            }
                            
                            if constexpr (op == MORPHO_OPERATION::DILATION) {
                                local_grad_weight_accessor[o][i][max_kh][max_kw] += grad_val;
                            } else { 
                                local_grad_weight_accessor[o][i][max_kh][max_kw] -= grad_val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (const auto& lgw : local_grad_weights) {
        grad_weight += lgw;
    }
    
    return {grad_input, grad_weight};
}