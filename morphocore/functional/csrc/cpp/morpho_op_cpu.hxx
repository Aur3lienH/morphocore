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

// ✅ Helper function for safe input access with implicit padding
template<typename T, MORPHO_OPERATION op>
inline T get_input_value_safe(
    const T* input_ptr,
    int64_t b, int64_t c, int64_t h, int64_t w,
    int64_t height, int64_t width,
    int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w
) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        return MorphoPolicyCPU<T, op>::padding_value();
    }
    return input_ptr[b * stride_b + c * stride_c + h * stride_h + w * stride_w];
}



template<MORPHO_OPERATION op, typename T, bool save_index, int KERNEL_SIZE>
auto morpho_operation_forward_cpu_template(torch::Tensor input, torch::Tensor weight) 
{
    static_assert(KERNEL_SIZE >= 1 && KERNEL_SIZE <= 15, "Kernel size must be between 1 and 15");
    static_assert(KERNEL_SIZE % 2 == 1, "Kernel size must be odd");
    
    const auto input_shape = input.sizes();
    const int64_t batch_size = input_shape[0];
    const int64_t in_channels = input_shape[1];
    const int64_t out_channels = weight.size(0);
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];
    
    // Constantes compile-time
    constexpr int64_t kernel_half = KERNEL_SIZE / 2;
    constexpr int64_t kernel_area = KERNEL_SIZE * KERNEL_SIZE;
    constexpr T PADDING_VAL = (op == MORPHO_OPERATION::DILATION) 
        ? -std::numeric_limits<T>::infinity() 
        : std::numeric_limits<T>::infinity();
    constexpr T INITIAL_VAL = PADDING_VAL;
    
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    
    torch::Tensor output = torch::empty({batch_size, out_channels, height, width}, 
                                       input_contig.options());
    
    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();
    
    // Strides optimisés pour tenseurs contigus
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_area;
    const int64_t weight_in_stride = kernel_area;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;
    
    torch::Tensor max_indices;
    int64_t* __restrict__ indices_ptr = nullptr;
    
    if constexpr (save_index) {
        max_indices = torch::empty({batch_size, out_channels, in_channels, height, width}, 
                                   torch::kLong);
        indices_ptr = max_indices.template data_ptr<int64_t>();
    }
    
    // Tiling adaptatif basé sur la taille du kernel
    constexpr int64_t TILE_H = (KERNEL_SIZE <= 3) ? 16 : (KERNEL_SIZE <= 5) ? 12 : 8;
    constexpr int64_t TILE_W = TILE_H;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(guided) collapse(3)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t o = 0; o < out_channels; o++) {
                for (int64_t h_tile = 0; h_tile < height; h_tile += TILE_H) {
                    const int64_t h_end = std::min(h_tile + TILE_H, height);
                    
                    // Pointeurs de base pré-calculés
                    const T* input_batch = input_ptr + b * input_batch_stride;
                    const T* weight_out = weight_ptr + o * weight_out_stride;
                    T* output_base = output_ptr + b * output_batch_stride + o * output_channel_stride;
                    
                    int64_t* indices_base = nullptr;
                    if constexpr (save_index) {
                        indices_base = indices_ptr + 
                            b * (out_channels * in_channels * height * width) + 
                            o * (in_channels * height * width);
                    }
                    
                    for (int64_t w_tile = 0; w_tile < width; w_tile += TILE_W) {
                        const int64_t w_end = std::min(w_tile + TILE_W, width);
                        
                        for (int64_t h = h_tile; h < h_end; h++) {
                            for (int64_t w = w_tile; w < w_end; w++) {
                                T channel_sum = static_cast<T>(0);
                                
                                for (int64_t i = 0; i < in_channels; i++) {
                                    const T* input_channel = input_batch + i * input_channel_stride;
                                    const T* weight_in = weight_out + i * weight_in_stride;
                                    
                                    T best_value = INITIAL_VAL;
                                    int64_t best_k = 0;
                                    int64_t k = 0;
                                    
                                    // LE COEUR DE L'APPROCHE HYBRIDE:
                                    // Le compilateur va unroll ces boucles automatiquement
                                    // grâce aux constantes compile-time KERNEL_SIZE
                                    
                                    #pragma unroll
                                    for (int64_t kh = 0; kh < KERNEL_SIZE; kh++) {
                                        const int64_t ph = h + kernel_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);
                                        
                                        #pragma unroll  
                                        for (int64_t kw = 0; kw < KERNEL_SIZE; kw++, k++) {
                                            const int64_t pw = w + kernel_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);
                                            
                                            // Accès mémoire avec bounds checking optimisé
                                            const T input_val = (h_valid && w_valid) 
                                                ? input_channel[ph * width + pw] 
                                                : PADDING_VAL;
                                            
                                            // Opérations directes - pas d'appels de fonction
                                            const T val = (op == MORPHO_OPERATION::DILATION) 
                                                ? input_val + weight_in[k]
                                                : input_val - weight_in[k];
                                            
                                            // Comparaison directe - pas d'appels de fonction
                                            const bool is_better = (op == MORPHO_OPERATION::DILATION) 
                                                ? (val > best_value) 
                                                : (val < best_value);
                                            
                                            if (is_better) {
                                                best_value = val;
                                                if constexpr (save_index) {
                                                    best_k = k;
                                                }
                                            }
                                        }
                                    }
                                    
                                    channel_sum += best_value;
                                    
                                    if constexpr (save_index) {
                                        indices_base[i * height * width + h * width + w] = best_k;
                                    }
                                }
                                
                                output_base[h * width + w] = channel_sum;
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
auto morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight) 
{

    using Policy = MorphoPolicyCPU<T, op>;
    
    const auto input_shape = input.sizes();
    const auto weight_shape = weight.sizes();
    const int64_t kernel_height = weight_shape[2];
    const int64_t kernel_width = weight_shape[3];
    if(kernel_height == 3 && kernel_width == 3)
    {
        return morpho_operation_forward_cpu_template<op,T,save_index,3>(input,weight);
    }
    if(kernel_height == 5 && kernel_width == 5)
    {
        return morpho_operation_forward_cpu_template<op,T,save_index,5>(input,weight);
    }
    if(kernel_height == 7 && kernel_width == 7)
    {
        return morpho_operation_forward_cpu_template<op,T,save_index,7>(input,weight);
    }
    
    const int64_t batch_size = input_shape[0];
    const int64_t in_channels = input_shape[1];
    const int64_t out_channels = weight_shape[0];
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];

    const int64_t kernel_height_half = kernel_height / 2;
    const int64_t kernel_width_half = kernel_width / 2;
    
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    
    torch::Tensor output = torch::empty({batch_size, out_channels, height, width}, 
                                       input_contig.options());
    
    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();
    
    // Optimized strides for contiguous tensors
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;
    
    torch::Tensor max_indices;
    int64_t* __restrict__ indices_ptr = nullptr;
    
    if constexpr (save_index) {
        max_indices = torch::empty({batch_size, out_channels, in_channels, height, width}, 
                                   torch::kLong);
        indices_ptr = max_indices.template data_ptr<int64_t>();
    }
    
    constexpr T PADDING_VAL = Policy::padding_value();
    constexpr T INITIAL_VAL = Policy::initial_value();
    
    // Smaller tiles for better cache utilization
    constexpr int64_t TILE_H = 16;
    constexpr int64_t TILE_W = 16;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(guided) collapse(3)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t o = 0; o < out_channels; o++) {
                for (int64_t h_tile = 0; h_tile < height; h_tile += TILE_H) {
                    const int64_t h_end = std::min(h_tile + TILE_H, height);
                    
                    const T* input_batch = input_ptr + b * input_batch_stride;
                    const T* weight_out = weight_ptr + o * weight_out_stride;
                    T* output_base = output_ptr + b * output_batch_stride + o * output_channel_stride;
                    
                    int64_t* indices_base = nullptr;
                    if constexpr (save_index) {
                        indices_base = indices_ptr + 
                            b * (out_channels * in_channels * height * width) + 
                            o * (in_channels * height * width);
                    }
                    
                    for (int64_t w_tile = 0; w_tile < width; w_tile += TILE_W) {
                        const int64_t w_end = std::min(w_tile + TILE_W, width);
                        
                        for (int64_t h = h_tile; h < h_end; h++) {
                            for (int64_t w = w_tile; w < w_end; w++) {
                                T channel_sum = static_cast<T>(0);
                                
                                for (int64_t i = 0; i < in_channels; i++) {
                                    const T* input_channel = input_batch + i * input_channel_stride;
                                    const T* weight_in = weight_out + i * weight_in_stride;
                                    
                                    T best_value = INITIAL_VAL;
                                    int64_t best_k = 0;
                                    int64_t k = 0;
                                    
                                    for (int64_t kh = 0; kh < kernel_height; kh++) {
                                        const int64_t ph = h + kernel_height_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);
                                        
                                        for (int64_t kw = 0; kw < kernel_width; kw++, k++) {
                                            const int64_t pw = w + kernel_width_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);
                                            
                                            const T input_val = (h_valid && w_valid) 
                                                ? input_channel[ph * width + pw] 
                                                : PADDING_VAL;
                                            
                                            const T val = Policy::combine(input_val, weight_in[k]);
                                            
                                            if (Policy::is_better(val, best_value)) {
                                                best_value = val;
                                                if constexpr (save_index) {
                                                    best_k = k;
                                                }
                                            }
                                        }
                                    }
                                    
                                    channel_sum += best_value;
                                    
                                    if constexpr (save_index) {
                                        indices_base[i * height * width + h * width + w] = best_k;
                                    }
                                }
                                
                                output_base[h * width + w] = channel_sum;
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
    
    // Ensure contiguous memory layout for better cache performance
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto grad_output_contig = grad_output.contiguous();
    auto max_indices_contig = max_indices.contiguous();
    
    auto grad_input = torch::zeros_like(input_contig);
    auto grad_weight = torch::zeros_like(weight_contig);
    
    // Use raw pointers for better performance
    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    const T* __restrict__ grad_output_ptr = grad_output_contig.template data_ptr<T>();
    T* __restrict__ grad_input_ptr = grad_input.template data_ptr<T>();
    
    // Pre-computed strides for contiguous tensors
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_height * kernel_width;
    const int64_t weight_in_stride = kernel_height * kernel_width;
    const int64_t grad_output_batch_stride = out_channels * height * width;
    const int64_t grad_output_channel_stride = height * width;
    
    const int64_t* __restrict__ indices_ptr = nullptr;
    int64_t indices_batch_stride = 0;
    int64_t indices_out_stride = 0;
    int64_t indices_in_stride = 0;
    
    if constexpr (save_index) {
        indices_ptr = max_indices_contig.template data_ptr<int64_t>();
        indices_batch_stride = out_channels * in_channels * height * width;
        indices_out_stride = in_channels * height * width;
        indices_in_stride = height * width;
    }
    
    // Thread-local storage to eliminate atomic operations
    const int num_threads = omp_get_max_threads();
    std::vector<torch::Tensor> local_grad_inputs(num_threads);
    std::vector<torch::Tensor> local_grad_weights(num_threads);
    
    for (int tid = 0; tid < num_threads; tid++) {
        local_grad_inputs[tid] = torch::zeros_like(grad_input);
        local_grad_weights[tid] = torch::zeros_like(grad_weight);
    }
    
    // Determine optimal tiling based on image size
    const int64_t total_pixels = height * width;
    const int64_t tile_h = (total_pixels <= 64*64) ? height : std::min(int64_t(32), height);
    const int64_t tile_w = (total_pixels <= 64*64) ? width : std::min(int64_t(32), width);
    
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        T* __restrict__ local_grad_input_ptr = local_grad_inputs[tid].template data_ptr<T>();
        T* __restrict__ local_grad_weight_ptr = local_grad_weights[tid].template data_ptr<T>();
        
        // Spatial-first loop order for better cache locality
        #pragma omp for collapse(4) schedule(dynamic)
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t h_tile = 0; h_tile < height; h_tile += tile_h) {
                for (int64_t w_tile = 0; w_tile < width; w_tile += tile_w) {
                    for (int64_t o = 0; o < out_channels; o++) {
                        
                        const int64_t h_end = std::min(h_tile + tile_h, height);
                        const int64_t w_end = std::min(w_tile + tile_w, width);
                        
                        // Pre-compute base pointers
                        const T* grad_output_base = grad_output_ptr + 
                            b * grad_output_batch_stride + o * grad_output_channel_stride;
                        
                        for (int64_t h = h_tile; h < h_end; h++) {
                            // Vectorize the inner width loop when possible
                            for (int64_t w = w_tile; w < w_end; w++) {
                                const T grad_val = grad_output_base[h * width + w];
                                
                                // Skip if gradient is zero to save computation
                                if (grad_val == static_cast<T>(0)) {
                                    continue;
                                }
                                
                                for (int64_t i = 0; i < in_channels; i++) {
                                    int64_t max_kh = 0, max_kw = 0;
                                    
                                    if constexpr (save_index) {
                                        // Optimized index access
                                        const int64_t indices_idx = 
                                            b * indices_batch_stride + 
                                            o * indices_out_stride + 
                                            i * indices_in_stride + 
                                            h * width + w;
                                        
                                        const int64_t kernel_idx = indices_ptr[indices_idx];
                                        max_kh = kernel_idx / kernel_width;
                                        max_kw = kernel_idx % kernel_width;
                                    } else {
                                        // Recompute max indices (fallback)
                                        T max_value = Policy::initial_value();
                                        
                                        const T* input_channel = input_ptr + 
                                            b * input_batch_stride + i * input_channel_stride;
                                        const T* weight_base = weight_ptr + 
                                            o * weight_out_stride + i * weight_in_stride;
                                        
                                        for (int64_t kh = 0; kh < kernel_height; kh++) {
                                            const int64_t ph = h + kernel_height_half - kh;
                                            const bool h_valid = (ph >= 0 && ph < height);
                                            
                                            for (int64_t kw = 0; kw < kernel_width; kw++) {
                                                const int64_t pw = w + kernel_width_half - kw;
                                                const bool w_valid = (pw >= 0 && pw < width);
                                                
                                                const T input_val = (h_valid && w_valid) 
                                                    ? input_channel[ph * width + pw]
                                                    : Policy::padding_value();
                                                
                                                const T val = Policy::combine(input_val, 
                                                    weight_base[kh * kernel_width + kw]);
                                                
                                                if (Policy::is_better(val, max_value)) {
                                                    max_value = val;
                                                    max_kh = kh;
                                                    max_kw = kw;
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Compute input gradient coordinate
                                    const int64_t ih = h + kernel_height_half - max_kh;
                                    const int64_t iw = w + kernel_width_half - max_kw;
                                    
                                    // Update input gradient (thread-local, no atomics)
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        const int64_t grad_input_idx = 
                                            b * input_batch_stride + 
                                            i * input_channel_stride + 
                                            ih * width + iw;
                                        local_grad_input_ptr[grad_input_idx] += grad_val;
                                    }
                                    
                                    // Update weight gradient (thread-local)
                                    const int64_t grad_weight_idx = 
                                        o * weight_out_stride + 
                                        i * weight_in_stride + 
                                        max_kh * kernel_width + max_kw;
                                    
                                    if constexpr (op == MORPHO_OPERATION::DILATION) {
                                        local_grad_weight_ptr[grad_weight_idx] += grad_val;
                                    } else {
                                        local_grad_weight_ptr[grad_weight_idx] -= grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Parallel reduction of thread-local results
    #pragma omp parallel
    {
        // Reduce grad_input
        #pragma omp for
        for (int64_t i = 0; i < grad_input.numel(); i++) {
            T sum = static_cast<T>(0);
            for (const auto& local_tensor : local_grad_inputs) {
                sum += local_tensor.template data_ptr<T>()[i];
            }
            grad_input_ptr[i] = sum;
        }
        
        // Reduce grad_weight  
        #pragma omp for
        for (int64_t i = 0; i < grad_weight.numel(); i++) {
            T sum = static_cast<T>(0);
            for (const auto& local_tensor : local_grad_weights) {
                sum += local_tensor.template data_ptr<T>()[i];
            }
            local_grad_weights[0].template data_ptr<T>()[i] = sum;
        }
    }
    
    grad_weight = local_grad_weights[0];
    
    return {grad_input, grad_weight};
}