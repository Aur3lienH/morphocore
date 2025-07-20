#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#define COARSING_LEVEL 2


template<typename scalar_t, MORPHO_OPERATION Op>
struct MorphoPolicy;

template<typename scalar_t> 
struct MorphoPolicy<scalar_t, MORPHO_OPERATION::DILATION>
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
};

template<typename scalar_t> 
struct MorphoPolicy<scalar_t, MORPHO_OPERATION::EROSION>
{
    __device__ __forceinline__ static scalar_t initial_value() 
    {
        return std::numeric_limits<scalar_t>::max();
    }
    
    __device__ __forceinline__ static scalar_t combine(scalar_t input, scalar_t weight) 
    {
        return input - weight;
    }
    
    __device__ __forceinline__ static bool is_better(scalar_t new_val, scalar_t current_best) 
    {
        return new_val < current_best;
    }
};


template <typename scalar_t>
__global__ void morpho_dilation_forward_kernel(
    const scalar_t* padded_input,  
    const scalar_t* weight,
    scalar_t* output, 
    int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width) {
        
    extern __shared__ char shared_mem[];
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t* shared_weight = (scalar_t*)shared_mem;
    // Load the weight in the shared memory
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel_height * kernel_width; i += blockDim.x * blockDim.y) {
        if (i < kernel_height * kernel_width) 
        {
            shared_weight[i] = weight[i];
        }
    }
    __syncthreads();
    if (h < height && w < width && b < batch_size) {
        const int c = 0;
        
        scalar_t max_value = std::numeric_limits<scalar_t>::lowest();
        int max_flat_index = -1; 
        
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int padded_h = h + kh;
                int padded_w = w + kw;
                
                scalar_t input_val = padded_input[
                    b * channels * padded_height * padded_width + 
                    c * padded_height * padded_width + 
                    padded_h * padded_width + 
                    padded_w];
                                          
                scalar_t weight_val = shared_weight[kh * kernel_width + kw];
                scalar_t val = input_val + weight_val;
                
                int flat_index = kh * kernel_width + kw;
                
                if (val > max_value) {
                    max_value = val;
                    max_flat_index = flat_index;
                }
            }
        }
        
        if(max_flat_index != -1) {
            int out_idx = b * channels * height * width + c * height * width + h * width + w;
            output[out_idx] = max_value;
            max_indexes[out_idx] = max_flat_index;
        }
    }
}


template <MORPHO_OPERATION Op, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_forward_specialized_kernel(
    const scalar_t* padded_input,  
    const scalar_t* weight,
    scalar_t* output, 
    int64_t* max_indexes,  // Shape: [batch_size, out_channels, in_channels, height, width]
    int batch_size,
    int in_channels,     
    int out_channels,    
    int height,
    int width) {

    extern __shared__ char shared_mem[];

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int padded_height = height + (KH / 2) * 2;
    const int padded_width = width + (KW / 2) * 2;
    const int kernel_height_half = KH / 2;
    const int kernel_width_half = KW / 2;

    using Policy = MorphoPolicy<scalar_t, Op>;
    
    if (h < height && w < width && b < batch_size) {
        
        for (int o = 0; o < out_channels; o++) {
            
            scalar_t output_value = 0; 
            
            for (int c = 0; c < in_channels; c++) {
                
                scalar_t best_value = Policy::initial_value();
                int max_flat_index = -1; // ✅ Index for THIS specific input channel
                
                #pragma unroll
                for (int kh = 0; kh < KH; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < KW; kw++) {
                        int padded_h = h - kh + 2 * kernel_height_half;
                        int padded_w = w - kw + 2 * kernel_width_half;
                        
                        scalar_t input_val = padded_input[
                            b * in_channels * padded_height * padded_width + 
                            c * padded_height * padded_width + 
                            padded_h * padded_width + 
                            padded_w];
                        
                        scalar_t weight_val = weight[
                            o * in_channels * KH * KW + 
                            c * KH * KW + 
                            kh * KW + 
                            kw];
                        
                        scalar_t val = Policy::combine(input_val, weight_val);
                        
                        int flat_index = kh * KW + kw;
                        
                        if(Policy::is_better(val, best_value)) {
                            best_value = val;
                            max_flat_index = flat_index;
                        }
                    }
                }
                
                output_value += best_value;
                
                // ✅ Store the winning index for THIS input channel
                int max_idx_index = b * out_channels * in_channels * height * width + 
                                   o * in_channels * height * width + 
                                   c * height * width + 
                                   h * width + 
                                   w;
                max_indexes[max_idx_index] = max_flat_index;
            }
            
            // ✅ Store final summed output value
            int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
            output[out_idx] = output_value;
        }
    }
}


template <MORPHO_OPERATION Op, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_forward_kernel(
    const scalar_t* padded_input,  
    const scalar_t* weight,
    scalar_t* output, 
    int64_t* max_indexes,
    int batch_size,
    int in_channels,     
    int out_channels,    
    int height,
    int width) {

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h >= height || w >= width || b >= batch_size) return;
    
    const int padded_height = height + (KH / 2) * 2;
    const int padded_width = width + (KW / 2) * 2;
    const int kernel_height_half = KH / 2;
    const int kernel_width_half = KW / 2;

    using Policy = MorphoPolicy<scalar_t, Op>;
    
    for (int o = 0; o < out_channels; o++) {
        scalar_t output_value = 0; 
        
        for (int c = 0; c < in_channels; c++) {
            scalar_t best_value = Policy::initial_value();
            int max_flat_index = -1;
            
            // ✅ Simple nested loops without #pragma unroll to avoid complexity
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int padded_h = h - kh + 2 * kernel_height_half;
                    int padded_w = w - kw + 2 * kernel_width_half;
                    
                    // ✅ Bounds checking
                    if (padded_h >= 0 && padded_h < padded_height && 
                        padded_w >= 0 && padded_w < padded_width) {
                        
                        scalar_t input_val = padded_input[
                            b * in_channels * padded_height * padded_width + 
                            c * padded_height * padded_width + 
                            padded_h * padded_width + 
                            padded_w];
                        
                        scalar_t weight_val = weight[
                            o * in_channels * KH * KW + 
                            c * KH * KW + 
                            kh * KW + 
                            kw];
                        
                        scalar_t val = Policy::combine(input_val, weight_val);
                        
                        if(Policy::is_better(val, best_value)) {
                            best_value = val;
                            max_flat_index = kh * KW + kw;
                        }
                    }
                }
            }
            
            output_value += best_value;
            
            // ✅ Store max index for this specific channel
            int max_idx_index = b * out_channels * in_channels * height * width + 
                               o * in_channels * height * width + 
                               c * height * width + 
                               h * width + 
                               w;
            max_indexes[max_idx_index] = max_flat_index;
        }
        
        // ✅ Store final output
        int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
        output[out_idx] = output_value;
    }
}



template <MORPHO_OPERATION Op, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_backward_kernel(
    const scalar_t* padded_input,
    const scalar_t* weight,
    scalar_t* grad_input,
    scalar_t* grad_weight,
    const scalar_t* grad_output,
    const int64_t* max_indexes,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h >= height || w >= width || b >= batch_size) return;
    
    const int kernel_height_half = KH / 2;
    const int kernel_width_half = KW / 2;
    
    for (int o = 0; o < out_channels; o++) {
        const int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
        const scalar_t grad = grad_output[out_idx];
        
        for (int c = 0; c < in_channels; c++) {
            // ✅ Get the winning index for this specific input channel
            int max_idx_index = b * out_channels * in_channels * height * width + 
                               o * in_channels * height * width + 
                               c * height * width + 
                               h * width + 
                               w;
            int best_flat_idx = max_indexes[max_idx_index];
            
            if (best_flat_idx != -1) {
                int best_kh = best_flat_idx / KW;
                int best_kw = best_flat_idx % KW;
                
                // ✅ Backpropagate to input
                int in_h = h - best_kh + kernel_height_half;
                int in_w = w - best_kw + kernel_width_half;
                
                // ✅ Bounds checking
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    scalar_t input_grad = (Op == MORPHO_OPERATION::EROSION) ? -grad : grad;
                    atomicAdd(&grad_input[b * in_channels * height * width + 
                                        c * height * width + 
                                        in_h * width + 
                                        in_w], input_grad);
                }
                
                // ✅ Backpropagate to weight - simple atomic without complex reductions
                scalar_t weight_grad = (Op == MORPHO_OPERATION::EROSION) ? -grad : grad;
                int weight_idx = o * in_channels * KH * KW + c * KH * KW + best_kh * KW + best_kw;
                atomicAdd(&grad_weight[weight_idx], weight_grad);
            }
        }
    }
}

template <MORPHO_OPERATION Op, typename scalar_t>
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
    int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        batch_size
    );
    
    // ✅ FIX: Remove shared memory usage entirely for now
    size_t shared_memory_size = 0;
    
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_forward_kernel<Op, float, 3, 3><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_forward_kernel<Op, float, 5, 5><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_forward_kernel<Op, float, 7, 7><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
    else if constexpr (std::is_same_v<scalar_t, double>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_forward_kernel<Op, double, 3, 3><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_forward_kernel<Op, double, 5, 5><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_forward_kernel<Op, double, 7, 7><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
}


template <MORPHO_OPERATION Op, typename scalar_t>
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
    int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        batch_size
    );
    
    // ✅ FIX: No shared memory to avoid complexity
    size_t shared_memory_size = 0;
    
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_backward_kernel<Op, float, 3, 3><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_backward_kernel<Op, float, 5, 5><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_backward_kernel<Op, float, 7, 7><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
    else if constexpr (std::is_same_v<scalar_t, double>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_backward_kernel<Op, double, 3, 3><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_backward_kernel<Op, double, 5, 5><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_backward_kernel<Op, double, 7, 7><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
}


#define INSTANTIATE_MORPHO_OP(T, H, W, OP) \
    template __global__ void morpho_operation_forward_kernel<MORPHO_OPERATION::OP, T, H, W>( \
        const T*, const T*, T*, int64_t*, int, int, int, int, int); \
    template __global__ void morpho_operation_backward_kernel<MORPHO_OPERATION::OP, T, H, W>( \
        const T*, const T*, T*, T*, const T*, const int64_t*, int, int, int, int, int);

template struct MorphoPolicy<float, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicy<float, MORPHO_OPERATION::EROSION>;
template struct MorphoPolicy<double, MORPHO_OPERATION::DILATION>;
template struct MorphoPolicy<double, MORPHO_OPERATION::EROSION>;

INSTANTIATE_MORPHO_OP(float, 3, 3, DILATION)
INSTANTIATE_MORPHO_OP(float, 3, 3, EROSION)
INSTANTIATE_MORPHO_OP(float, 5, 5, DILATION)
INSTANTIATE_MORPHO_OP(float, 5, 5, EROSION)
INSTANTIATE_MORPHO_OP(float, 7, 7, DILATION)
INSTANTIATE_MORPHO_OP(float, 7, 7, EROSION)
INSTANTIATE_MORPHO_OP(double, 3, 3, DILATION)
INSTANTIATE_MORPHO_OP(double, 3, 3, EROSION)
INSTANTIATE_MORPHO_OP(double, 5, 5, DILATION)
INSTANTIATE_MORPHO_OP(double, 5, 5, EROSION)
INSTANTIATE_MORPHO_OP(double, 7, 7, DILATION)
INSTANTIATE_MORPHO_OP(double, 7, 7, EROSION)


// Template instantiation for dilation !
template void morpho_forward_kernel<MORPHO_OPERATION::DILATION, float>(const float* input,
    const float* weight,
    float* output,
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

template void morpho_forward_kernel<MORPHO_OPERATION::DILATION, double>(const double* input,
    const double* weight,
    double* output,
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

template void morpho_backward_kernel<MORPHO_OPERATION::DILATION, float>(const float* padded_input,
    const float* weight, 
    float* grad_input,
    float* grad_weight,
    const float* grad_output,
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

template void morpho_backward_kernel<MORPHO_OPERATION::DILATION, double>(const double* padded_input,
    const double* weight, 
    double* grad_input,
    double* grad_weight,
    const double* grad_output,
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

// Template instantation for erosion !
template void morpho_forward_kernel<MORPHO_OPERATION::EROSION, float>(const float* input,
    const float* weight,
    float* output,
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

template void morpho_forward_kernel<MORPHO_OPERATION::EROSION, double>(const double* input,
    const double* weight,
    double* output,
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

template void morpho_backward_kernel<MORPHO_OPERATION::EROSION, float>(const float* padded_input,
    const float* weight, 
    float* grad_input,
    float* grad_weight,
    const float* grad_output,
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

template void morpho_backward_kernel<MORPHO_OPERATION::EROSION, double>(const double* padded_input,
    const double* weight, 
    double* grad_input,
    double* grad_weight,
    const double* grad_output,
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