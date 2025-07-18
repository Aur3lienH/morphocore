#include "dilation.cuh"
#define COARSING_LEVEL 2

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


template <MorphoOp Op, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_forward_specialized_kernel(
    const scalar_t* padded_input,  
    const scalar_t* weight,
    scalar_t* output, 
    int64_t* max_indexes,
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
                int max_flat_index = -1; 
                
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
            }
            
            int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
            output[out_idx] = output_value;
            max_indexes[out_idx] = max_flat_index;
        }
    }
}


template <MorphoOp Op, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_backward_specialized_kernel(
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

    extern __shared__ char shared_mem[];
    using Policy = MorphoPolicy<scalar_t, Op>;

    // ✅ Use smaller shared memory - only for current block's contribution
    scalar_t* shared_grad_weight = (scalar_t*)(shared_mem);
    const int local_weight_size = out_channels * in_channels * KH * KW;
    const int kernel_height_half = KH / 2;
    const int kernel_width_half = KW / 2;
    const int padded_height = height + (KH / 2) * 2;
    const int padded_width = width + (KW / 2) * 2;

    // Initialize shared memory
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < local_weight_size; i += blockDim.x * blockDim.y) {
       shared_grad_weight[i] = 0;
    }
    __syncthreads();

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = linear_tid / 32;
    int lane_id = linear_tid % 32;

    // ✅ Use local arrays for per-thread weight gradients
    scalar_t local_weight_grad[KH * KW] = {0};  // Only for current (o,c) pair

    if (h < height && w < width && b < batch_size) {
        
        for (int o = 0; o < out_channels; o++) {
            const int out_idx = b * out_channels * height * width + o * height * width + h * width + w;
            const scalar_t grad = grad_output[out_idx];
            
            for (int c = 0; c < in_channels; c++) {
                
                // Clear local accumulator
                for (int i = 0; i < KH * KW; i++) local_weight_grad[i] = 0;
                
                // Recompute winning position
                scalar_t best_value = Policy::initial_value();
                int best_flat_idx = -1;
                
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
                        
                        if (Policy::is_better(val, best_value)) {
                            best_value = val;
                            best_flat_idx = kh * KW + kw;
                        }
                    }
                }
                
                if (best_flat_idx != -1) {
                    int best_kh = best_flat_idx / KW;
                    int best_kw = best_flat_idx % KW;
                    
                    // Backpropagate to input
                    int in_h = h - best_kh + kernel_height_half;
                    int in_w = w - best_kw + kernel_width_half;
                    
                    in_h = max(0, min(height - 1, in_h));
                    in_w = max(0, min(width - 1, in_w));
                    
                    scalar_t input_grad = (Op == MorphoOp::EROSION) ? -grad : grad;
                    
                    atomicAdd(&grad_input[b * in_channels * height * width + 
                                        c * height * width + 
                                        in_h * width + 
                                        in_w], input_grad);
                    
                    // Accumulate weight gradient locally
                    scalar_t weight_grad = (Op == MorphoOp::EROSION) ? -grad : grad;
                    local_weight_grad[best_flat_idx] += weight_grad;
                }
                
                // ✅ Warp-level reduction for weight gradients
                #pragma unroll
                for (int k = 0; k < KH * KW; k++) {
                    scalar_t warp_sum = local_weight_grad[k];
                    
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
                    }
                    
                    if (lane_id == 0) {
                        int global_weight_idx = o * in_channels * KH * KW + c * KH * KW + k;
                        atomicAdd(&shared_grad_weight[global_weight_idx], warp_sum);
                    }
                }
            }
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int k = lane_id; k < local_weight_size; k += 32) {
            const scalar_t val = shared_grad_weight[k];
            if (val != 0) {
                atomicAdd(&grad_weight[k], val);
            }
        }
    }
}



template <typename scalar_t>
__global__ void morpho_dilation_backward_kernel(
    scalar_t* grad_input,
    scalar_t* grad_weight,
    const scalar_t* grad_output,
    const int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half) {

    extern __shared__ char shared_mem[];

    scalar_t* shared_grad_weight = (scalar_t*)(shared_mem);

    // Set the gradient to 0
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel_height * kernel_width; i += blockDim.x * blockDim.y) 
    {
       shared_grad_weight[i] = 0;
    }
   __syncthreads(); 

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane_id = linear_tid % 32;

    scalar_t grad_val = 0;
    int k_index = -1;
    bool is_active = false;
    
    if (h < height && w < width && b < batch_size) {
        const int c = 0;
        
        const int idx = b * channels * height * width + c * height * width + h * width + w;
        const scalar_t grad = grad_output[idx];
        const int max_flat_index = max_indexes[idx];

        if (max_flat_index != -1)
        {
            const int kh = max_flat_index / kernel_width;
            const int kw = max_flat_index % kernel_width;
            
            int in_h = h + kh - kernel_height_half;
            int in_w = w + kw - kernel_width_half;
            
            in_h = max(0, min(height - 1, in_h));
            in_w = max(0, min(width - 1, in_w));
            
            atomicAdd(&grad_input[b * channels * height * width + 
                                c * height * width + 
                                in_h * width + 
                                in_w], grad * (1.0f / (kernel_height * kernel_width)));
                            
            grad_val = grad * (1.0f / (kernel_height * kernel_width));
            k_index = kh * kernel_width + kw;
            is_active = true;
        }
        
        
    }


    // Accumulate in the first lane and than atomic add
    for (int k = 0; k < kernel_height * kernel_width; k++) {
        unsigned mask = __ballot_sync(0xffffffff, k_index == k);
        
        if (mask > 0) { 
            scalar_t my_val = (k_index == k) ? grad_val : 0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                my_val += __shfl_down_sync(mask, my_val, offset);
            }
            if (k_index == k && lane_id == __ffs(mask) - 1) {
                atomicAdd(&shared_grad_weight[k], my_val);
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel_height * kernel_width; i += blockDim.x * blockDim.y) {
       if (shared_grad_weight[i] != 0) {
           atomicAdd(&grad_weight[i], shared_grad_weight[i]);
       }
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
    int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        batch_size
    );
    
    // ✅ Fixed: No shared memory for weights in multi-channel case
    // (would be too large: out_channels * in_channels * kernel_height * kernel_width)
    size_t shared_memory_size = 0;
    
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_forward_specialized_kernel<Op, float, 3, 3><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_forward_specialized_kernel<Op, float, 5, 5><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_forward_specialized_kernel<Op, float, 7, 7><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
    else if constexpr (std::is_same_v<scalar_t, double>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_forward_specialized_kernel<Op, double, 3, 3><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_forward_specialized_kernel<Op, double, 5, 5><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_forward_specialized_kernel<Op, double, 7, 7><<<blocks, threads, shared_memory_size>>>(
                input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
    
    int kernel_height_half = kernel_height / 2;
    int kernel_width_half = kernel_width / 2;
    
    size_t general_shared_memory_size = out_channels * in_channels * kernel_height * kernel_width * sizeof(scalar_t);
    
    morpho_operation_general_kernel<Op, scalar_t><<<blocks, threads, general_shared_memory_size>>>(
        input, weight, output, max_indexes, batch_size, in_channels, out_channels, height, width,
        padded_height, padded_width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
}


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
    int kernel_width)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        batch_size
    );
    
    // ✅ Fixed: Shared memory for weight gradient accumulation
    size_t shared_memory_size = out_channels * in_channels * kernel_height * kernel_width * sizeof(scalar_t);
    
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_backward_specialized_kernel<Op, float, 3, 3><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_backward_specialized_kernel<Op, float, 5, 5><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_backward_specialized_kernel<Op, float, 7, 7><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
    }
    else if constexpr (std::is_same_v<scalar_t, double>) {
        if (kernel_height == 3 && kernel_width == 3) {
            morpho_operation_backward_specialized_kernel<Op, double, 3, 3><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 5 && kernel_width == 5) {
            morpho_operation_backward_specialized_kernel<Op, double, 5, 5><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
        else if (kernel_height == 7 && kernel_width == 7) {
            morpho_operation_backward_specialized_kernel<Op, double, 7, 7><<<blocks, threads, shared_memory_size>>>(
                padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
                batch_size, in_channels, out_channels, height, width);
            return;
        }
    }


    int kernel_height_half = kernel_height / 2;
    int kernel_width_half = kernel_width / 2;
    
    morpho_operation_backward_general_kernel<Op, scalar_t><<<blocks, threads, shared_memory_size>>>(
        padded_input, weight, grad_input, grad_weight, grad_output, max_indexes, 
        batch_size, in_channels, out_channels, height, width,
        padded_height, padded_width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
}


#define INSTANTIATE_MORPHO_SPECIALIZED_OP(T, H, W, OP) \
    template __global__ void morpho_operation_forward_specialized_kernel<MorphoOp::OP, T, H, W>( \
        const T*, const T*, T*, int64_t*, int, int, int, int, int); \
    template __global__ void morpho_operation_backward_specialized_kernel<MorphoOp::OP, T, H, W>( \
        const T*, const T*, T*, T*, const T*, const int64_t*, int, int, int, int, int);



INSTANTIATE_MORPHO_SPECIALIZED_OP(float, 3, 3, DILATION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(float, 3, 3, EROSION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(float, 5, 5, DILATION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(float, 5, 5, EROSION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(float, 7, 7, DILATION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(float, 7, 7, EROSION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(double, 3, 3, DILATION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(double, 3, 3, EROSION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(double, 5, 5, DILATION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(double, 5, 5, EROSION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(double, 7, 7, DILATION)
INSTANTIATE_MORPHO_SPECIALIZED_OP(double, 7, 7, EROSION)

template void forward_kernel<float>(const float* input,
    const float* weight,
    float* output,
    int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width);


template void forward_kernel<double>(const double* input,
    const double* weight,
    double* output,
    int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width);


template void backward_kernel<float>(float* grad_input,
    float* grad_weight,
    const float* grad_output,
    const int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width);

template void backward_kernel<double>(double* grad_input,
    double* grad_weight,
    const double* grad_output,
    const int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width,
    int padded_height,
    int padded_width,
    int kernel_height,
    int kernel_width);