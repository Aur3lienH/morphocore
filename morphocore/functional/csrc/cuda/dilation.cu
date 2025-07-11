#include "cuda/dilation.cuh"
#define COARSING_LEVEL 2
#ifndef MORPHO_SPECIALIZATION_LIST
#define MORPHO_SPECIALIZATION_LIST \
    X(float, 3, 3) \
    X(float, 5, 5) \
    X(float, 7, 7) \
    X(double, 3, 3) \
    X(double, 5, 5) \
    X(double, 7, 7)
#endif





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


template <typename scalar_t, int KH, int KW>
__global__ void morpho_dilation_forward_specialized_kernel(
    const scalar_t* padded_input,  
    const scalar_t* weight,
    scalar_t* output, 
    int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width) {


    extern __shared__ char shared_mem[];
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int padded_height = height + (KH / 2) * 2;
    const int padded_width = width + (KW / 2) * 2;

    //Load the weight in the shared memory
    scalar_t* shared_weight = (scalar_t*)shared_mem;
    #pragma unroll
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < KH * KW; i += blockDim.x * blockDim.y) {
        if (i < KH * KW) 
        {
            shared_weight[i] = weight[i];
        }
    }
    __syncthreads();
    if (h < height && w < width && b < batch_size) {
        const int c = 0;
        
        //Get the max value
        scalar_t max_value = std::numeric_limits<scalar_t>::lowest();
        int max_flat_index = -1; 
        
        #pragma unroll
        for (int kh = 0; kh < KH; kh++) {
            #pragma unroll
            for (int kw = 0; kw < KW; kw++) {
                int padded_h = h + kh;
                int padded_w = w + kw;
                
                scalar_t input_val = padded_input[
                    b * channels * padded_height * padded_width + 
                    c * padded_height * padded_width + 
                    padded_h * padded_width + 
                    padded_w];
                                          
                scalar_t weight_val = shared_weight[kh * KW + kw];
                scalar_t val = input_val + weight_val;
                
                int flat_index = kh * KW + kw;
                
                bool is_greater = (val > max_value);
                max_value = is_greater ? val : max_value;
                max_flat_index = is_greater ? flat_index : max_flat_index;
            }
        }

        // Update if result
        if(max_flat_index != -1) {
            int out_idx = b * channels * height * width + c * height * width + h * width + w;
            output[out_idx] = max_value;
            max_indexes[out_idx] = max_flat_index;
        }
    }
}


template <typename scalar_t, int KH, int KW>
__global__ void morpho_dilation_backward_specialized_kernel(
    scalar_t* grad_input,
    scalar_t* grad_weight,
    const scalar_t* grad_output,
    const int64_t* max_indexes,
    int batch_size,
    int channels,
    int height,
    int width) {

    extern __shared__ char shared_mem[];

    scalar_t* shared_grad_weight = (scalar_t*)(shared_mem);

    const int K_SIZE_MUL = KH * KW;
    const float GRAD_MULTIPLIER = (1.0f / (K_SIZE_MUL));

    // Set the gradient to 0
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < K_SIZE_MUL; i += blockDim.x * blockDim.y) 
    {
       shared_grad_weight[i] = 0;
    }
   __syncthreads();

    
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = linear_tid / 32;
    int lane_id = linear_tid % 32;

    scalar_t local_weight_grad[KH * KW] = {0};

    scalar_t grad_val = 0;
    int k_index = -1;
    
    if (h < height && w < width && b < batch_size) {
        const int c = 0;
        
        const int idx = b * channels * height * width + c * height * width + h * width + w;
        const scalar_t grad = grad_output[idx];
        const int max_flat_index = max_indexes[idx];

        if (max_flat_index != -1)
        {
            const int kh = max_flat_index / KW;
            const int kw = max_flat_index % KW;
            
            int in_h = h + kh - KH/2;
            int in_w = w + kw - KW/2;
            
            in_h = max(0, min(height - 1, in_h));
            in_w = max(0, min(width - 1, in_w));
            
            atomicAdd(&grad_input[b * channels * height * width + 
                                c * height * width + 
                                in_h * width + 
                                in_w], grad * GRAD_MULTIPLIER);
                        
            local_weight_grad[max_flat_index] += grad * GRAD_MULTIPLIER;
        }

    }


    // Accumulate in the first lane and than atomic add
    #pragma unroll
    for (int k = 0; k < KH * KW; k++) {
        scalar_t warp_sum = local_weight_grad[k];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Do the atomic only on the first lane
        if (lane_id == 0) {
            atomicAdd(&shared_grad_weight[k], warp_sum);
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int k = lane_id; k < KH * KW; k += 32) {
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


// Create a macro to make the initialization of each kernels easier.
#define INSTANTIATE_MORPHO_SPECIALIZED(T, H, W) \
    template __global__ void morpho_dilation_forward_specialized_kernel<T, H, W>( \
        const T*, const T*, T*, int64_t*, int, int, int, int); \
    template __global__ void morpho_dilation_backward_specialized_kernel<T, H, W>( \
        T*, T*, const T*, const int64_t*, int, int, int, int);


// Initialize all of the kernels to make them go faster.
#define X(T, H, W) INSTANTIATE_MORPHO_SPECIALIZED(T, H, W)
MORPHO_SPECIALIZATION_LIST
#undef X
