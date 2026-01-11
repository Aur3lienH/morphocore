#include "smorph_kernels.cuh"
#include "smorph.cuh"
#include "morpho_merge_policy.cuh"
#include "atomic_ops.cuh"

/// @brief CUDA kernel for forward pass of soft morphological operations with IDENTITY merge.
/// Each output channel processes only its corresponding input channel.
/// Constraint: in_channels == out_channels
/// Weight shape: [channels, 1, kernel_height, kernel_width]
template <typename scalar_t>
__global__ void smorph_forward_identity_kernel(
    const scalar_t* input, 
    const scalar_t* weight,
    const scalar_t* alpha,
    scalar_t* output,
    scalar_t* references_indexes,
    scalar_t* sum_values,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
)
{
    extern __shared__ char shared_mem[];
    scalar_t* shared_weight = (scalar_t*)shared_mem;
    const int total_weight_size = channels * kernel_height * kernel_width;
    scalar_t* shared_input = shared_weight + total_weight_size;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
    const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;

    const int padded_tile_size = PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;

    const int threads_per_block = blockDim.x * blockDim.y;

    // Load weights into shared memory
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < total_weight_size; idx += threads_per_block) {
        if (idx < total_weight_size) 
        {
           shared_weight[idx] = weight[idx];
        }
    }
    __syncthreads();

    const int tile_start_x = blockIdx.x * TILE_WIDTH - kernel_width_half;
    const int tile_start_y = blockIdx.y * TILE_HEIGHT - kernel_height_half;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    // Load input tile into shared memory
    #pragma unroll
    for(int c = 0; c < channels; c++)
    {
        #pragma unroll
        for (int i = tid; i < PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH; i += total_threads) {
            
            int local_y = i / PADDED_TILE_WIDTH;
            int local_x = i % PADDED_TILE_WIDTH;
            
            int global_h = tile_start_y + local_y;
            int global_w = tile_start_x + local_x;
            
            int shared_idx = c * padded_tile_size + local_y * PADDED_TILE_WIDTH + local_x;
            
            // Check if position is valid
            if (global_h >= 0 && global_h < height && global_w >= 0 && global_w < width) {
                int global_idx = b * channels * height * width + 
                                c * height * width +
                                global_h * width + 
                                global_w;
                
                shared_input[shared_idx] = input[global_idx];
            } else {
                shared_input[shared_idx] = std::numeric_limits<scalar_t>::quiet_NaN();
            }
        }
    }
    
    __syncthreads();
    
    if (h < height && w < width && b < batch_size) {
        const int local_h = threadIdx.y;
        const int local_w = threadIdx.x;
        
        // Process each channel independently (IDENTITY: c_out == c_in)
        for(int c = 0; c < channels; c++)
        {
            const scalar_t alpha_val = alpha[c];
            const scalar_t abs_alpha = abs(alpha_val);
            const bool is_dilation_like = alpha_val >= 0;
            
            scalar_t reference_value = is_dilation_like ? 
                std::numeric_limits<scalar_t>::lowest() : 
                std::numeric_limits<scalar_t>::max();
            
            // First pass: find reference value (max for dilation, min for erosion)
            int k = 0;
            #pragma unroll
            for (int kh = 0; kh < kernel_height; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_width; kw++, k++) {
                    int shared_h = local_h + kernel_height_half + kernel_height_half - kh;
                    int shared_w = local_w + kernel_width_half + kernel_width_half - kw;
                    
                    int input_index = 
                        shared_w 
                        + shared_h * PADDED_TILE_WIDTH
                        + c * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT;
                    
                    scalar_t input_val = shared_input[input_index];
                    
                    if (!isnan(input_val)) {
                        int weight_index = k + c * kernel_width * kernel_height;
                        
                        scalar_t weight_val = shared_weight[weight_index];
                        scalar_t val = input_val + weight_val;
                        
                        if (is_dilation_like) {
                            reference_value = max(val, reference_value);
                        } else {
                            reference_value = min(val, reference_value);
                        }
                    }
                }
            }

            // Second pass: compute sum of exponentials
            scalar_t sum = 0.0;
            k = 0;
            #pragma unroll
            for (int kh = 0; kh < kernel_height; kh++) 
            {
                #pragma unroll
                for (int kw = 0; kw < kernel_width; kw++, k++) 
                {
                    int shared_h = local_h + kernel_height_half + kernel_height_half - kh;
                    int shared_w = local_w + kernel_width_half + kernel_width_half - kw;
                    
                    int input_index = 
                        shared_w 
                        + shared_h * PADDED_TILE_WIDTH
                        + c * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT;
                    
                    scalar_t input_val = shared_input[input_index];
                    
                    if (!isnan(input_val)) {
                        int weight_index = k + c * kernel_width * kernel_height;
                        
                        scalar_t weight_val = shared_weight[weight_index];
                        scalar_t val = input_val + weight_val;
                        
                        scalar_t exp_term = is_dilation_like ? 
                            exp((val - reference_value) * abs_alpha) : 
                            exp((reference_value - val) * abs_alpha);
                        
                        sum += exp_term;
                    }
                }
            }
                        
            // Third pass: compute weighted result
            scalar_t weighted_result = 0.0;
            k = 0;
            #pragma unroll
            for (int kh = 0; kh < kernel_height; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_width; kw++, k++) {
                    int shared_h = local_h + kernel_height_half + kernel_height_half - kh;
                    int shared_w = local_w + kernel_width_half + kernel_width_half - kw;

                    int input_index = 
                        shared_w 
                        + shared_h * PADDED_TILE_WIDTH
                        + c * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT;

                    scalar_t input_val = shared_input[input_index];
                    
                    if (!isnan(input_val)) {
                        int weight_index = k + c * kernel_width * kernel_height;

                        scalar_t weight_val = shared_weight[weight_index];
                        scalar_t val = input_val + weight_val;
                        
                        scalar_t exp_term = is_dilation_like ? 
                            exp((val - reference_value) * abs_alpha) : 
                            exp((reference_value - val) * abs_alpha);

                        scalar_t soft_weight = exp_term / sum;
                        weighted_result += soft_weight * val;
                    }
                }
            }
            
            // Write output directly (no merge needed)
            output[b * channels * height * width + c * height * width + h * width + w] = weighted_result;
        }
    }
}