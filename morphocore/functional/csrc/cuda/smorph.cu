#include "smorph.cuh"

template <typename scalar_t>
__global__ void morpho_smorph_forward_kernel(
    const scalar_t* input, 
    const scalar_t* weight,
    const scalar_t* alpha,
    scalar_t* output,
    scalar_t* references_indexes,
    scalar_t* sum_values,
    int batch_size,
    int input_channel,
    int output_channel,
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
    const int total_weight_size = output_channel * input_channel * kernel_height * kernel_width;
    scalar_t* shared_input = shared_weight + total_weight_size;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
    const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;

    const int padded_tile_size = PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;
    const int total_input_size = input_channel * padded_tile_size;

    const int threads_per_block = blockDim.x * blockDim.y;

    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < total_weight_size; idx += threads_per_block) {
        if (idx < total_weight_size) 
        {
           shared_weight[idx] = weight[idx];
        }
    }
    __syncthreads();

    // Load input data into shared memory (including padding)
    const int tile_start_x = blockIdx.x * TILE_WIDTH - kernel_width_half;
    const int tile_start_y = blockIdx.y * TILE_HEIGHT - kernel_height_half;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    #pragma unroll
    for(int ci = 0; ci < input_channel; ci++)
    {
        #pragma unroll
        for (int i = tid; i < PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH; i += total_threads) {
            int local_y = i / PADDED_TILE_WIDTH;
            int local_x = i % PADDED_TILE_WIDTH;
            
            int global_h = tile_start_y + local_y;
            int global_w = tile_start_x + local_x;
            
            // Apply boundary clamping
            global_h = min(max(global_h, 0), height - 1);
            global_w = min(max(global_w, 0), width - 1);
            int shared_idx = ci * padded_tile_size + local_y * PADDED_TILE_WIDTH + local_x;
            int global_idx = b * input_channel * height * width + 
                            ci * height * width +
                            global_h * width + 
                            global_w;
            
            shared_input[shared_idx] = input[global_idx];
        }
    }
    
    __syncthreads();
    if (h < height && w < width && b < batch_size) {
        
        const int local_h = threadIdx.y;
        const int local_w = threadIdx.x;
        for(int co = 0; co < output_channel; co++)
        {
            scalar_t res = 0;
            for(int ci = 0; ci < input_channel; ci++)
            {
                int global_index = 
                                b * input_channel * output_channel * height * width 
                                + co * input_channel * height * width 
                                + ci * height * width
                                + h * width 
                                + w;
                scalar_t reference_value = alpha[0] >= 0 ? std::numeric_limits<scalar_t>::lowest() : std::numeric_limits<scalar_t>::max();
                #pragma unroll
                for (int kh = 0; kh < kernel_height; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int shared_h = local_h + kh;
                        int shared_w = local_w + kw;

                        if (shared_h < PADDED_TILE_HEIGHT && shared_w < PADDED_TILE_WIDTH) {
                            
                            int input_index = 
                                    shared_w 
                                    + shared_h * PADDED_TILE_WIDTH
                                    + ci * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT;
                            scalar_t input_val = shared_input[input_index];
                            int weight_index = 
                                    kw
                                    + kh * kernel_width
                                    + ci * kernel_width * kernel_height
                                    + co * input_channel * kernel_width * kernel_height;
                            scalar_t weight_val = shared_weight[weight_index];
                                                        
                            scalar_t val = input_val + weight_val;
                            
                            reference_value = alpha[0] >= 0 ? max(val, reference_value) : min(val, reference_value);
                        }
                        
                    }
                }

                references_indexes[global_index] = reference_value;

                scalar_t sum = 0.0;
                #pragma unroll
                for (int kh = 0; kh < kernel_height; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int shared_h = local_h + kh;
                        int shared_w = local_w + kw;
                        if (shared_h < PADDED_TILE_HEIGHT && shared_w < PADDED_TILE_WIDTH) {
                            int input_index = 
                                    shared_w 
                                    + shared_h * PADDED_TILE_WIDTH
                                    + ci * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT;
                            int weight_index = 
                                    kw
                                    + kh * kernel_width
                                    + ci * kernel_width * kernel_height
                                    + co * input_channel * kernel_width * kernel_height;
                            scalar_t input_val = shared_input[input_index];
                                                    
                            scalar_t weight_val = shared_weight[weight_index];
                            scalar_t val = input_val + weight_val;
                            sum += alpha[0] >= 0 ? __expf((val - reference_value) * alpha[0]) : __expf((reference_value - val) * (-alpha[0]));
                        }
                        
                    }
                }
                sum_values[global_index] = sum;
                
                #pragma unroll
                for (int kh = 0; kh < kernel_height; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int shared_h = local_h + kh;
                        int shared_w = local_w + kw;
                        if (shared_h < PADDED_TILE_HEIGHT && shared_w < PADDED_TILE_WIDTH) {
                            int input_index = 
                                    shared_w 
                                    + shared_h * PADDED_TILE_WIDTH
                                    + ci * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT;
                            int weight_index = 
                                    kw
                                    + kh * kernel_width
                                    + ci * kernel_width * kernel_height
                                    + co * input_channel * kernel_width * kernel_height;

                            scalar_t input_val = shared_input[input_index];
                                                    
                            scalar_t weight_val = shared_weight[weight_index];

                            scalar_t val = input_val + weight_val;
                            scalar_t soft_weight = alpha[0] >= 0 ? __expf((val - reference_value) * alpha[0]) / sum : __expf((reference_value - val) * (-alpha[0])) / sum;

                            res += soft_weight * val;
                        }
                    }
                }
            }
            output[b * output_channel * height * width + co * height * width + h * width + w] = res;
        }
        
        
    }
}



template <typename scalar_t>
__global__ void morpho_smorph_backward_kernel(
    const scalar_t* grad_output, 
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    const scalar_t* alpha,
    const scalar_t* input,
    const scalar_t* weights,
    const scalar_t* output,
    const scalar_t* reference_values,
    const scalar_t* sum_values,
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
    scalar_t* alpha_accu = (scalar_t*)shared_mem;
    scalar_t* weight_accu = alpha_accu + 1;

    // Calculate position in output tensor
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *alpha_accu = 0.0f;  // Initialize alpha gradient accumulator
    }

    // Initialize weight gradient accumulators
    #pragma unroll
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel_height * kernel_width; i += blockDim.x * blockDim.y) 
    {
        if (i < kernel_height * kernel_width) 
        {
            weight_accu[i] = 0.0f;
        }
    }

    __syncthreads();
    
    if (h < height && w < width && b < batch_size) {
        
        const int c = 0;
        const scalar_t alpha_val = alpha[0];
        scalar_t alpha_grad = 0.0f;
        const scalar_t gradient_out = grad_output[b * channels * height * width + c * height * width + h * width + w];
        const scalar_t smorph_out = output[b * channels * height * width + c * height * width + h * width + w];
        const scalar_t reference_value = reference_values[b * channels * height * width + c * height * width + h * width + w];
        const scalar_t sum = sum_values[b * channels * height * width + c * height * width + h * width + w];



        #pragma unroll
        for (int kh = 0; kh < kernel_height; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kernel_width; kw++) {


                int in_h = min(max(h + kh - kernel_height_half, 0), height - 1);
                int in_w = min(max(w + kw - kernel_width_half, 0), width - 1);


                int64_t unfolded_index = kw + 
                                        kh*kernel_width + 
                                        w*kernel_width*kernel_height + 
                                        h*width*kernel_height*kernel_width + 
                                        c*height*width*kernel_height*kernel_width + 
                                        b*channels*height*width*kernel_height*kernel_width;
                
                int64_t input_index = b * channels * height * width + 
                    c * height * width + 
                    in_h * width + 
                    in_w;
                scalar_t input_value = input[input_index];
                int weight_idx = kh * kernel_width + kw;
                scalar_t weight_value = weights[weight_idx];
                scalar_t val = input_value + weight_value;
                scalar_t p = alpha[0] >= 0 ? __expf((val - reference_value) * alpha[0]) / sum : __expf((reference_value - val) * (-alpha[0])) / sum;

                scalar_t grad_val = p + alpha_val * p * (val - smorph_out);
                grad_val *= gradient_out;

                atomicAdd(&grad_input[input_index], grad_val);
                atomicAdd(&weight_accu[weight_idx], grad_val);
                alpha_grad += p * val * (val - smorph_out);

            }
        }
        atomicAdd(alpha_accu, alpha_grad * gradient_out);

        
    }

    __syncthreads();

    #pragma unroll
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel_height * kernel_width; i += blockDim.x * blockDim.y) {
        if (i < kernel_height * kernel_width) {
            atomicAdd(&grad_weight[i], weight_accu[i]);
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        
        atomicAdd(&grad_alpha[0], alpha_accu[0]);

    }
}

template __global__ void morpho_smorph_forward_kernel(
    const float* input, 
    const float* weight,
    const float* alpha,
    float* output,
    float* reference_values,
    float* max_values,
    int batch_size,
    int in_channel,
    int out_channel,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);

template __global__ void morpho_smorph_forward_kernel(
    const double* input, 
    const double* weight,
    const double* alpha,
    double* output,
    double* reference_values,
    double* max_values,
    int batch_size,
    int in_channel,
    int out_channel,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);


template __global__ void morpho_smorph_backward_kernel(
    const float* grad_output, 
    float* grad_input,
    float* grad_weight,
    float* grad_alpha,
    const float* alpha,
    const float* input,
    const float* weights,
    const float* output,
    const float* reference_values,
    const float* sum_values,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);

template __global__ void morpho_smorph_backward_kernel(
    const double* grad_output, 
    double* grad_input,
    double* grad_weight,
    double* grad_alpha,
    const double* alpha,
    const double* input,
    const double* weights,
    const double* output,
    const double* reference_values,
    const double* sum_values,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);