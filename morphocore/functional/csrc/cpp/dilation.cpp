#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <omp.h>

#include "cpp/dilation.hxx"

#ifdef WITH_CUDA
#include "cuda/dilation.cuh"
#endif



// Check if CUDA is available and input is on CUDA devic
torch::Tensor MorphoDilation::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor input,
    torch::Tensor weight
) {
    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    
    // Uncomment validation checks for safer code
    // TORCH_CHECK(input.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    // TORCH_CHECK(weight.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");
    // TORCH_CHECK(input_shape[1] == weight_shape[1], "Input channels must match weight input channels");
    // TORCH_CHECK(input.sizes()[1] == 1, "Input Channels different to 1 not yet implemented!");
    
    torch::Tensor output;
    torch::Tensor max_index;
    
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
        auto batch_size = input_shape[0];
        auto channels = input_shape[1]; // Should be 1 because only one channel for the moment !!! TODO
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;
        
        // Create output tensors (For faster processing don't create them each time)
        output = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        max_index = torch::zeros({batch_size, weight_shape[0], height, width}, 
            input.options().dtype(torch::kLong));
        
        // Padding replicate
        auto padded_input = torch::nn::functional::pad(input,
            torch::nn::functional::PadFuncOptions(
                {kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}
            ).mode(torch::kReplicate));
        
        dim3 threads(32, 8);
        dim3 blocks((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y,
                  batch_size);
        
        int shared_memory_size = sizeof(float) * kernel_width * kernel_height;
        
        #define X(TYPE, H, W) \
        if constexpr (std::is_same_v<T, TYPE>) { \
            if (kernel_h == H && kernel_w == W) { \
                return morpho_dilaton_forward<TYPE, H, W>(
                    
                ); \
            } \
        }
        MORPHO_SPECIALIZATION_LIST
        #undef X
        else
        {
            
        }
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        #else
        TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } else {
        // CPU implementation TODO !!!
        output = morpho_dilaton_forward_cpu(input, weight);
        max_index = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, 
                               input.options().dtype(torch::kLong));
    }
    ctx->save_for_backward({input, weight, max_index});
    return output;
}


torch::autograd::tensor_list MorphoDilation::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto max_index = saved[2];
    auto grad_output = grad_outputs[0];
    
    torch::Tensor grad_input;
    torch::Tensor grad_weight;
    
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
        auto input_shape = input.sizes();
        auto weight_shape = weight.sizes();
        auto batch_size = input_shape[0];
        auto channels = input_shape[1]; 
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;
        
        // Create gradient tensors (Should not be initialized every time for better performances !!!!!!)
        grad_input = torch::zeros_like(input);
        grad_weight = torch::zeros_like(weight);
        
        dim3 threads(32, 8);
        dim3 blocks((width + threads.x - 1) / threads.x,
                   (height + threads.y - 1) / threads.y,
                    batch_size);

        int shared_mem_size = kernel_height * kernel_width * sizeof(float);

        if(kernel_height == 5 && kernel_width == 5)
        {
            morpho_dilation_backward_specialized_kernel<float, 5, 5><<<blocks, threads, shared_mem_size>>>(
                grad_input.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                max_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width
            );
        }
        else
        {
            morpho_dilation_backward_kernel<float><<<blocks, threads, shared_mem_size>>>(
                grad_input.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                max_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width,
                kernel_height,
                kernel_width,
                kernel_height_half,
                kernel_width_half);
        }
        



        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        #else
        TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } else {
        // Use CPU implementation
        auto result = morpho_dilaton_backward_cpu(
            input, 
            weight, 
            max_index,
            grad_output
        );
        grad_input = result[0];
        grad_weight = result[1];
    }
    return {grad_input, grad_weight};
}


torch::Tensor morpho_dilaton_forward_cpu(torch::Tensor input, torch::Tensor weight) {
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
    torch::Tensor max_index = torch::zeros({batch_size, out_channels, height, width}, 
                                          input.options().dtype(torch::kLong));

    auto padded_input = torch::nn::functional::pad(
        input, 
        torch::nn::functional::PadFuncOptions({kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half})
        .mode(torch::kReplicate)
    );
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; b++) {


        for (int64_t h = 0; h < height; h++) {
            for (int64_t w = 0; w < width; w++) {

                float max_value = -std::numeric_limits<float>::infinity();
                int64_t max_flat_index = 0;


                for (int64_t kh = 0; kh < kernel_height; kh++) {
                    for (int64_t kw = 0; kw < kernel_width; kw++) {
                        int64_t ph = h + kh;
                        int64_t pw = w + kw;
                        float val = padded_input.index({b, 0, ph, pw}).item<float>() + 
                                   weight.index({0, 0, kh, kw}).item<float>();
                        int64_t flat_index = kh * kernel_width + kw;
                        if (val > max_value || (val == max_value && flat_index < max_flat_index)) {
                            max_flat_index = flat_index;
                            max_value = val;
                        }
                    }
                }
                max_index.index_put_({b, 0, h, w}, max_flat_index);

                output.index_put_({b, 0, h, w}, max_value);
            }
        }
    }
    
    return output;
}

torch::autograd::tensor_list morpho_dilaton_backward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor max_index,
    torch::Tensor grad_output) {
    
    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    
    auto batch_size = input_shape[0];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;
    
    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; b++) {
        auto local_grad_weight = torch::zeros_like(weight);
        for (int64_t h = 0; h < height; h++) {
            for (int64_t w = 0; w < width; w++) {
                int64_t flat_idx = max_index.index({b, 0, h, w}).item<int64_t>();
                int64_t max_h = flat_idx / kernel_width;
                int64_t max_w = flat_idx % kernel_width;
                
                float grad_val = grad_output.index({b, 0, h, w}).item<float>();
                
                int64_t ih = h + max_h - kernel_height_half;
                int64_t iw = w + max_w - kernel_width_half;
                
                int64_t padded_ih = std::max(static_cast<int64_t>(0), std::min(height - 1, ih));
                int64_t padded_iw = std::max(static_cast<int64_t>(0), std::min(width - 1, iw));
                
                grad_input.index({b, 0, padded_ih, padded_iw}) += grad_val;
                local_grad_weight.index({0, 0, max_h, max_w}) += grad_val;
            }
        }
        #pragma omp critical
        {
            grad_weight += local_grad_weight;
        }
    }
    
    return {grad_input, grad_weight};
}

