#include "erosion.hpp"

#include <torch/torch.h>
#include <torch/extension.h>
#include <omp.h>



#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../cuda/erosion.cuh"
#endif




// Check if CUDA is available and input is on CUDA devic
torch::Tensor MorphoErosion::forward(
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
    torch::Tensor min_index;
    
    if (input.is_cuda()) {
        #ifdef WITH_CUDAS
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
        min_index = torch::zeros({batch_size, weight_shape[0], height, width}, 
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
        
        // Launch specialized kernels for know often used kernels
        if (kernel_height == 30 && kernel_width == 30) {
            morpho_erosion_forward_specialized_kernel<float, 30, 30><<<blocks, threads, shared_memory_size>>>(
                padded_input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                min_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width
            );
        }
        else if(kernel_height == 5 && kernel_width == 5)
        {
            morpho_erosion_forward_specialized_kernel<float, 5, 5><<<blocks, threads, shared_memory_size>>>(
                padded_input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                min_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width
            );
        } 
        else {
            morpho_erosion_forward_kernel<float><<<blocks, threads, shared_memory_size>>>(
                padded_input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                min_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width,
                padded_input.size(2),
                padded_input.size(3),
                kernel_height_half,
                kernel_width_half
            );
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
        //output = morpho_dilaton_forward_cpu(input, weight);
        min_index = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, 
                               input.options().dtype(torch::kLong));
    }
    ctx->save_for_backward({input, weight, min_index});
    return output;
}


torch::autograd::tensor_list MorphoErosion::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto min_index = saved[2];
    auto grad_output = grad_outputs[0];
    
    torch::Tensor grad_input;
    torch::Tensor grad_weight;
    
    if (input.is_cuda()) {
        #ifdef WITH_CUDAS
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

        if (kernel_height == 30 && kernel_width == 30) {
            morpho_erosion_backward_specialized_kernel<float, 30, 30><<<blocks, threads, shared_mem_size>>>(
                grad_input.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                min_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width
            );
        }
        else if(kernel_height == 5 && kernel_width == 5)
        {
            morpho_erosion_backward_specialized_kernel<float, 5, 5><<<blocks, threads, shared_mem_size>>>(
                grad_input.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                min_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width
            );
        }
        else
        {
            morpho_erosion_backward_kernel<float><<<blocks, threads, shared_mem_size>>>(
                grad_input.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                min_index.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width,
                kernel_height,
                kernel_width,
                kernel_height_half,
                kernel_width_half);
        }
            
        // Launch kernel based on scalar type
        // AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_erosion_backward_cuda", ([&] {
        //     morpho_erosion_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
        //         grad_input.data_ptr<scalar_t>(),
        //         grad_weight.data_ptr<scalar_t>(),
        //         grad_output.data_ptr<scalar_t>(),
        //         min_index.data_ptr<int64_t>(),
        //         batch_size,
        //         channels,
        //         height,
        //         width,
        //         kernel_height,
        //         kernel_width,
        //         kernel_height_half,
        //         kernel_width_half
        //     );
        // }));



        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        #else
        TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } else {
        // Use CPU implementation
        // auto result = morpho_dilaton_backward_cpu(
        //     input, 
        //     weight, 
        //     min_index,
        //     grad_output
        // );
        // grad_input = result[0];
        // grad_weight = result[1];
    }
    return {grad_input, grad_weight};
}