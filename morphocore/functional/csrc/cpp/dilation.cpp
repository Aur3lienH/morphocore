#include "dilation.hpp"
#include <torch/torch.h>
#include <torch/extension.h>
#include <omp.h>


#include "reshaper.hpp"


#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../cuda/dilation.cuh"
#endif



torch::Tensor MorphoDilation::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor input,
    torch::Tensor weight
) {

    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);

    auto input_shape = input_correct_dims.sizes();
    auto weight_shape = weight_correct_dims.sizes();
    auto batch_size = input_shape[0];
    auto channels = input_shape[1];
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;

    // Uncomment validation checks for safer code
    // TORCH_CHECK(input.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    // TORCH_CHECK(weight.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");
    // TORCH_CHECK(input_shape[1] == weight_shape[1], "Input channels must match weight input channels");
    // TORCH_CHECK(input.sizes()[1] == 1, "Input Channels different to 1 not yet implemented!");
    torch::Tensor output;    
    torch::Tensor max_indices;
    
    
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
        
        output = torch::zeros({batch_size, weight_shape[0], height, width}, input_correct_dims.options());
        
        max_indices = torch::zeros({batch_size, weight_shape[0], height, width}, 
                                       torch::TensorOptions(input_correct_dims.device()).dtype(torch::kLong));
        
        auto padded_input = torch::constant_pad_nd(
            input_correct_dims, 
            torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, 
            -std::numeric_limits<float>::infinity()  // Use float for the constant, it will be cast appropriately
        );
        
        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;
        

        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "morpho_dilation_forward", ([&] {
            forward_kernel<scalar_t>(
                padded_input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                max_indices.data_ptr<int64_t>(),
                batch_size,
                channels,
                height,
                width,
                padded_height,
                padded_width,
                kernel_height,
                kernel_width
            );
        }));
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        
        // Synchronize to ensure kernel completion (optional, for debugging)
        // cudaDeviceSynchronize();
        
        #else
        TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } 
    else 
    {
        output = AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "morpho_dilaton_forward_cpu", [&] {
                                                return morpho_dilaton_forward_cpu<scalar_t>(input_correct_dims, weight_correct_dims);
                                            });
    }
    output = input_dim_manager.ToOriginalTensor(output);
    ctx->save_for_backward({input_correct_dims, weight_correct_dims, max_indices});
    return output;
}


torch::autograd::tensor_list MorphoDilation::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto max_indices = saved[2];
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
    
        
        // Calculate padded dimensions (needed for backward kernel)
        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;
        
        // Dispatch to the appropriate backward kernel based on tensor type
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_dilation_backward", ([&] {
            backward_kernel<scalar_t>(
                grad_input.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                max_indices.data_ptr<int64_t>(),  
                batch_size,
                channels,
                height,
                width,
                padded_height,
                padded_width,
                kernel_height,
                kernel_width
            );
            
            // ALTERNATIVE: Si vous n'avez pas max_indexes, utilisez le kernel générique uniquement
            // morpho_dilation_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            //     grad_input.data_ptr<scalar_t>(),
            //     grad_weight.data_ptr<scalar_t>(),
            //     grad_output.data_ptr<scalar_t>(),
            //     nullptr,  // max_indexes non disponible
            //     batch_size, channels, height, width,
            //     kernel_height, kernel_width, kernel_height_half, kernel_width_half
            // );
        }));
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        
        // Optional: Synchronize for debugging
        // cudaDeviceSynchronize();
        
        #else
            TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } 
    else 
    {
        // Use CPU implementation
        auto result = AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_dilaton_backward_cpu", [&] {
                                                return morpho_dilaton_backward_cpu<scalar_t>(input, weight, grad_output);
                                            });
        grad_input = result[0];
        grad_weight = result[1];
    }
    return {grad_input, grad_weight};
}

// torch::autograd::tensor_list morpho_dilaton_backward_cpu_indices(
//     torch::Tensor input,
//     torch::Tensor weight,
//     torch::Tensor max_indices,
//     torch::Tensor grad_output) {
    
//     auto input_shape = input.sizes();
//     auto weight_shape = weight.sizes();
    
//     auto batch_size = input_shape[0];
//     auto height = input_shape[2];
//     auto width = input_shape[3];
//     auto kernel_height = weight_shape[2];
//     auto kernel_width = weight_shape[3];
//     auto kernel_height_half = kernel_height / 2;
//     auto kernel_width_half = kernel_width / 2;
    
//     auto grad_input = torch::zeros_like(input);
//     auto grad_weight = torch::zeros_like(weight);
    
//     for (int64_t b = 0; b < batch_size; b++) {
//         auto local_grad_weight = torch::zeros_like(weight);
        
//         for (int64_t h = 0; h < height; h++) {
//             for (int64_t w = 0; w < width; w++) {
//                 // Get the winning kernel position
//                 int64_t flat_idx = max_indices.index({b, 0, h, w}).item<int64_t>();
//                 int64_t max_h = flat_idx / kernel_width;
//                 int64_t max_w = flat_idx % kernel_width;
                
//                 float grad_val = grad_output.index({b, 0, h, w}).item<float>();
                
//                 // ✅ CORRECT: Calculate input position using inverse of forward formula
//                 // Forward used: ph = h - kh + 2 * kernel_height_half
//                 // So input position is: ih = (ph - h + kh) - kernel_height_half
//                 // Which simplifies to: ih = h - max_h + kernel_height_half
//                 int64_t ih = h - max_h + kernel_height_half;
//                 int64_t iw = w - max_w + kernel_width_half;
                
//                 // ✅ CORRECT: Check bounds and only accumulate if valid
//                 if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
//                     grad_input.index({b, 0, ih, iw}) += grad_val;
//                 }
//                 // If ih/iw are out of bounds, they correspond to padded regions
//                 // which don't exist in the original input, so we ignore them
                
//                 // ✅ CORRECT: Always accumulate weight gradient
//                 local_grad_weight.index({0, 0, max_h, max_w}) += grad_val;
//             }
//         }
        
//         #pragma omp critical
//         {
//             grad_weight += local_grad_weight;
//         }
//     }
    
//     return {grad_input, grad_weight};
// }


