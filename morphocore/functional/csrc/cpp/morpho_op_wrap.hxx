#pragma once
#include "morpho_op.hpp"


template <MorphoOpertion op>
torch::Tensor MorphoOpertion<op>::forward(torch::autograd::AutogradContext* ctx,torch::Tensor input,torch::Tensor weight) 
{

    using Policy = MorphoPolicyCPU
    
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);

    auto input_shape = input_correct_dims.sizes();
    auto weight_shape = weight_correct_dims.sizes();
    auto batch_size = input_shape[0];
    auto in_channels = input_shape[1];        // ✅ Renamed for clarity
    auto out_channels = weight_shape[0];      // ✅ Added out_channels
    auto height = input_shape[2];
    auto width = input_shape[3];
    auto kernel_height = weight_shape[2];
    auto kernel_width = weight_shape[3];
    auto kernel_height_half = kernel_height / 2;
    auto kernel_width_half = kernel_width / 2;

    // ✅ Validation checks (now with proper channel names)
    TORCH_CHECK(input.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");
    TORCH_CHECK(input_shape[1] == weight_shape[1], "Input channels must match weight input channels");
    
    torch::Tensor output;    
    torch::Tensor max_indices;
    
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
        
        output = torch::zeros({batch_size, out_channels, height, width}, input_correct_dims.options());
        
        max_indices = torch::zeros({batch_size, out_channels, height, width}, 
                                        torch::TensorOptions(input_correct_dims.device()).dtype(torch::kLong));
        
        auto padding_value = get_padding_value<Op>(input_correct_dims.scalar_type());
        
        auto padded_input = torch::constant_pad_nd(
            input_correct_dims, 
            torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, 
            padding_value
        );
        
        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;
        
        constexpr const char* operation_name = (Op == MorphoOp::DILATION) ? "morpho_dilation_forward" : "morpho_erosion_forward";
        
        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), operation_name, ([&] {
            if constexpr (Op == MorphoOp::DILATION) {
                dilation_forward_kernel<scalar_t>(
                    padded_input.data_ptr<scalar_t>(),
                    weight_correct_dims.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    max_indices.data_ptr<int64_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    height,
                    width,
                    padded_height,
                    padded_width,
                    kernel_height,
                    kernel_width
                );
            } else {
                erosion_forward_kernel<scalar_t>(
                    padded_input.data_ptr<scalar_t>(),
                    weight_correct_dims.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    max_indices.data_ptr<int64_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    height,
                    width,
                    padded_height,
                    padded_width,
                    kernel_height,
                    kernel_width
                );
            }
        }));
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        
        #else
        TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } 
    else 
    {
        // ✅ CPU implementation with template dispatch
        constexpr const char* cpu_operation_name = (Op == MorphoOp::DILATION) ? "morpho_dilation_forward_cpu" : "morpho_erosion_forward_cpu";
        
        output = AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), cpu_operation_name, [&] {
            if constexpr (Op == MorphoOp::DILATION) {
                return morpho_dilation_forward_cpu<scalar_t>(input_correct_dims, weight_correct_dims);
            } else {
                return morpho_erosion_forward_cpu<scalar_t>(input_correct_dims, weight_correct_dims);
            }
        });
    }
    
    output = input_dim_manager.ToOriginalTensor(output);
    ctx->save_for_backward({input_correct_dims, weight_correct_dims, max_indices});
    return output;
}

torch::autograd::tensor_list MorphoOpertion<op>::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) 
{
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
        auto in_channels = input_shape[1];   
        auto out_channels = weight_shape[0];  
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;
        
        // Create gradient tensors
        grad_input = torch::zeros_like(input);
        grad_weight = torch::zeros_like(weight);
        
        // ✅ Recreate padded input for backward pass
        auto padding_value = get_padding_value<Op>(input.scalar_type());
        auto padded_input = torch::constant_pad_nd(
            input, 
            torch::IntList{kernel_width_half, kernel_width_half, kernel_height_half, kernel_height_half}, 
            padding_value
        );
        
        int padded_height = height + 2 * kernel_height_half;
        int padded_width = width + 2 * kernel_width_half;
        
        // ✅ Template-based dispatch with operation-specific naming
        constexpr const char* operation_name = (Op == MorphoOp::DILATION) ? "morpho_dilation_backward" : "morpho_erosion_backward";
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), operation_name, ([&] {
            // ✅ Use appropriate kernel based on operation
            if constexpr (Op == MorphoOp::DILATION) {
                dilation_backward_kernel<scalar_t>(
                    padded_input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    grad_input.data_ptr<scalar_t>(),
                    grad_weight.data_ptr<scalar_t>(),
                    grad_output.data_ptr<scalar_t>(),
                    max_indices.data_ptr<int64_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    height,
                    width,
                    padded_height,
                    padded_width,
                    kernel_height,
                    kernel_width
                );
            } else {
                erosion_backward_kernel<scalar_t>(
                    padded_input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    grad_input.data_ptr<scalar_t>(),
                    grad_weight.data_ptr<scalar_t>(),
                    grad_output.data_ptr<scalar_t>(),
                    max_indices.data_ptr<int64_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    height,
                    width,
                    padded_height,
                    padded_width,
                    kernel_height,
                    kernel_width
                );
            }
        }));
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        
        #else
        TORCH_CHECK(false, "CUDA not available for this build");
        #endif
    } 
    else 
    {
        // ✅ CPU implementation with template dispatch
        constexpr const char* cpu_operation_name = (Op == MorphoOp::DILATION) ? "morpho_dilation_backward_cpu" : "morpho_erosion_backward_cpu";
        
        auto result = AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), cpu_operation_name, [&] {
            if constexpr (Op == MorphoOp::DILATION) {
                return morpho_dilation_backward_cpu<scalar_t>(input, weight, grad_output);
            } else {
                return morpho_erosion_backward_cpu<scalar_t>(input, weight, grad_output);
            }
        });
        grad_input = result[0];
        grad_weight = result[1];
    }
    
    return {grad_input, grad_weight};
}