#include "smorph.cuh"
#include <torch/torch.h>

#include "cpp/smorph.hxx"

#ifdef WITH_CUDA
#include "cuda/smorph.cuh"
#endif


torch::Tensor Smorph::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor alpha
) 
{
    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    TORCH_CHECK(input.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");
    TORCH_CHECK(input_shape[1] == weight_shape[1], "Input channels must match weight input channels");
    TORCH_CHECK(input.sizes()[1] == 1, "Input Channels different to 1 not yet implemented!");

    if(is_cuda(input))
    {
        #ifdef WITH_CUDA
        auto batch_size = input_shape[0];
        auto channels = input_shape[1]; 
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;
        
        dim3 threads(32, 8);
        dim3 blocks((width + threads.x - 1) / threads.x,
                   (height + threads.y - 1) / threads.y,
                    batch_size);
        cudaFuncSetAttribute(morpho_smorph_forward_kernel<float>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, 
            cudaSharedmemCarveoutMaxShared);
        // Create output tensors on the same device as input
        torch::Tensor output = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        torch::Tensor reference_values = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        torch::Tensor max_values = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
        const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_forward_kernel", ([&] {
            int64_t shared_memory_size = sizeof(scalar_t) * kernel_height * kernel_width +  // Weight buffer
            sizeof(scalar_t) * PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;
            morpho_smorph_forward_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), reference_values.data_ptr<scalar_t>(), max_values.data_ptr<scalar_t>(), batch_size, 1, 1, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
        }));

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel params: batch=%d, channels=%d, height=%d, width=%d, kernel=(%d,%d), half=(%d,%d)\n",
                batch_size, channels, height, width, kernel_height, kernel_width, 
                kernel_height_half, kernel_width_half);
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        ctx->save_for_backward({output, reference_values, max_values, alpha, input, weight});
        return output;
        #endif
    }
    else
    {
        TORCH_CHECK(false, "CPU version not implemented yet !");
    }
}

torch::autograd::tensor_list Smorph::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs
) {
    if(is_cuda(grad_outputs[0]))
    {
        #ifdef WITH_CUDA
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto values = saved[1];
        auto probabilities = saved[2];
        auto alpha = saved[3];
        auto input = saved[4];
        auto weight = saved[5];
        
        auto input_shape = input.sizes();
        auto weight_shape = weight.sizes();
        auto batch_size = input_shape[0];
        auto channels = input_shape[1]; // Should be 1
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;

        torch::Tensor grad_input = torch::zeros_like(input);
        torch::Tensor grad_weight = torch::zeros_like(weight);
        torch::Tensor grad_alpha = torch::zeros_like(alpha);
        int shared_memory_size = 0;
        shared_memory_size += sizeof(float); //alpha
        shared_memory_size += sizeof(float) * kernel_height * kernel_width;

        dim3 threads(32, 8);
        dim3 blocks((width + threads.x - 1) / threads.x,
                   (height + threads.y - 1) / threads.y,
                    batch_size);
        cudaFuncSetAttribute(morpho_smorph_backward_kernel<float>, 
                        cudaFuncAttributePreferredSharedMemoryCarveout, 
                        cudaSharedmemCarveoutMaxShared);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_backward_kernel", ([&] {
            morpho_smorph_backward_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
                grad_outputs[0].data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                grad_alpha.data_ptr<scalar_t>(),
                alpha.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                probabilities.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                kernel_height,
                kernel_width,
                kernel_height_half,
                kernel_width_half
            );
        }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        return {grad_input, grad_weight, grad_alpha};
        #endif
    }
    else
    {
        TORCH_CHECK(false, "CPU version not implemented yet !");
    }
}