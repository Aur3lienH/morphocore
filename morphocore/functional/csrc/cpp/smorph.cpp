#include "smorph.hpp"
#include "smorph_cpu.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include <torch/torch.h>
#include "reshaper.hpp"
#include "smorph_cpu_forward_identity.hxx"
#include "smorph_cpu_backward_identity.hxx"

#ifdef WITH_CUDA
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

    #include "../cuda/smorph.cuh"
#endif

/// @brief Forward pass of the Soft Morphological layer (dispatcher between CPU and CUDA).
/// @tparam merge Merge channel strategy to use.
/// @param ctx Autograd context to save tensors for backward pass
/// @param input Input tensor of shape (batch_size, in_channels, height, width)
/// @param weight Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
/// @param alpha Learnable parameter controlling the operation either dilation or erosion or something between the two
/// @return Output tensor of shape (batch_size, out_channels, height, width)
template <MORPHO_MERGE merge>
torch::Tensor Smorph<merge>::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor alpha)
{
    TensorInputDim input_dim_manager(input);
    TensorWeightDim weight_dim_manager(weight);
    torch::Tensor input_correct_dims = input_dim_manager.To4DTensor(input);
    torch::Tensor weight_correct_dims = weight_dim_manager.To4DTensor(weight);
    auto input_shape = input_correct_dims.sizes();
    auto weight_shape = weight_correct_dims.sizes();

    TORCH_CHECK(input_correct_dims.dim() == 4, "Input must be a 4d tensor (batch_size, in_channels, height, width)");
    TORCH_CHECK(weight_correct_dims.dim() == 4, "Weight must be a 4d tensor (out_channels, in_channels, kernel_height, kernel_width)");

    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        auto batch_size = input_shape[0];
        auto in_channels = input_shape[1];
        auto out_channels = weight_shape[0];
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;

        // cudaFuncSetAttribute(morpho_smorph_forward_kernel<float>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        // Create output tensors on the same device as input
        torch::Tensor output = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        torch::Tensor reference_values = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        torch::Tensor max_values = torch::zeros({batch_size, weight_shape[0], height, width}, input.options());
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_forward_kernel", ([&] {
                                       if (merge == MORPHO_MERGE::IDENTITY)
                                       {
                                           TORCH_CHECK(false, "Identity merge not supported in CUDA yet");
                                       }
                                       else
                                       {
                                           morpho_smorph_forward_kernel<merge, scalar_t>(
                                               input_correct_dims.data_ptr<scalar_t>(), weight_correct_dims.data_ptr<scalar_t>(),
                                               alpha.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), reference_values.data_ptr<scalar_t>(),
                                               max_values.data_ptr<scalar_t>(), batch_size, in_channels, out_channels, height, width, kernel_height,
                                               kernel_width, kernel_height_half, kernel_width_half);
                                       }
                                   }));
        output = input_dim_manager.ToOriginalTensor(output);
        ctx->save_for_backward({output, reference_values, max_values, alpha, input_correct_dims, weight_correct_dims});
        return output;
#endif
    }
    else
    {
        torch::Tensor output = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        torch::Tensor reference_values = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        torch::Tensor max_values = torch::zeros({input_shape[0], weight_shape[0], input_shape[2], input_shape[3]}, input.options());
        AT_DISPATCH_FLOATING_TYPES(input_correct_dims.scalar_type(), "smorph_forward_cpu", [&] {
            if (merge == MORPHO_MERGE::IDENTITY)
            {
                output = smorph_forward_cpu_identity<scalar_t>(input_correct_dims, weight_correct_dims, alpha);
            }
            else
            {
                output = smorph_forward_cpu<merge, scalar_t>(input_correct_dims, weight_correct_dims, alpha);
            }
        });
        output = input_dim_manager.ToOriginalTensor(output);
        ctx->save_for_backward({output, reference_values, max_values, alpha, input_correct_dims, weight_correct_dims});
        return output;
    }
    return torch::Tensor();
}

/// @brief Backward pass of the Soft Morphological layer
/// @tparam merge Merge strategy to use
/// @param ctx Autograd context to retrieve saved tensors from forward pass
/// @param grad_outputs Gradient of the output tensor
/// @return Gradients of the input, weight and alpha tensors
template <MORPHO_MERGE merge>
torch::autograd::tensor_list Smorph<merge>::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs)
{

    auto saved = ctx->get_saved_variables();
    auto output = saved[0];
    auto values = saved[1];
    auto probabilities = saved[2];
    auto alpha = saved[3];
    auto input = saved[4];
    auto weight = saved[5];
    torch::Tensor grad_input = torch::zeros_like(input, grad_outputs[0].options());
    torch::Tensor grad_weight = torch::zeros_like(weight, grad_outputs[0].options());
    torch::Tensor grad_alpha = torch::zeros_like(alpha, grad_outputs[0].options());
    if (grad_outputs[0].is_cuda())
    {
#ifdef WITH_CUDA

        auto input_shape = input.sizes();
        auto weight_shape = weight.sizes();
        auto batch_size = input_shape[0];
        auto in_channels = input_shape[1];
        auto height = input_shape[2];
        auto width = input_shape[3];
        auto out_channels = weight_shape[0];
        auto kernel_height = weight_shape[2];
        auto kernel_width = weight_shape[3];
        auto kernel_height_half = kernel_height / 2;
        auto kernel_width_half = kernel_width / 2;

        // Ensure contiguous memory layout because CUDA is **** !
        auto input_contig = input.contiguous();
        auto weight_contig = weight.contiguous();
        auto alpha_contig = alpha.contiguous();
        auto grad_output_contig = grad_outputs[0].contiguous();

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_backward_kernel", ([&] {
                                       if (merge == MORPHO_MERGE::IDENTITY)
                                       {
                                           TORCH_CHECK(false, "Identity merge not supported in CUDA yet");
                                       }
                                       else
                                       {
                                           morpho_smorph_backward_kernel<merge, scalar_t>(
                                               grad_output_contig.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
                                               grad_weight.data_ptr<scalar_t>(), grad_alpha.data_ptr<scalar_t>(), alpha_contig.data_ptr<scalar_t>(),
                                               input_contig.data_ptr<scalar_t>(), weight_contig.data_ptr<scalar_t>(), batch_size, in_channels,
                                               out_channels, height, width, kernel_height, kernel_width, kernel_height_half, kernel_width_half);
                                       }
                                   }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
        }
        return {grad_input, grad_weight, grad_alpha};
#endif
    }
    else
    {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "morpho_smorph_backward_cpu", [&] {
            if (merge == MORPHO_MERGE::IDENTITY)
            {
                smorph_backward_cpu_identity<scalar_t>(input, weight, alpha, grad_outputs[0], grad_input, grad_weight, grad_alpha);
            }
            else
            {
                smorph_backward_cpu<merge, scalar_t>(input, weight, alpha, grad_outputs[0], grad_input, grad_weight, grad_alpha);
            }
        });
        return {grad_input, grad_weight, grad_alpha};
    }
    return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

template class Smorph<MORPHO_MERGE::MIN>;
template class Smorph<MORPHO_MERGE::MAX>;
template class Smorph<MORPHO_MERGE::AVERAGE>;
template class Smorph<MORPHO_MERGE::ADD>;
template class Smorph<MORPHO_MERGE::IDENTITY>;

// ----------------------------------SMORPH FORWARD PASS CPU----------------------------------------

#define INSTANTIATE_SMORPH_FORWARD(MERGE, TYPE)                                                                                                      \
    template auto smorph_forward_cpu<MORPHO_MERGE::MERGE, TYPE>(torch::Tensor, torch::Tensor, torch::Tensor);

INSTANTIATE_SMORPH_FORWARD(MIN, float)
INSTANTIATE_SMORPH_FORWARD(MAX, float)
INSTANTIATE_SMORPH_FORWARD(AVERAGE, float)
INSTANTIATE_SMORPH_FORWARD(ADD, float)
INSTANTIATE_SMORPH_FORWARD(MIN, double)
INSTANTIATE_SMORPH_FORWARD(MAX, double)
INSTANTIATE_SMORPH_FORWARD(AVERAGE, double)
INSTANTIATE_SMORPH_FORWARD(ADD, double)

template torch::Tensor smorph_forward_cpu_identity<float>(torch::Tensor, torch::Tensor, torch::Tensor);
template torch::Tensor smorph_forward_cpu_identity<double>(torch::Tensor, torch::Tensor, torch::Tensor);

// ----------------------------------SMORPH FORWARD PASS CPU----------------------------------------

// ----------------------------------SMORPH BACKWARD PASS CPU----------------------------------------

#define INSTANTIATE_SMORPH_BACKWARD(MERGE, TYPE)                                                                                                     \
    template void smorph_backward_cpu<MORPHO_MERGE::MERGE, TYPE>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,          \
                                                                 torch::Tensor, torch::Tensor);

INSTANTIATE_SMORPH_BACKWARD(MIN, float)
INSTANTIATE_SMORPH_BACKWARD(MAX, float)
INSTANTIATE_SMORPH_BACKWARD(AVERAGE, float)
INSTANTIATE_SMORPH_BACKWARD(ADD, float)
INSTANTIATE_SMORPH_BACKWARD(MIN, double)
INSTANTIATE_SMORPH_BACKWARD(MAX, double)
INSTANTIATE_SMORPH_BACKWARD(AVERAGE, double)
INSTANTIATE_SMORPH_BACKWARD(ADD, double)

template void smorph_backward_cpu_identity<float>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                  torch::Tensor);
template void smorph_backward_cpu_identity<double>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor);

// ----------------------------------SMORPH BACKWARD PASS CPU----------------------------------------
