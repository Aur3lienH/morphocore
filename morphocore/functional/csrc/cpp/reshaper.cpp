#include "reshaper.hpp"


TensorInputDim::TensorInputDim(const torch::Tensor& input)
{
    original_shape = input.sizes().vec();
    original_dims = input.dim();
    if(original_dims == 4)
    {
        batch_size = input.size(0);
        channels = input.size(1);
        height = input.size(2);
        width = input.size(3);
    }
    else if(original_dims == 3)
    {
        batch_size = 1;
        channels = input.size(0);
        height = input.size(1);
        width = input.size(2);
    }
    else if(original_dims == 2)
    {
        batch_size = 1;
        channels = 1;
        height = input.size(0);
        width = input.size(1);
    }
    else
    {
        throw std::runtime_error("Invalid input tensor: please give a 2D, 3D or 4D input tensor, received: " + std::to_string(original_dims) + " dimensions !");
    }
}



torch::Tensor TensorInputDim::To4DTensor(const torch::Tensor &input)
{
    if(original_dims == 4)
    {
        return input;
    }
    else if(original_dims == 3)
    {
        return input.unsqueeze(0);
    }
    else if(original_dims == 2)
    {
        return input.unsqueeze(0).unsqueeze(0);
    }
    else
    {
        throw std::runtime_error("Invalid input tensor: please give a 2D, 3D or 4D input tensor, received: " + std::to_string(original_dims) + " dimensions !");
    }
}

torch::Tensor TensorInputDim::ToOriginalTensor(const torch::Tensor &input)
{
    if(input.dim() != 4)
    {
        throw std::runtime_error("Invalid tensor: Should give this function a 4D tensor, received: " + std::to_string(original_dims) + " dimensions !");
    }
    if(original_dims == 4)
    {
        return input;
    }
    else if(original_dims == 3)
    {
        return input.squeeze(0);
    }
    else if(original_dims == 2)
    {
        return input.squeeze(0).squeeze(0);
    }
    else
    {
        throw std::runtime_error("Invalid input tensor: please give a 2D, 3D or 4D input tensor, received: " + std::to_string(input.dim()) + " dimensions !");
    }
}


// Constructor implementation
TensorWeightDim::TensorWeightDim(const torch::Tensor& input)
{
    original_shape = input.sizes().vec();
    original_dims = input.dim();
    
    if(original_dims == 4)
    {
        // [batch, out_channels, in_channels, height, width]
        out_channels = input.size(0);
        in_channels = input.size(1);
        height = input.size(2);
        width = input.size(3);
    }
    else if(original_dims == 2)
    {
        // [height, width]
        out_channels = 1;
        in_channels = 1;
        height = input.size(0);
        width = input.size(1);
    }
    else
    {
        throw std::invalid_argument(
            "Invalid weight tensor dimensions: expected 2D, 3D, 4D or 5D tensor, "
            "received " + std::to_string(original_dims) + "D tensor"
        );
    }
}

torch::Tensor TensorWeightDim::To4DTensor(const torch::Tensor& input)
{
    if(original_dims == 4)
    {
        return input;  
    }
    else if(original_dims == 2)
    {
        // [height, width] -> [1, 1, height, width]
        return input.unsqueeze(0).unsqueeze(0);
    }
    else
    {
        throw std::invalid_argument(
            "Invalid weight tensor dimensions: expected 2D or 4D tensor, received " + std::to_string(original_dims) + "D tensor"
        );
    }
}

// Convert 5D tensor back to original dimensions
torch::Tensor TensorWeightDim::ToOriginalTensor(const torch::Tensor& input)
{
    if(input.dim() != 4)
    {
        throw std::invalid_argument(
            "Invalid tensor dimensions in ToOriginalTensor: expected 4D tensor, received " + std::to_string(input.dim()) + "D tensor"
        );
    }
    
    if(original_dims == 4)
    {
        return input;
    }
    else if(original_dims == 2)
    {
        // [1, 1, height, width] -> [height, width]
        return input.squeeze(0).squeeze(0);
    }
    else
    {
        throw std::invalid_argument("Invalid original tensor dimensions: expected 2D or 4D, but original_dims is " + std::to_string(original_dims));
    }
}