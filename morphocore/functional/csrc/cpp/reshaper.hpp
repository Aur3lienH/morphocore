#pragma once
#include <torch/torch.h>

/// @brief Utility struct to handle and reshape input tensors to 4D for forward pass and convert it back.
struct TensorInputDim
{
    int64_t batch_size;
    int64_t channels;
    int64_t height;
    int64_t width;


    int64_t original_dims;
    std::vector<int64_t> original_shape;
    TensorInputDim(const torch::Tensor& input);

    torch::Tensor To4DTensor(const torch::Tensor& input);
    torch::Tensor ToOriginalTensor(const torch::Tensor& input);
};

/// @brief Utility struct to handle and reshape weight tensors to 4D for forward pass and convert it back.
struct TensorWeightDim
{
    int64_t in_channels;
    int64_t out_channels;
    int64_t height;
    int64_t width;

    int64_t original_dims;
    std::vector<int64_t> original_shape;
    TensorWeightDim(const torch::Tensor& input);

    torch::Tensor To4DTensor(const torch::Tensor& input);
    torch::Tensor ToOriginalTensor(const torch::Tensor& input);
};