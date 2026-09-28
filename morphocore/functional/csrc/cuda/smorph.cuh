#pragma once
#include <ATen/core/Tensor.h>

#include "../cpp/morpho_op_enum.hpp"

// CUDA Smorph and Smorph SCM (MORPHO_MERGE::SMOOTH), see smorph.cu. The tensors are contiguous and
// on the current device.

/// @brief Forward pass.
/// @param input [batch, in_channels, height, width].
/// @param weight [out_channels, in_channels, kernel_height, kernel_width] ([channels, 1, ...] for IDENTITY).
/// @param alpha Parameter of every channel pair, [out_channels * in_channels] ([channels] for IDENTITY).
/// @param beta [out_channels] for SMOOTH, undefined otherwise.
/// @param output [batch, out_channels, height, width].
template <typename T>
void smorph_forward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& alpha, const at::Tensor& beta, const at::Tensor& output,
                         MORPHO_MERGE merge);

/// @brief Backward pass, adding the gradients into the zero initialised grad_input, grad_weight,
///        grad_alpha (one value per channel pair, like alpha) and grad_beta (SMOOTH only).
/// @param output Output of the forward pass.
template <typename T>
void smorph_backward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& alpha, const at::Tensor& beta, const at::Tensor& output,
                          const at::Tensor& grad_output, const at::Tensor& grad_input, const at::Tensor& grad_weight, const at::Tensor& grad_alpha,
                          const at::Tensor& grad_beta, MORPHO_MERGE merge);
