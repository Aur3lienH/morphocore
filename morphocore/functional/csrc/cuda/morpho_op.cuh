#pragma once
#include <ATen/core/Tensor.h>

#include "../cpp/morpho_op_enum.hpp"

// CUDA dilation and erosion, see morpho_op.cu. The tensors are contiguous and on the current
// device; the input is not padded: out of image taps read `fill`, or the clamped edge pixel when
// `replicate` is set.

/// @brief Forward pass.
/// @param input [batch, in_channels, height, width].
/// @param weight [out_channels, in_channels, kernel_height, kernel_width] ([channels, 1, ...] for IDENTITY).
/// @param output [batch, out_channels, height, width], or undefined to compute the indices only.
/// @param indices int32 [batch, out_channels, height, width] receiving the packed winning taps
///        (channel << 16 | tap) of MAX and MIN merges, or undefined.
template <MORPHO_OPERATION op, typename T>
void morpho_forward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& output, const at::Tensor& indices,
                         MORPHO_MERGE merge, bool replicate, T fill);

/// @brief Backward pass, adding the gradients into grad_input and grad_weight (zero initialised).
/// @param indices Saved indices of a MAX or MIN merge, or undefined to recompute them.
template <MORPHO_OPERATION op, typename T>
void morpho_backward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& indices, const at::Tensor& grad_output,
                          const at::Tensor& grad_input, const at::Tensor& grad_weight, MORPHO_MERGE merge, bool replicate, T fill);
