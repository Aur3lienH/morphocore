#pragma once
#include "../cpp/morpho_op_enum.hpp"

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
__global__ void morpho_operation_forward_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* output, int64_t* max_indexes,
                                                int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height,
                                                int kernel_width);

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t>
void morpho_operation_backward_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* grad_input, scalar_t* grad_weight,
                                      const scalar_t* grad_output, int batch_size, int in_channels, int out_channels, int height, int width,
                                      int kernel_height, int kernel_width, dim3 blocks, dim3 threads);