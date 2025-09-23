#pragma once
#include "../cpp/morpho_op_enum.hpp"

template <MORPHO_OPERATION Op, MORPHO_MERGE merge, typename scalar_t, int KH, int KW>
__global__ void morpho_operation_forward_kernel(const scalar_t* padded_input, const scalar_t* weight, scalar_t* output, int* max_indexes,
                                                int batch_size, int in_channels, int out_channels, int height, int width, int kernel_height,
                                                int kernel_width);