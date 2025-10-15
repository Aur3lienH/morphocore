#pragma once
#include "../cpp/morpho_op_enum.hpp"

const int TILE_WIDTH = 8;
const int TILE_HEIGHT = 8;

template <MORPHO_MERGE merge, typename scalar_t>
__global__ void smorph_forward_kernel(
    const scalar_t* input, 
    const scalar_t* weight,
    const scalar_t* alpha,
    scalar_t* output,
    scalar_t* reference_values,
    scalar_t* max_values,
    int batch_size,
    int in_channel,
    int out_channel,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);

template <MORPHO_MERGE merge, typename scalar_t>
__global__ void smorph_backward_kernel(
    const scalar_t* grad_output, 
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    const scalar_t* alpha,
    const scalar_t* input,
    const scalar_t* weights,
    int batch_size,
    int input_channels,
    int output_channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);
