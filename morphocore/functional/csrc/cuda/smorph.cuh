template <typename scalar_t>
__global__ void morpho_smorph_forward_kernel(
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

template <typename scalar_t>
__global__ void morpho_smorph_backward_kernel(
    const scalar_t* grad_output, 
    scalar_t* grad_input,
    scalar_t* grad_weight,
    scalar_t* grad_alpha,
    const scalar_t* alpha,
    const scalar_t* input,
    const scalar_t* weights,
    const scalar_t* output,
    const scalar_t* reference_values,
    const scalar_t* max_values,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_height,
    int kernel_width,
    int kernel_height_half,
    int kernel_width_half
);