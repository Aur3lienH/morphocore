// template <typename scalar_t>
// __global__ void morpho_erosion_forward_kernel(
//     const scalar_t* input,
//     const scalar_t* weight,
//     scalar_t* output,
//     int64_t* min_indexes,
//     int batch_size,
//     int channels,
//     int height,
//     int width,
//     int padded_height,
//     int padded_width,
//     int kernel_height,
//     int kernel_width
// );

// template <typename scalar_t>
// __global__ void morpho_erosion_backward_kernel(
//     scalar_t* grad_input,
//     scalar_t* grad_weight,
//     const scalar_t* grad_output,
//     const int64_t* min_indexes,
//     int batch_size,
//     int channels,
//     int height,
//     int width,
//     int kernel_height,
//     int kernel_width,
//     int kernel_height_half,
//     int kernel_width_half);

// template <typename scalar_t, int KH, int KW>
// __global__ void morpho_erosion_forward_specialized_kernel(
//     const scalar_t* padded_input,  
//     const scalar_t* weight,
//     scalar_t* output, 
//     int64_t* min_indexes,
//     int batch_size,
//     int channels,
//     int height,
//     int width);


// template <typename scalar_t, int KH, int KW>
// __global__ void morpho_erosion_backward_specialized_kernel(
//     scalar_t* grad_input,
//     scalar_t* grad_weight,
//     const scalar_t* grad_output,
//     const int64_t* min_indexes,
//     int batch_size,
//     int channels,
//     int height,
//     int width);