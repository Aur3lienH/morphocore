#pragma once

// TODO: Dispatch for the kernel sizes 3x3, 5x5, 7x7 for better performance
//  Not used yet
/// @brief Perform the forward standard morphological operation on CPU using a specialized template for fixed kernel sizes (Optimizing for the
/// compiler).
/// @tparam op Morphological operation type (e.g., dilation, erosion).
/// @tparam merge Merge channel policy.
/// @tparam T Data type of tensor elements.
/// @tparam save_index If true, saves the index of the maximal/minimal kernel position
/// @tparam KERNEL_SIZE Fixed kernel size (must be odd and between 1 and 15).
/// @param input Input tensor of shape [batch, channels, height, width].
/// @param weight Morphological kernel tensor of shape [out_channels, in_channels, kernel_height, kernel_width] where kernel_height and kernel_width
/// must equal KERNEL_SIZE.
/// @return Output tensor, or tuple (output, indices) if save_index is true.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index, int KERNEL_SIZE>
auto morpho_operation_forward_cpu_template(torch::Tensor input, torch::Tensor weight)
{
    static_assert(KERNEL_SIZE >= 1 && KERNEL_SIZE <= 15, "Kernel size must be between 1 and 15");
    static_assert(KERNEL_SIZE % 2 == 1, "Kernel size must be odd");

    const auto input_shape = input.sizes();
    const int64_t batch_size = input_shape[0];
    const int64_t in_channels = input_shape[1];
    const int64_t out_channels = weight.size(0);
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];

    // Constantes compile-time
    constexpr int64_t kernel_half = KERNEL_SIZE / 2;
    constexpr int64_t kernel_area = KERNEL_SIZE * KERNEL_SIZE;
    constexpr T PADDING_VAL = (op == MORPHO_OPERATION::DILATION) ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    constexpr T INITIAL_VAL = PADDING_VAL;

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    torch::Tensor output = torch::empty({batch_size, out_channels, height, width}, input_contig.options());

    const T* __restrict__ input_ptr = input_contig.template data_ptr<T>();
    const T* __restrict__ weight_ptr = weight_contig.template data_ptr<T>();
    T* __restrict__ output_ptr = output.template data_ptr<T>();

    // Strides optimisés pour tenseurs contigus
    const int64_t input_batch_stride = in_channels * height * width;
    const int64_t input_channel_stride = height * width;
    const int64_t weight_out_stride = in_channels * kernel_area;
    const int64_t weight_in_stride = kernel_area;
    const int64_t output_batch_stride = out_channels * height * width;
    const int64_t output_channel_stride = height * width;

    torch::Tensor max_indices;
    int64_t* __restrict__ indices_ptr = nullptr;

    if constexpr (save_index)
    {
        max_indices = torch::empty({batch_size, out_channels, in_channels, height, width}, torch::kLong);
        indices_ptr = max_indices.template data_ptr<int64_t>();
    }

    // Tiling adaptatif basé sur la taille du kernel
    constexpr int64_t TILE_H = (KERNEL_SIZE <= 3) ? 16 : (KERNEL_SIZE <= 5) ? 12 : 8;
    constexpr int64_t TILE_W = TILE_H;

#pragma omp parallel
    {
#pragma omp for schedule(guided) collapse(3)
        for (int64_t b = 0; b < batch_size; b++)
        {
            for (int64_t o = 0; o < out_channels; o++)
            {
                for (int64_t h_tile = 0; h_tile < height; h_tile += TILE_H)
                {
                    const int64_t h_end = std::min(h_tile + TILE_H, height);

                    const T* input_batch = input_ptr + b * input_batch_stride;
                    const T* weight_out = weight_ptr + o * weight_out_stride;
                    T* output_base = output_ptr + b * output_batch_stride + o * output_channel_stride;

                    int64_t* indices_base = nullptr;
                    if constexpr (save_index)
                    {
                        indices_base = indices_ptr + b * (out_channels * in_channels * height * width) + o * (in_channels * height * width);
                    }

                    for (int64_t w_tile = 0; w_tile < width; w_tile += TILE_W)
                    {
                        const int64_t w_end = std::min(w_tile + TILE_W, width);

                        for (int64_t h = h_tile; h < h_end; h++)
                        {
                            for (int64_t w = w_tile; w < w_end; w++)
                            {
                                T channel_sum = static_cast<T>(0);

                                for (int64_t i = 0; i < in_channels; i++)
                                {
                                    const T* input_channel = input_batch + i * input_channel_stride;
                                    const T* weight_in = weight_out + i * weight_in_stride;

                                    T best_value = INITIAL_VAL;
                                    int64_t best_k = 0;
                                    int64_t k = 0;

#pragma unroll
                                    for (int64_t kh = 0; kh < KERNEL_SIZE; kh++)
                                    {
                                        const int64_t ph = h + kernel_half - kh;
                                        const bool h_valid = (ph >= 0 && ph < height);

#pragma unroll
                                        for (int64_t kw = 0; kw < KERNEL_SIZE; kw++, k++)
                                        {
                                            const int64_t pw = w + kernel_half - kw;
                                            const bool w_valid = (pw >= 0 && pw < width);

                                            const T input_val = (h_valid && w_valid) ? input_channel[ph * width + pw] : PADDING_VAL;

                                            const T val = (op == MORPHO_OPERATION::DILATION) ? input_val + weight_in[k] : input_val - weight_in[k];

                                            const bool is_better = (op == MORPHO_OPERATION::DILATION) ? (val > best_value) : (val < best_value);

                                            if (is_better)
                                            {
                                                best_value = val;
                                                if constexpr (save_index)
                                                {
                                                    best_k = k;
                                                }
                                            }
                                        }
                                    }

                                    channel_sum += best_value;

                                    if constexpr (save_index)
                                    {
                                        indices_base[i * height * width + h * width + w] = best_k;
                                    }
                                }

                                output_base[h * width + w] = channel_sum;
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (save_index)
    {
        return std::make_tuple(output, max_indices);
    }
    else
    {
        return output;
    }
}