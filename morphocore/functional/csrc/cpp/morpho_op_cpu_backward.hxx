#pragma once
#include <vector>

#include <ATen/Parallel.h>
#include <torch/all.h>

#include "morpho_cpu_simd.hpp"
#include "morpho_op_cpu_kernels.hxx"
#include "morpho_op_enum.hpp"
#include "morpho_padding_policy.hpp"

/// @brief Band units [begin, end) of the backward pass, compiled once per instruction set.
/// See morpho_operation_backward_cpu for the band decomposition.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
struct MorphoBackwardBands
{
    static constexpr bool single_winner = merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN;
    static constexpr bool identity = merge == MORPHO_MERGE::IDENTITY;
    static constexpr T weight_sign = op == MORPHO_OPERATION::DILATION ? static_cast<T>(1) : static_cast<T>(-1);

    struct Args
    {
        MorphoGeometryCPU g;
        const T* input;             // Read unless save_index.
        const int32_t* indices;     // Packed winning taps, if save_index.
        const T* weight;
        const T* grad_output;
        T* grad_input;
        T* grad_weights;            // One weight gradient per thread.
        const int64_t* tap_offset;  // Per tap k: pad_h - kh and pad_w - kw.
        int64_t band_height;
        int64_t bands_in_pass;
        int64_t parity;
        T fill;
        T scale;  // Gradient factor of AVERAGE, 1 otherwise.
    };

    /// @brief MAX and MIN: each output pixel sends its whole gradient to the single winning tap, a
    ///        scalar scatter per pixel. Border taps read padding, not a pixel, and receive no input
    ///        gradient (REPLICATE included: the gradient is not routed to the copied edge pixel).
    template <int BYTES>
    static MORPHO_INLINE void run_single_winner(const Args& a, int64_t b, int64_t h_begin, int64_t h_end, const MorphoTile<T>& tile,
                                                MorphoRowScratch<T>& scratch, T* grad_weight_local)
    {
        const MorphoGeometryCPU& g = a.g;
        const int64_t input_plane = g.height * g.width;
        T* grad_input_batch = a.grad_input + b * g.in_channels * input_plane;

        for (int64_t o = 0; o < g.out_channels; o++)
        {
            const T* weight_out = a.weight + o * g.in_channels * g.taps;
            T* grad_weight_out = grad_weight_local + o * g.in_channels * g.taps;

            for (int64_t h = h_begin; h < h_end; h++)
            {
                const int64_t out_row = (b * g.out_channels + o) * g.height + h;
                const T* grad_row = a.grad_output + out_row * g.width;

                const int32_t* winner;
                if constexpr (save_index)
                {
                    winner = a.indices + out_row * g.width;
                }
                else
                {
                    morpho_row_merge<BYTES, op, merge, T, true>(g, tile, weight_out, h - h_begin, scratch);
                    winner = scratch.winner.data();
                }

                for (int64_t w = 0; w < g.width; w++)
                {
                    const T grad = grad_row[w];
                    if (grad == static_cast<T>(0)) continue;

                    const int64_t channel = winner[w] >> 16;
                    const int64_t k = winner[w] & 0xFFFF;
                    const int64_t ih = h + a.tap_offset[2 * k];
                    const int64_t iw = w + a.tap_offset[2 * k + 1];
                    if (static_cast<uint64_t>(ih) < static_cast<uint64_t>(g.height) && static_cast<uint64_t>(iw) < static_cast<uint64_t>(g.width))
                    {
                        grad_input_batch[channel * input_plane + ih * g.width + iw] += grad;
                    }
                    grad_weight_out[channel * g.taps + k] += weight_sign * grad;
                }
            }
        }
    }

    /// @brief ADD, AVERAGE and IDENTITY: every (output, input) channel pair routes the gradient of each
    ///        pixel to its extremal tap, with the vectorised transpose morpho_row_backward. The input
    ///        gradient is accumulated in a padded tile whose halo (the border taps) is dropped.
    template <int BYTES>
    static MORPHO_INLINE void run_per_pair(const Args& a, int64_t b, int64_t h_begin, int64_t h_end, const MorphoTile<T>& tile,
                                           MorphoTile<T>& grad_tile, MorphoRowScratch<T>& scratch, std::vector<T>& weight_lanes,
                                           T* grad_weight_local)
    {
        const MorphoGeometryCPU& g = a.g;
        constexpr int64_t LANES = MorphoSimd<T, BYTES>::lanes;
        std::fill(grad_tile.data.begin(), grad_tile.data.end(), static_cast<T>(0));

        for (int64_t o = 0; o < g.out_channels; o++)
        {
            // IDENTITY reads its own channel, ADD and AVERAGE every channel.
            const int64_t i_begin = identity ? o : 0;
            const int64_t i_end = identity ? o + 1 : g.in_channels;
            for (int64_t i = i_begin; i < i_end; i++)
            {
                const int64_t wi = identity ? 0 : i;
                const T* weight_pair = a.weight + (o * g.weight_in_channels + wi) * g.taps;
                std::fill(weight_lanes.begin(), weight_lanes.end(), static_cast<T>(0));

                for (int64_t h = h_begin; h < h_end; h++)
                {
                    const T* grad_row = a.grad_output + ((b * g.out_channels + o) * g.height + h) * g.width;
                    T* grad = scratch.best.data();
                    for (int64_t w = 0; w < g.width; w++)
                    {
                        grad[w] = grad_row[w] * a.scale;
                    }
                    std::fill(grad + g.width, grad + g.width_rounded, static_cast<T>(0));

                    morpho_row_backward<BYTES, op, T>(g, tile.channel(i), weight_pair, h - h_begin, grad, grad_tile.channel(i), weight_lanes.data());
                }

                T* grad_weight_pair = grad_weight_local + (o * g.weight_in_channels + wi) * g.taps;
                for (int64_t k = 0; k < g.taps; k++)
                {
                    T sum = static_cast<T>(0);
                    for (int64_t j = 0; j < LANES; j++)
                    {
                        sum += weight_lanes[k * LANES + j];
                    }
                    grad_weight_pair[k] += weight_sign * sum;
                }
            }
        }

        // Tile row t holds input row h_begin - pad_h + t; keep the rows inside the image and drop the
        // side columns, which are the padding.
        const int64_t input_plane = g.height * g.width;
        const int64_t y_begin = std::max<int64_t>(0, h_begin - g.pad_h);
        const int64_t y_end = std::min(g.height, h_end + g.pad_h);
        for (int64_t i = 0; i < g.in_channels; i++)
        {
            for (int64_t y = y_begin; y < y_end; y++)
            {
                const T* src = grad_tile.channel(i) + (y - h_begin + g.pad_h) * g.padded_width + g.pad_w;
                T* dst = a.grad_input + (b * g.in_channels + i) * input_plane + y * g.width;
                for (int64_t x = 0; x < g.width; x++)
                {
                    dst[x] += src[x];
                }
            }
        }
    }

    template <int BYTES>
    static MORPHO_INLINE void run(const Args& a, MorphoWorkQueue& queue)
    {
        const MorphoGeometryCPU& g = a.g;
        T* grad_weight_local = a.grad_weights + static_cast<int64_t>(at::get_thread_num()) * g.out_channels * g.weight_in_channels * g.taps;
        MorphoRowScratch<T> scratch(g);
        MorphoTile<T> tile;
        MorphoTile<T> grad_tile;
        std::vector<T> weight_lanes(g.taps * MorphoSimd<T, BYTES>::lanes);

        for (int64_t unit; queue.next(unit);)
        {
            const int64_t b = unit / a.bands_in_pass;
            const int64_t h_begin = (2 * (unit % a.bands_in_pass) + a.parity) * a.band_height;
            const int64_t h_end = std::min(h_begin + a.band_height, g.height);

            // The saved indices already hold every winning tap, the input values are not needed.
            if constexpr (!save_index)
            {
                tile.template load<kind>(g, a.input + b * g.in_channels * g.height * g.width, 0, g.in_channels, h_begin, h_end, a.fill);
            }

            if constexpr (single_winner)
            {
                run_single_winner<BYTES>(a, b, h_begin, h_end, tile, scratch, grad_weight_local);
            }
            else
            {
                grad_tile.resize_like(tile);
                run_per_pair<BYTES>(a, b, h_begin, h_end, tile, grad_tile, scratch, weight_lanes, grad_weight_local);
            }
        }
    }

    MORPHO_TARGET_AVX2 static void run_avx2(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_AVX2_BYTES>(a, queue); }

    static void run_generic(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_GENERIC_BYTES>(a, queue); }
};

/// @brief Backward morphological operation on CPU, for every merge policy.
/// The extremum is a selection, so each output pixel sends its gradient to one tap per merged
/// channel (ADD, AVERAGE), to the tap of the winning channel (MAX, MIN), or to one tap of its own
/// channel (IDENTITY).
///
/// Parallelisation: the output rows of a band [h_begin, h_end) only write the input rows
/// [h_begin - pad_h, h_end + pad_h). Output rows are cut in bands of at least kernel_height rows and
/// processed in two passes, even bands then odd bands: two bands of the same pass are one band
/// apart, so their windows never overlap and the input gradient is written without atomics nor per
/// thread copies. Only the small weight gradient is accumulated per thread.
///
/// @tparam op Morphological operation type (DILATION or EROSION).
/// @tparam merge Merge policy for combining results across channels.
/// @tparam kind Padding family used for out of bounds taps.
/// @tparam T Data type of tensor elements.
/// @tparam save_index If true, read the winning taps from max_indices instead of recomputing them.
/// @param max_indices Packed indices from morpho_operation_forward_cpu, used when save_index.
/// @param fill Border fill value, see morpho_fill_value. Unused when kind is REPLICATE.
/// @return The gradient tensors for the input and the weight.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
torch::autograd::tensor_list morpho_operation_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output,
                                                           torch::Tensor max_indices, T fill)
{
    using Bands = MorphoBackwardBands<op, merge, kind, T, save_index>;
    static_assert(!save_index || Bands::single_winner, "Saved indices only work with MAX or MIN merge");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto grad_output_contig = grad_output.contiguous();
    const MorphoGeometryCPU g = MorphoGeometryCPU::make<T>(input_contig, weight_contig);

    torch::Tensor grad_input = torch::zeros_like(input_contig);
    torch::Tensor grad_weight = torch::zeros_like(weight_contig);

    torch::Tensor indices;
    if constexpr (save_index)
    {
        indices = max_indices.contiguous();
    }

    std::vector<int64_t> tap_offset(2 * g.taps);
    for (int64_t k = 0; k < g.taps; k++)
    {
        tap_offset[2 * k] = g.pad_h - k / g.kernel_width;
        tap_offset[2 * k + 1] = g.pad_w - k % g.kernel_width;
    }

    const int64_t weight_numel = grad_weight.numel();
    const int num_threads = at::get_num_threads();
    std::vector<T> local_grad_weights(static_cast<size_t>(num_threads) * weight_numel, static_cast<T>(0));

    typename Bands::Args args{};
    args.g = g;
    args.input = input_contig.template data_ptr<T>();
    args.indices = save_index ? indices.template data_ptr<int32_t>() : nullptr;
    args.weight = weight_contig.template data_ptr<T>();
    args.grad_output = grad_output_contig.template data_ptr<T>();
    args.grad_input = grad_input.template data_ptr<T>();
    args.grad_weights = local_grad_weights.data();
    args.tap_offset = tap_offset.data();
    // Two bands of a pass must be one band apart: at least kernel_height rows per band. The per pair
    // path holds an input and a gradient tile.
    const int64_t tile_channels = save_index ? 0 : (Bands::single_winner ? 1 : 2) * g.in_channels;
    args.band_height = g.band_height(g.batch_size, tile_channels, g.kernel_height, sizeof(T));
    args.fill = fill;
    args.scale = merge == MORPHO_MERGE::AVERAGE ? static_cast<T>(1.0) / g.in_channels : static_cast<T>(1.0);
    const auto bands_fn = morpho_cpu_dispatch(&Bands::run_avx2, &Bands::run_generic);

    // With saved indices each pixel is a single scatter; otherwise every tap is revisited with its argmax.
    const double cost = save_index ? static_cast<double>(g.batch_size * g.out_channels * g.height * g.width) * 2.0
                                   : g.cost_ns(Bands::identity ? 1 : g.in_channels, 0.5);
    const int64_t bands = (g.height + args.band_height - 1) / args.band_height;
    for (args.parity = 0; args.parity < 2; args.parity++)
    {
        args.bands_in_pass = (bands - args.parity + 1) / 2;
        MorphoWorkQueue(g.batch_size * args.bands_in_pass).run([&](MorphoWorkQueue& queue) { bands_fn(args, queue); }, cost / 2);
    }

    T* grad_weight_ptr = grad_weight.template data_ptr<T>();
    for (int t = 0; t < num_threads; t++)
    {
        const T* local = local_grad_weights.data() + static_cast<size_t>(t) * weight_numel;
        for (int64_t i = 0; i < weight_numel; i++)
        {
            grad_weight_ptr[i] += local[i];
        }
    }

    return {grad_input, grad_weight};
}
