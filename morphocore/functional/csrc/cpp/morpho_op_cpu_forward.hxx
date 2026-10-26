#pragma once
#include <tuple>

#include <ATen/Parallel.h>
#include <torch/all.h>

#include "morpho_cpu_simd.hpp"
#include "morpho_merge_policy.hpp"
#include "morpho_op_cpu_kernels.hxx"
#include "morpho_op_enum.hpp"
#include "morpho_padding_policy.hpp"

/// @brief Work units [begin, end) of the forward pass, compiled once per instruction set.
/// A unit is a band of output rows of one batch element, and of one channel for IDENTITY merge.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
struct MorphoForwardBands
{
    static constexpr bool identity = merge == MORPHO_MERGE::IDENTITY;

    struct Args
    {
        MorphoGeometryCPU g;
        const T* input;
        const T* weight;
        T* output;
        int32_t* indices;
        T fill;
        int64_t band_height;
        int64_t bands;
    };

    template <int BYTES>
    static MORPHO_INLINE void run(const Args& a, MorphoWorkQueue& queue)
    {
        const MorphoGeometryCPU& g = a.g;
        MorphoRowScratch<T> scratch(g);
        MorphoTile<T> tile;

        for (int64_t unit; queue.next(unit);)
        {
            const int64_t h_begin = (unit % a.bands) * a.band_height;
            const int64_t h_end = std::min(h_begin + a.band_height, g.height);
            // IDENTITY units are (batch, channel, band), the others (batch, band).
            const int64_t b = identity ? unit / (a.bands * g.out_channels) : unit / a.bands;
            const int64_t o_begin = identity ? (unit / a.bands) % g.out_channels : 0;
            const int64_t o_end = identity ? o_begin + 1 : g.out_channels;
            const T* input_batch = a.input + b * g.in_channels * g.height * g.width;

            if constexpr (identity)
            {
                tile.template load<kind>(g, input_batch, o_begin, o_end, h_begin, h_end, a.fill);
            }
            else
            {
                tile.template load<kind>(g, input_batch, 0, g.in_channels, h_begin, h_end, a.fill);
            }

            for (int64_t o = o_begin; o < o_end; o++)
            {
                for (int64_t h = h_begin; h < h_end; h++)
                {
                    const T* result;
                    const int32_t* result_index;
                    if constexpr (identity)
                    {
                        morpho_row_extremum<BYTES, op, T, save_index>(g, tile.channel(0), a.weight + o * g.taps, h - h_begin, scratch.best.data(),
                                                                      scratch.arg.data());
                        result = scratch.best.data();
                        result_index = scratch.arg.data();
                    }
                    else
                    {
                        morpho_row_merge<BYTES, op, merge, T, save_index>(g, tile, a.weight + o * g.in_channels * g.taps, h - h_begin, scratch);
                        result = scratch.merged.data();
                        result_index = scratch.winner.data();
                    }

                    const int64_t out_row = (b * g.out_channels + o) * g.height + h;
                    T* out = a.output + out_row * g.width;
                    for (int64_t w = 0; w < g.width; w++)
                    {
                        out[w] = MorphoMergePolicy<T, merge>::finalize(result[w], g.in_channels);
                    }
                    if constexpr (save_index)
                    {
                        std::copy(result_index, result_index + g.width, a.indices + out_row * g.width);
                    }
                }
            }
        }
    }

    MORPHO_TARGET_AVX2 static void run_avx2(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_AVX2_BYTES>(a, queue); }

    static void run_generic(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_GENERIC_BYTES>(a, queue); }
};

/// @brief Forward morphological operation on CPU, for every merge policy.
/// @tparam op Morphological operation type (DILATION or EROSION).
/// @tparam merge Merge policy for combining results across channels. IDENTITY pairs output channel c
///         with input channel c only, and expects a weight of shape [channels, 1, kernel_height, kernel_width].
/// @tparam kind Padding family used for out of bounds taps.
/// @tparam T Data type of tensor elements.
/// @tparam save_index If true, also return for every output pixel the morpho_pack_index of the tap
///         that produced it (MAX, MIN and IDENTITY merges, where a single tap does), so that the
///         backward pass does not have to recompute it.
/// @param input Input tensor of shape [batch, in_channels, height, width].
/// @param weight Structuring element of shape [out_channels, in_channels, kernel_height, kernel_width].
/// @param fill Border fill value, see morpho_fill_value. Unused when kind is REPLICATE.
/// @return (output, indices); indices is an undefined tensor unless save_index.
template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING_KIND kind, typename T, bool save_index>
std::tuple<torch::Tensor, torch::Tensor> morpho_operation_forward_cpu(torch::Tensor input, torch::Tensor weight, T fill)
{
    static_assert(!save_index || merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN || merge == MORPHO_MERGE::IDENTITY,
                  "Saved indices only exist when a single tap produces each output");
    using Bands = MorphoForwardBands<op, merge, kind, T, save_index>;

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    const MorphoGeometryCPU g = MorphoGeometryCPU::make<T>(input_contig, weight_contig);

    if constexpr (Bands::identity)
    {
        TORCH_CHECK(g.out_channels == g.in_channels, "Identity mode requires weight.shape[0] == input.shape[1]");
        TORCH_CHECK(g.weight_in_channels == 1, "Identity mode requires weight.shape[1] == 1");
    }
    if constexpr (save_index)
    {
        TORCH_CHECK(g.taps < (1 << 16) && g.in_channels < (1 << 15), "Structuring element or channel count too large to save indices");
    }

    torch::Tensor output = torch::empty({g.batch_size, g.out_channels, g.height, g.width}, input_contig.options());
    torch::Tensor indices;
    if constexpr (save_index)
    {
        indices = torch::empty({g.batch_size, g.out_channels, g.height, g.width}, input_contig.options().dtype(torch::kInt));
    }

    typename Bands::Args args{};
    args.g = g;
    args.input = input_contig.template data_ptr<T>();
    args.weight = weight_contig.template data_ptr<T>();
    args.output = output.template data_ptr<T>();
    args.indices = save_index ? indices.template data_ptr<int32_t>() : nullptr;
    args.fill = fill;

    const int64_t outer_units = Bands::identity ? g.batch_size * g.out_channels : g.batch_size;
    args.band_height = g.band_height(outer_units, Bands::identity ? 1 : g.in_channels, 1, sizeof(T));
    args.bands = (g.height + args.band_height - 1) / args.band_height;

    const auto bands_fn = morpho_cpu_dispatch(&Bands::run_avx2, &Bands::run_generic);
    // Tracking the argmax roughly triples the cost of a tap.
    const double cost = g.cost_ns(Bands::identity ? 1 : g.in_channels, save_index ? 0.3 : 0.1);
    MorphoWorkQueue(outer_units * args.bands).run([&](MorphoWorkQueue& queue) { bands_fn(args, queue); }, cost);

    return {output, indices};
}
