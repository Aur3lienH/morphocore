#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/all.h>

#include "morpho_cpu_simd.hpp"
#include "morpho_merge_policy.hpp"
#include "morpho_op_cpu_kernels.hxx"
#include "morpho_op_enum.hpp"

// CPU kernels of the Smorph operation, a smooth approximation of dilation (alpha -> +inf) and
// erosion (alpha -> -inf). For an output pixel and a channel pair (o, i), over the taps k that read
// a pixel inside the image (the border is excluded, not padded):
//
//     v_k = x_k + w_k,   ref = max_k v_k if alpha >= 0 else min_k v_k,
//     e_k = exp(alpha (v_k - ref)),   S = sum_k e_k,   p_k = e_k / S,   out = sum_k p_k v_k.
//
// alpha (v_k - ref) is never positive, so no exponential overflows, and the extremal tap gives
// e = 1, so S >= 1. The channels are then merged like MorphoMergePolicy, or, for MORPHO_MERGE::SMOOTH
// (Smorph SCM), by the same soft extremum over the channel outputs c_i with a parameter beta per
// output channel: q_i = exp(beta (c_i - ref_c)), final = sum_i q_i c_i / sum_i q_i.
//
// The code reuses the organisation of the morphological kernels (morpho_op_cpu_kernels.hxx): bands
// of rows, padded tiles, blocks of MORPHO_CPU_BLOCK pixels in vector registers, one compilation per
// instruction set. Out of image taps are masked per lane instead of read from the padding.

/// @brief Per lane validity of the taps of one block: tap (kh, kw) of block pixel w0 + j reads the input
///        column w0 + j + pad_w - kw, valid when inside [0, width).
template <int BYTES, typename T>
struct SmorphBlockMask
{
    using Simd = MorphoSimd<T, BYTES>;
    using mask = typename Simd::mask;
    static constexpr int64_t L = Simd::lanes;
    static constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;

    mask column[R];  // Input column read by tap kw = 0.
    mask in_row[R];  // Pixel lies inside the output row.
    int64_t width;

    MORPHO_INLINE SmorphBlockMask(const MorphoGeometryCPU& g, int64_t w0) : width(g.width)
    {
        for (int64_t r = 0; r < R; r++)
        {
            const mask pixel = Simd::iota() + static_cast<typename Simd::index_type>(w0 + r * L);
            column[r] = pixel + static_cast<typename Simd::index_type>(g.pad_w);
            in_row[r] = pixel < static_cast<typename Simd::index_type>(g.width);
        }
    }

    MORPHO_INLINE mask valid(int64_t r, int64_t kw) const
    {
        const mask c = column[r] - static_cast<typename Simd::index_type>(kw);
        return (c >= 0) & (c < static_cast<typename Simd::index_type>(width));
    }
};

/// @brief Taps rows kh whose input row h + pad_h - kh is inside the image, as [begin, end).
MORPHO_INLINE void smorph_valid_rows(const MorphoGeometryCPU& g, int64_t h, int64_t& kh_begin, int64_t& kh_end)
{
    kh_begin = std::max<int64_t>(0, h + g.pad_h - (g.height - 1));
    kh_end = std::min<int64_t>(g.kernel_height, h + g.pad_h + 1);
}

/// @brief Soft extremum of one channel pair over one block of pixels.
/// @param origin Tile element read by tap (0, 0) of the first pixel of the block, see morpho_row_origin.
/// @param h Absolute output row, to exclude the taps outside the image.
/// @param ref Output, hard extremum ref of each pixel.
/// @param inv_sum Output, 1 / S.
/// @param out Output, soft extremum.
/// @param exps If not null, receives e_k for every tap, taps * MORPHO_CPU_BLOCK values (0 for excluded taps).
template <int BYTES, typename T>
MORPHO_INLINE void smorph_block_forward(const MorphoGeometryCPU& g, const SmorphBlockMask<BYTES, T>& m, const T* __restrict__ origin,
                                        const T* __restrict__ weight, T alpha, int64_t h, typename MorphoSimd<T, BYTES>::vec* ref,
                                        typename MorphoSimd<T, BYTES>::vec* inv_sum, typename MorphoSimd<T, BYTES>::vec* out, T* __restrict__ exps)
{
    using Simd = MorphoSimd<T, BYTES>;
    using vec = typename Simd::vec;
    constexpr int64_t L = Simd::lanes;
    constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;
    const bool dilation_like = alpha >= 0;

    int64_t kh_begin, kh_end;
    smorph_valid_rows(g, h, kh_begin, kh_end);

    for (int64_t r = 0; r < R; r++)
    {
        ref[r] = Simd::splat(dilation_like ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity());
    }
    for (int64_t kh = kh_begin; kh < kh_end; kh++)
    {
        const T* row = origin - kh * g.padded_width;
        for (int64_t kw = 0; kw < g.kernel_width; kw++)
        {
            const T wk = weight[kh * g.kernel_width + kw];
            for (int64_t r = 0; r < R; r++)
            {
                const vec v = Simd::load(row - kw + r * L) + wk;
                const auto better = dilation_like ? (v > ref[r]) : (v < ref[r]);
                ref[r] = Simd::select(m.valid(r, kw) & better, v, ref[r]);
            }
        }
    }

    vec sum[R];
    vec weighted[R];
    for (int64_t r = 0; r < R; r++)
    {
        sum[r] = vec{};
        weighted[r] = vec{};
    }
    if (exps != nullptr)
    {
        std::fill(exps, exps + g.taps * MORPHO_CPU_BLOCK<T>, static_cast<T>(0));
    }
    for (int64_t kh = kh_begin; kh < kh_end; kh++)
    {
        const T* row = origin - kh * g.padded_width;
        for (int64_t kw = 0; kw < g.kernel_width; kw++)
        {
            const int64_t k = kh * g.kernel_width + kw;
            const T wk = weight[k];
            for (int64_t r = 0; r < R; r++)
            {
                const vec v = Simd::load(row - kw + r * L) + wk;
                const vec e = Simd::select(m.valid(r, kw), Simd::exp((v - ref[r]) * alpha), vec{});
                sum[r] += e;
                weighted[r] += e * v;
                if (exps != nullptr)
                {
                    Simd::store(exps + k * MORPHO_CPU_BLOCK<T> + r * L, e);
                }
            }
        }
    }

    for (int64_t r = 0; r < R; r++)
    {
        // Pixels past the end of the row may have no valid tap; keep their lanes finite.
        const vec safe_sum = Simd::select(m.in_row[r], sum[r], Simd::splat(T(1)));
        inv_sum[r] = T(1) / safe_sum;
        out[r] = weighted[r] * inv_sum[r];
    }
}

/// @brief Backward of one channel pair over one block of pixels, from the exponentials saved by
///        smorph_block_forward. With g the output gradient of the pixel, tap k receives
///        c_k = p_k (1 + alpha (v_k - out)) g in the input gradient (at the pixel it read) and the weight
///        gradient, and alpha receives sum_k p_k (v_k - ref) (v_k - out) g.
/// @param grad Output gradient of the block pixels for this pair, zero past the end of the row.
/// @param grad_origin Input gradient tile element of tap (0, 0) of the first pixel of the block.
/// @param grad_weight_lanes Weight gradient partial sums, taps * lanes values.
/// @param grad_alpha_lanes Alpha gradient partial sums, lanes values.
template <int BYTES, typename T>
MORPHO_INLINE void smorph_block_backward(const MorphoGeometryCPU& g, const SmorphBlockMask<BYTES, T>& m, const T* __restrict__ origin,
                                         const T* __restrict__ weight, T alpha, int64_t h, const typename MorphoSimd<T, BYTES>::vec* ref,
                                         const typename MorphoSimd<T, BYTES>::vec* inv_sum, const typename MorphoSimd<T, BYTES>::vec* out,
                                         const T* __restrict__ exps, const typename MorphoSimd<T, BYTES>::vec* grad, T* __restrict__ grad_origin,
                                         T* __restrict__ grad_weight_lanes, T* __restrict__ grad_alpha_lanes)
{
    using Simd = MorphoSimd<T, BYTES>;
    using vec = typename Simd::vec;
    constexpr int64_t L = Simd::lanes;
    constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;

    int64_t kh_begin, kh_end;
    smorph_valid_rows(g, h, kh_begin, kh_end);

    vec scaled[R];  // g / S
    for (int64_t r = 0; r < R; r++)
    {
        scaled[r] = grad[r] * inv_sum[r];
    }

    vec alpha_sum = vec{};
    // kh is the inner loop so that the partially overlapping updates of taps (kh, kw) and (kh, kw + 1)
    // are far apart, see morpho_row_backward.
    for (int64_t kw = 0; kw < g.kernel_width; kw++)
    {
        for (int64_t kh = kh_begin; kh < kh_end; kh++)
        {
            const int64_t k = kh * g.kernel_width + kw;
            const T wk = weight[k];
            const T* row = origin - kh * g.padded_width - kw;
            T* dst = grad_origin - kh * g.padded_width - kw;
            vec weight_sum = vec{};
            for (int64_t r = 0; r < R; r++)
            {
                const vec v = Simd::load(row + r * L) + wk;
                // Excluded taps saved e = 0, and lanes past the row have g = 0, so c is 0 there.
                const vec pg = Simd::load(exps + k * MORPHO_CPU_BLOCK<T> + r * L) * scaled[r];
                const vec centered = v - out[r];
                const vec c = pg + alpha * pg * centered;
                Simd::store(dst + r * L, Simd::load(dst + r * L) + c);
                weight_sum += c;
                alpha_sum += Simd::select(m.valid(r, kw), pg * (v - ref[r]) * centered, vec{});
            }
            T* lanes = grad_weight_lanes + k * L;
            Simd::store(lanes, Simd::load(lanes) + weight_sum);
        }
    }
    Simd::store(grad_alpha_lanes, Simd::load(grad_alpha_lanes) + alpha_sum);
}

/// @brief Merge one more channel output into the running merge of a block, with the semantics of
///        MorphoMergePolicy::merge (IDENTITY and SMOOTH do not merge here).
template <MORPHO_MERGE merge, typename V>
MORPHO_INLINE V smorph_merge(V merged, V out)
{
    if constexpr (merge == MORPHO_MERGE::ADD || merge == MORPHO_MERGE::AVERAGE)
    {
        return merged + out;
    }
    else if constexpr (merge == MORPHO_MERGE::MAX)
    {
        return merged < out ? out : merged;  // std::max(merged, out)
    }
    else if constexpr (merge == MORPHO_MERGE::MIN)
    {
        return out < merged ? out : merged;  // std::min(merged, out)
    }
    else
    {
        return out;
    }
}

/// @brief Soft extremum over the channels of a block (MORPHO_MERGE::SMOOTH).
/// @param outputs Soft extremum of each channel, channels * MORPHO_CPU_BLOCK values.
/// @param ref Output, hard extremum over the channels.
/// @param inv_sum Output, 1 / sum_i q_i.
/// @param merged Output, sum_i q_i c_i / sum_i q_i.
template <int BYTES, typename T>
MORPHO_INLINE void smorph_soft_merge(int64_t channels, const T* outputs, T beta, typename MorphoSimd<T, BYTES>::vec* ref,
                                     typename MorphoSimd<T, BYTES>::vec* inv_sum, typename MorphoSimd<T, BYTES>::vec* merged)
{
    using Simd = MorphoSimd<T, BYTES>;
    using vec = typename Simd::vec;
    constexpr int64_t L = Simd::lanes;
    constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;
    const bool max_like = beta >= 0;

    for (int64_t r = 0; r < R; r++)
    {
        ref[r] = Simd::load(outputs + r * L);
        for (int64_t i = 1; i < channels; i++)
        {
            const vec c = Simd::load(outputs + i * MORPHO_CPU_BLOCK<T> + r * L);
            ref[r] = Simd::select(max_like ? (c > ref[r]) : (c < ref[r]), c, ref[r]);
        }
        vec sum = vec{};
        vec weighted = vec{};
        for (int64_t i = 0; i < channels; i++)
        {
            const vec c = Simd::load(outputs + i * MORPHO_CPU_BLOCK<T> + r * L);
            const vec q = Simd::exp((c - ref[r]) * beta);
            sum += q;
            weighted += q * c;
        }
        inv_sum[r] = T(1) / sum;
        merged[r] = weighted * inv_sum[r];
    }
}

/// @brief Worker scratch of the Smorph kernels.
template <typename T>
struct SmorphScratch
{
    std::vector<T> row;      // width_rounded values.
    std::vector<T> exps;     // taps * MORPHO_CPU_BLOCK values.
    std::vector<T> outputs;  // in_channels * MORPHO_CPU_BLOCK values, soft extremum of each channel.

    explicit SmorphScratch(const MorphoGeometryCPU& g)
        : row(g.width_rounded), exps(g.taps * MORPHO_CPU_BLOCK<T>), outputs(g.in_channels * MORPHO_CPU_BLOCK<T>)
    {
    }
};

/// @brief Alpha of an (output, input) channel pair: [channels] for IDENTITY, [out_channels, in_channels] otherwise.
template <MORPHO_MERGE merge>
MORPHO_INLINE int64_t smorph_alpha_index(const MorphoGeometryCPU& g, int64_t o, int64_t i)
{
    return merge == MORPHO_MERGE::IDENTITY ? o : o * g.in_channels + i;
}

/// @brief Work units of the Smorph forward pass, compiled once per instruction set. Same units as
///        MorphoForwardBands: a band of rows of one batch element, and of one channel for IDENTITY.
template <MORPHO_MERGE merge, typename T>
struct SmorphForwardBands
{
    static constexpr bool identity = merge == MORPHO_MERGE::IDENTITY;

    struct Args
    {
        MorphoGeometryCPU g;
        const T* input;
        const T* weight;
        const T* alpha;
        const T* beta;  // SMOOTH only, one per output channel.
        T* output;
        int64_t band_height;
        int64_t bands;
    };

    template <int BYTES>
    static MORPHO_INLINE void run(const Args& a, MorphoWorkQueue& queue)
    {
        using Simd = MorphoSimd<T, BYTES>;
        using vec = typename Simd::vec;
        constexpr int64_t L = Simd::lanes;
        constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;
        using MergePolicy = MorphoMergePolicy<T, merge>;

        const MorphoGeometryCPU& g = a.g;
        SmorphScratch<T> scratch(g);
        MorphoTile<T> tile;

        for (int64_t unit; queue.next(unit);)
        {
            const int64_t h_begin = (unit % a.bands) * a.band_height;
            const int64_t h_end = std::min(h_begin + a.band_height, g.height);
            const int64_t b = identity ? unit / (a.bands * g.out_channels) : unit / a.bands;
            const int64_t o_begin = identity ? (unit / a.bands) % g.out_channels : 0;
            const int64_t o_end = identity ? o_begin + 1 : g.out_channels;
            const int64_t c_begin = identity ? o_begin : 0;
            const int64_t c_end = identity ? o_end : g.in_channels;
            // The padding values are never used: out of image taps are masked.
            tile.template load<MORPHO_PADDING_KIND::FILL>(g, a.input + b * g.in_channels * g.height * g.width, c_begin, c_end, h_begin, h_end, T(0));

            for (int64_t o = o_begin; o < o_end; o++)
            {
                for (int64_t h = h_begin; h < h_end; h++)
                {
                    for (int64_t w0 = 0; w0 < g.width_rounded; w0 += MORPHO_CPU_BLOCK<T>)
                    {
                        const SmorphBlockMask<BYTES, T> m(g, w0);
                        vec merged[R];
                        for (int64_t r = 0; r < R; r++)
                        {
                            merged[r] = Simd::splat(MergePolicy::get_init_value());
                        }

                        for (int64_t i = c_begin; i < c_end; i++)
                        {
                            const T* weight_pair = a.weight + (o * g.weight_in_channels + (identity ? 0 : i)) * g.taps;
                            const T alpha = a.alpha[smorph_alpha_index<merge>(g, o, i)];
                            const T* origin = morpho_row_origin(g, tile.channel(i - c_begin), h - h_begin) + w0;
                            vec ref[R], inv_sum[R], out[R];
                            smorph_block_forward<BYTES, T>(g, m, origin, weight_pair, alpha, h, ref, inv_sum, out, nullptr);
                            for (int64_t r = 0; r < R; r++)
                            {
                                if constexpr (merge == MORPHO_MERGE::SMOOTH)
                                {
                                    Simd::store(scratch.outputs.data() + i * MORPHO_CPU_BLOCK<T> + r * L, out[r]);
                                }
                                merged[r] = smorph_merge<merge>(merged[r], out[r]);
                            }
                        }

                        if constexpr (merge == MORPHO_MERGE::SMOOTH)
                        {
                            vec ref[R], inv_sum[R];
                            smorph_soft_merge<BYTES, T>(g.in_channels, scratch.outputs.data(), a.beta[o], ref, inv_sum, merged);
                        }
                        for (int64_t r = 0; r < R; r++)
                        {
                            // MorphoMergePolicy::finalize, on a vector.
                            const vec final = merge == MORPHO_MERGE::AVERAGE ? merged[r] / static_cast<T>(g.in_channels) : merged[r];
                            Simd::store(scratch.row.data() + w0 + r * L, final);
                        }
                    }

                    std::copy(scratch.row.data(), scratch.row.data() + g.width, a.output + ((b * g.out_channels + o) * g.height + h) * g.width);
                }
            }
        }
    }

    MORPHO_TARGET_AVX2 static void run_avx2(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_AVX2_BYTES>(a, queue); }

    static void run_generic(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_GENERIC_BYTES>(a, queue); }
};

/// @brief Work units of the Smorph backward pass, compiled once per instruction set. Same two pass
///        band decomposition as MorphoBackwardBands.
template <MORPHO_MERGE merge, typename T>
struct SmorphBackwardBands
{
    static constexpr bool identity = merge == MORPHO_MERGE::IDENTITY;
    static constexpr bool winner_takes_all = merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN;
    static constexpr bool soft_merge = merge == MORPHO_MERGE::SMOOTH;
    /// The channel outputs are needed before the gradient of any channel pair.
    static constexpr bool needs_channel_outputs = winner_takes_all || soft_merge;
    /// Channels whose soft extremum is within this distance of the MAX / MIN share the gradient.
    static constexpr T winner_tolerance = static_cast<T>(1e-6);

    struct Args
    {
        MorphoGeometryCPU g;
        const T* input;
        const T* weight;
        const T* alpha;
        const T* beta;  // SMOOTH only, one per output channel.
        const T* grad_output;
        T* grad_input;
        T* grad_weights;  // One weight gradient per thread.
        T* grad_alphas;   // One alpha gradient per thread.
        T* grad_betas;    // One beta gradient per thread, SMOOTH only.
        int64_t alpha_numel;
        int64_t band_height;
        int64_t bands_in_pass;
        int64_t parity;
        T scale;  // Gradient factor of AVERAGE, 1 otherwise.
    };

    template <int BYTES>
    static MORPHO_INLINE void run(const Args& a, MorphoWorkQueue& queue)
    {
        using Simd = MorphoSimd<T, BYTES>;
        using vec = typename Simd::vec;
        constexpr int64_t L = Simd::lanes;
        constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;
        constexpr int64_t BLOCK = MORPHO_CPU_BLOCK<T>;

        const MorphoGeometryCPU& g = a.g;
        const int64_t input_plane = g.height * g.width;
        const int thread = at::get_thread_num();
        T* grad_weight_local = a.grad_weights + static_cast<int64_t>(thread) * g.out_channels * g.weight_in_channels * g.taps;
        T* grad_alpha_local = a.grad_alphas + static_cast<int64_t>(thread) * a.alpha_numel;
        T* grad_beta_local = soft_merge ? a.grad_betas + static_cast<int64_t>(thread) * g.out_channels : nullptr;

        SmorphScratch<T> scratch(g);
        MorphoTile<T> tile;
        MorphoTile<T> grad_tile;
        std::vector<T> weight_lanes(g.weight_in_channels * g.taps * L);
        std::vector<T> alpha_lanes(g.in_channels * L);

        for (int64_t unit; queue.next(unit);)
        {
            const int64_t b = unit / a.bands_in_pass;
            const int64_t h_begin = (2 * (unit % a.bands_in_pass) + a.parity) * a.band_height;
            const int64_t h_end = std::min(h_begin + a.band_height, g.height);
            tile.template load<MORPHO_PADDING_KIND::FILL>(g, a.input + b * g.in_channels * input_plane, 0, g.in_channels, h_begin, h_end, T(0));
            grad_tile.resize_like(tile);
            std::fill(grad_tile.data.begin(), grad_tile.data.end(), static_cast<T>(0));

            for (int64_t o = 0; o < g.out_channels; o++)
            {
                const int64_t i_begin = identity ? o : 0;
                const int64_t i_end = identity ? o + 1 : g.in_channels;
                std::fill(weight_lanes.begin(), weight_lanes.end(), static_cast<T>(0));
                std::fill(alpha_lanes.begin(), alpha_lanes.end(), static_cast<T>(0));
                vec beta_lanes = vec{};

                for (int64_t h = h_begin; h < h_end; h++)
                {
                    const T* grad_row = a.grad_output + ((b * g.out_channels + o) * g.height + h) * g.width;
                    T* grad = scratch.row.data();
                    for (int64_t w = 0; w < g.width; w++)
                    {
                        grad[w] = grad_row[w] * a.scale;
                    }
                    std::fill(grad + g.width, grad + g.width_rounded, static_cast<T>(0));

                    for (int64_t w0 = 0; w0 < g.width_rounded; w0 += BLOCK)
                    {
                        const SmorphBlockMask<BYTES, T> m(g, w0);
                        vec grad_block[R];
                        for (int64_t r = 0; r < R; r++)
                        {
                            grad_block[r] = Simd::load(grad + w0 + r * L);
                        }

                        // MAX and MIN: the channels within winner_tolerance of the extremum share the gradient.
                        // SMOOTH: channel i receives dfinal/dc_i = p_i (1 + beta (c_i - final)) of the gradient.
                        vec winners[R];
                        vec target[R];
                        vec soft_ref[R], soft_inv_sum[R], soft_final[R];
                        if constexpr (needs_channel_outputs)
                        {
                            for (int64_t i = 0; i < g.in_channels; i++)
                            {
                                vec ref[R], inv_sum[R], out[R];
                                smorph_block_forward<BYTES, T>(g, m, pair_origin(a, tile, i, h - h_begin) + w0, pair_weight(a, o, i),
                                                               pair_alpha(a, o, i), h, ref, inv_sum, out, nullptr);
                                for (int64_t r = 0; r < R; r++)
                                {
                                    Simd::store(scratch.outputs.data() + i * BLOCK + r * L, out[r]);
                                    if (i == 0)
                                    {
                                        target[r] = out[r];
                                    }
                                    else if constexpr (merge == MORPHO_MERGE::MAX)
                                    {
                                        target[r] = Simd::select(out[r] > target[r], out[r], target[r]);
                                    }
                                    else if constexpr (merge == MORPHO_MERGE::MIN)
                                    {
                                        target[r] = Simd::select(out[r] < target[r], out[r], target[r]);
                                    }
                                }
                            }
                        }
                        if constexpr (winner_takes_all)
                        {
                            for (int64_t r = 0; r < R; r++)
                            {
                                winners[r] = vec{};
                                for (int64_t i = 0; i < g.in_channels; i++)
                                {
                                    winners[r] += Simd::select(is_winner(Simd::load(scratch.outputs.data() + i * BLOCK + r * L), target[r]), Simd::splat(T(1)),
                                                               vec{});
                                }
                                grad_block[r] = grad_block[r] / Simd::select(winners[r] > T(0), winners[r], Simd::splat(T(1)));
                            }
                        }
                        if constexpr (soft_merge)
                        {
                            const T beta = a.beta[o];
                            smorph_soft_merge<BYTES, T>(g.in_channels, scratch.outputs.data(), beta, soft_ref, soft_inv_sum, soft_final);
                            for (int64_t i = 0; i < g.in_channels; i++)
                            {
                                for (int64_t r = 0; r < R; r++)
                                {
                                    const vec c = Simd::load(scratch.outputs.data() + i * BLOCK + r * L);
                                    const vec p = Simd::exp((c - soft_ref[r]) * beta) * soft_inv_sum[r];
                                    beta_lanes += p * (c - soft_ref[r]) * (c - soft_final[r]) * grad_block[r];
                                }
                            }
                        }

                        for (int64_t i = i_begin; i < i_end; i++)
                        {
                            vec pair_grad[R];
                            for (int64_t r = 0; r < R; r++)
                            {
                                pair_grad[r] = grad_block[r];
                                if constexpr (winner_takes_all)
                                {
                                    pair_grad[r] = Simd::select(is_winner(Simd::load(scratch.outputs.data() + i * BLOCK + r * L), target[r]),
                                                                grad_block[r], vec{});
                                }
                                else if constexpr (soft_merge)
                                {
                                    const T beta = a.beta[o];
                                    const vec c = Simd::load(scratch.outputs.data() + i * BLOCK + r * L);
                                    const vec p = Simd::exp((c - soft_ref[r]) * beta) * soft_inv_sum[r];
                                    pair_grad[r] = grad_block[r] * (p + beta * p * (c - soft_final[r]));
                                }
                            }

                            const T* origin = pair_origin(a, tile, i, h - h_begin) + w0;
                            const T* weight_pair = pair_weight(a, o, i);
                            const T alpha = pair_alpha(a, o, i);
                            vec ref[R], inv_sum[R], out[R];
                            smorph_block_forward<BYTES, T>(g, m, origin, weight_pair, alpha, h, ref, inv_sum, out, scratch.exps.data());

                            T* grad_origin = morpho_row_origin(g, grad_tile.channel(i), h - h_begin) + w0;
                            const int64_t wi = identity ? 0 : i;
                            smorph_block_backward<BYTES, T>(g, m, origin, weight_pair, alpha, h, ref, inv_sum, out, scratch.exps.data(), pair_grad,
                                                            grad_origin, weight_lanes.data() + wi * g.taps * L, alpha_lanes.data() + i * L);
                        }
                    }
                }

                for (int64_t i = i_begin; i < i_end; i++)
                {
                    const int64_t wi = identity ? 0 : i;
                    T* grad_weight_pair = grad_weight_local + (o * g.weight_in_channels + wi) * g.taps;
                    for (int64_t k = 0; k < g.taps; k++)
                    {
                        grad_weight_pair[k] += horizontal_sum(weight_lanes.data() + (wi * g.taps + k) * L, L);
                    }
                    grad_alpha_local[smorph_alpha_index<merge>(g, o, i)] += horizontal_sum(alpha_lanes.data() + i * L, L);
                }
                if constexpr (soft_merge)
                {
                    T beta_values[L];
                    Simd::store(beta_values, beta_lanes);
                    grad_beta_local[o] += horizontal_sum(beta_values, L);
                }
            }

            // Keep the gradient tile rows inside the image, without the padding columns.
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
    }

    MORPHO_TARGET_AVX2 static void run_avx2(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_AVX2_BYTES>(a, queue); }

    static void run_generic(const Args& a, MorphoWorkQueue& queue) { run<MORPHO_SIMD_GENERIC_BYTES>(a, queue); }

private:
    static MORPHO_INLINE const T* pair_origin(const Args& a, const MorphoTile<T>& tile, int64_t i, int64_t h_in_band)
    {
        return morpho_row_origin(a.g, tile.channel(i), h_in_band);
    }

    static MORPHO_INLINE const T* pair_weight(const Args& a, int64_t o, int64_t i)
    {
        return a.weight + (o * a.g.weight_in_channels + (identity ? 0 : i)) * a.g.taps;
    }

    static MORPHO_INLINE T pair_alpha(const Args& a, int64_t o, int64_t i) { return a.alpha[smorph_alpha_index<merge>(a.g, o, i)]; }

    template <typename V>
    static MORPHO_INLINE auto is_winner(V out, V target)
    {
        const V distance = out - target;
        return (distance < winner_tolerance) & (distance > -winner_tolerance);
    }

    static MORPHO_INLINE T horizontal_sum(const T* values, int64_t n)
    {
        T sum = static_cast<T>(0);
        for (int64_t j = 0; j < n; j++)
        {
            sum += values[j];
        }
        return sum;
    }
};

/// @brief Alpha of every (output, input) channel pair, see smorph_alpha_index. A single alpha is shared
///        by all the pairs.
template <MORPHO_MERGE merge>
torch::Tensor smorph_pair_alphas(const torch::Tensor& alpha, const MorphoGeometryCPU& g)
{
    const int64_t pairs = merge == MORPHO_MERGE::IDENTITY ? g.out_channels : g.out_channels * g.in_channels;
    TORCH_CHECK(alpha.numel() == pairs || alpha.numel() == 1, "Alpha must have one value, or shape ",
                merge == MORPHO_MERGE::IDENTITY ? "[channels]" : "[out_channels, in_channels]");
    return alpha.numel() == pairs ? alpha.contiguous() : alpha.reshape({1}).expand({pairs}).contiguous();
}

/// @brief Forward Smorph on CPU, for every merge policy.
/// @tparam merge Merge policy across input channels; IDENTITY pairs output channel c with input channel c.
/// @param input Input tensor of shape [batch, in_channels, height, width].
/// @param weight Structuring element of shape [out_channels, in_channels, kernel_height, kernel_width]
///        ([channels, 1, kernel_height, kernel_width] for IDENTITY).
/// @param alpha Shape [out_channels, in_channels] ([channels] for IDENTITY).
/// @param beta Shape [out_channels], SMOOTH merge only.
/// @return Output tensor of shape [batch, out_channels, height, width].
template <MORPHO_MERGE merge, typename T>
torch::Tensor smorph_forward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor beta = {})
{
    using Bands = SmorphForwardBands<merge, T>;
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    const MorphoGeometryCPU g = MorphoGeometryCPU::make<T>(input_contig, weight_contig);

    if constexpr (Bands::identity)
    {
        TORCH_CHECK(g.out_channels == g.in_channels, "Identity mode requires weight.shape[0] == input.shape[1]");
        TORCH_CHECK(g.weight_in_channels == 1, "Identity mode requires weight.shape[1] == 1, got ", g.weight_in_channels);
    }
    const torch::Tensor alpha_contig = smorph_pair_alphas<merge>(alpha, g);

    torch::Tensor beta_contig;
    if constexpr (merge == MORPHO_MERGE::SMOOTH)
    {
        beta_contig = beta.contiguous();
        TORCH_CHECK(beta_contig.numel() == g.out_channels, "Beta must have shape [out_channels]");
    }

    torch::Tensor output = torch::empty({g.batch_size, g.out_channels, g.height, g.width}, input_contig.options());

    typename Bands::Args args{};
    args.g = g;
    args.input = input_contig.template data_ptr<T>();
    args.weight = weight_contig.template data_ptr<T>();
    args.alpha = alpha_contig.template data_ptr<T>();
    args.beta = merge == MORPHO_MERGE::SMOOTH ? beta_contig.template data_ptr<T>() : nullptr;
    args.output = output.template data_ptr<T>();
    const int64_t outer_units = Bands::identity ? g.batch_size * g.out_channels : g.batch_size;
    args.band_height = g.band_height(outer_units, Bands::identity ? 1 : g.in_channels, 1, sizeof(T));
    args.bands = (g.height + args.band_height - 1) / args.band_height;

    const auto bands_fn = morpho_cpu_dispatch(&Bands::run_avx2, &Bands::run_generic);
    const double cost = g.cost_ns(Bands::identity ? 1 : g.in_channels, 1.0);
    MorphoWorkQueue(outer_units * args.bands).run([&](MorphoWorkQueue& queue) { bands_fn(args, queue); }, cost);
    return output;
}

/// @brief Backward Smorph on CPU, for every merge policy. Writes the gradients of the input, the
///        weight, alpha (and beta for SMOOTH) into the given contiguous tensors of their shapes.
template <MORPHO_MERGE merge, typename T>
void smorph_backward_cpu(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor grad_output, torch::Tensor grad_input,
                         torch::Tensor grad_weight, torch::Tensor grad_alpha, torch::Tensor beta = {}, torch::Tensor grad_beta = {})
{
    using Bands = SmorphBackwardBands<merge, T>;
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto grad_output_contig = grad_output.contiguous();
    const MorphoGeometryCPU g = MorphoGeometryCPU::make<T>(input_contig, weight_contig);
    const torch::Tensor alpha_contig = smorph_pair_alphas<merge>(alpha, g);

    TORCH_CHECK(grad_input.is_contiguous() && grad_weight.is_contiguous() && grad_alpha.is_contiguous(), "Gradients must be contiguous");
    grad_input.zero_();
    grad_weight.zero_();
    grad_alpha.zero_();

    const int num_threads = at::get_num_threads();
    const int64_t weight_numel = grad_weight.numel();
    const int64_t alpha_numel = alpha_contig.numel();
    std::vector<T> local_grad_weights(static_cast<size_t>(num_threads) * weight_numel, static_cast<T>(0));
    std::vector<T> local_grad_alphas(static_cast<size_t>(num_threads) * alpha_numel, static_cast<T>(0));
    constexpr bool soft_merge = merge == MORPHO_MERGE::SMOOTH;
    torch::Tensor beta_contig;
    std::vector<T> local_grad_betas;
    if constexpr (soft_merge)
    {
        beta_contig = beta.contiguous();
        TORCH_CHECK(grad_beta.is_contiguous(), "Gradients must be contiguous");
        grad_beta.zero_();
        local_grad_betas.assign(static_cast<size_t>(num_threads) * g.out_channels, static_cast<T>(0));
    }

    typename Bands::Args args{};
    args.g = g;
    args.input = input_contig.template data_ptr<T>();
    args.weight = weight_contig.template data_ptr<T>();
    args.alpha = alpha_contig.template data_ptr<T>();
    args.beta = soft_merge ? beta_contig.template data_ptr<T>() : nullptr;
    args.grad_output = grad_output_contig.template data_ptr<T>();
    args.grad_input = grad_input.template data_ptr<T>();
    args.grad_weights = local_grad_weights.data();
    args.grad_alphas = local_grad_alphas.data();
    args.grad_betas = local_grad_betas.data();
    args.alpha_numel = alpha_numel;
    // Two bands of a pass must be one band apart, see morpho_operation_backward_cpu. A band holds an
    // input and a gradient tile.
    args.band_height = g.band_height(g.batch_size, 2 * g.in_channels, g.kernel_height, sizeof(T));
    args.scale = merge == MORPHO_MERGE::AVERAGE ? static_cast<T>(1.0) / g.in_channels : static_cast<T>(1.0);

    const auto bands_fn = morpho_cpu_dispatch(&Bands::run_avx2, &Bands::run_generic);
    // Forward recomputation plus gradient routing; MAX, MIN and SMOOTH first evaluate every channel once more.
    const double cost = g.cost_ns(Bands::identity ? 1 : g.in_channels, Bands::needs_channel_outputs ? 3.0 : 2.0);
    const int64_t bands = (g.height + args.band_height - 1) / args.band_height;
    for (args.parity = 0; args.parity < 2; args.parity++)
    {
        args.bands_in_pass = (bands - args.parity + 1) / 2;
        MorphoWorkQueue(g.batch_size * args.bands_in_pass).run([&](MorphoWorkQueue& queue) { bands_fn(args, queue); }, cost / 2);
    }

    T* grad_weight_ptr = grad_weight.template data_ptr<T>();
    T* grad_alpha_ptr = grad_alpha.template data_ptr<T>();
    // A shared alpha receives the gradient of every pair.
    const bool shared_alpha = grad_alpha.numel() != alpha_numel;
    for (int t = 0; t < num_threads; t++)
    {
        for (int64_t j = 0; j < weight_numel; j++)
        {
            grad_weight_ptr[j] += local_grad_weights[t * weight_numel + j];
        }
        for (int64_t j = 0; j < alpha_numel; j++)
        {
            grad_alpha_ptr[shared_alpha ? 0 : j] += local_grad_alphas[t * alpha_numel + j];
        }
        if constexpr (soft_merge)
        {
            T* grad_beta_ptr = grad_beta.template data_ptr<T>();
            for (int64_t o = 0; o < g.out_channels; o++)
            {
                grad_beta_ptr[o] += local_grad_betas[t * g.out_channels + o];
            }
        }
    }
}
