#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/all.h>

#include "morpho_cpu_simd.hpp"
#include "morpho_merge_policy.hpp"
#include "morpho_op_enum.hpp"
#include "morpho_op_policy.hpp"
#include "morpho_padding_policy.hpp"

// Building blocks shared by the CPU forward and backward passes.
//
// Both passes split the output in bands of rows; a band first pads the input rows it reads into a
// small tile (MorphoTile), then works one output row at a time on that tile. The row is cut
// in blocks of MORPHO_CPU_BLOCK_BYTES; a block keeps its running extremum and argmax in a few
// independent vector registers while every tap is streamed over it, so the innermost loop is
// branch free, contiguous and not bound by the latency of a single dependency chain.
//
// The taps are visited in the same order as a naive loop (kh then kw, strict comparison), so ties
// resolve to the first tap and results are bit identical to it.
//
// Functions templated on BYTES are compiled once per instruction set, see morpho_cpu_simd.hpp.

/// @brief Bytes of output pixels processed together: 4 AVX2 or 8 SSE registers of accumulators.
inline constexpr int64_t MORPHO_CPU_BLOCK_BYTES = 128;

/// @brief Number of output pixels processed together.
template <typename T>
inline constexpr int64_t MORPHO_CPU_BLOCK = MORPHO_CPU_BLOCK_BYTES / sizeof(T);

/// @brief Packed index of the tap that produced an output, shared with the CUDA kernels:
///        the input channel in the high bits, the flat tap index kh * kernel_width + kw in the low 16.
MORPHO_INLINE int32_t morpho_pack_index(int64_t channel, int64_t tap)
{
    return static_cast<int32_t>((channel << 16) | tap);
}

/// @brief Work units shared by the workers of a parallel region, handed out one by one on demand.
/// Static splits leave the fast cores idle on hybrid CPUs (performance and efficiency cores) or
/// when a core is busy with something else; pulling units balances the load dynamically.
class MorphoWorkQueue
{
public:
    explicit MorphoWorkQueue(int64_t units) : next_(0), units_(units) {}

    /// @brief Claim the next unit.
    /// @return false once every unit has been claimed.
    bool next(int64_t& unit)
    {
        unit = next_.fetch_add(1, std::memory_order_relaxed);
        return unit < units_;
    }

    /// @brief Run worker(queue) on the threads of the ATen pool until the queue is drained.
    /// A worker allocates its scratch once, then claims units. at::get_thread_num() is unique per worker.
    /// @param cost_ns Rough single thread duration of the whole job. Each thread must get enough of it
    ///        to pay for waking it up, so small jobs run on fewer threads, or inline on the caller's.
    template <typename Worker>
    void run(const Worker& worker, double cost_ns)
    {
        constexpr double MIN_NS_PER_WORKER = 8000.0;
        const int64_t useful = static_cast<int64_t>(cost_ns / MIN_NS_PER_WORKER);
        const int64_t workers = std::max<int64_t>(1, std::min<int64_t>({static_cast<int64_t>(at::get_num_threads()), units_, useful}));
        at::parallel_for(0, workers, 1, [&](int64_t, int64_t) { worker(*this); });
    }

private:
    std::atomic<int64_t> next_;
    int64_t units_;
};

/// @brief Shapes and derived strides of one CPU morphological operation.
struct MorphoGeometryCPU
{
    int64_t batch_size;
    int64_t in_channels;
    int64_t out_channels;
    int64_t weight_in_channels;  // 1 for IDENTITY merge, in_channels otherwise.
    int64_t height;
    int64_t width;
    int64_t kernel_height;
    int64_t kernel_width;
    int64_t taps;           // kernel_height * kernel_width.
    int64_t pad_h;          // kernel_height / 2, rows added on each side.
    int64_t pad_w;          // kernel_width / 2, columns added on each side.
    int64_t padded_width;   // width + 2 * pad_w.
    int64_t width_rounded;  // width rounded up to a whole number of blocks.
    int64_t block;          // MORPHO_CPU_BLOCK of the element type.

    template <typename T>
    static MorphoGeometryCPU make(const torch::Tensor& input, const torch::Tensor& weight)
    {
        MorphoGeometryCPU g;
        g.batch_size = input.size(0);
        g.in_channels = input.size(1);
        g.height = input.size(2);
        g.width = input.size(3);
        g.out_channels = weight.size(0);
        g.weight_in_channels = weight.size(1);
        g.kernel_height = weight.size(2);
        g.kernel_width = weight.size(3);
        g.taps = g.kernel_height * g.kernel_width;
        g.pad_h = g.kernel_height / 2;
        g.pad_w = g.kernel_width / 2;
        g.padded_width = g.width + 2 * g.pad_w;
        g.block = MORPHO_CPU_BLOCK<T>;
        g.width_rounded = (g.width + g.block - 1) / g.block * g.block;
        return g;
    }

    /// @brief Rough single thread duration of an operation, for MorphoWorkQueue::run.
    /// @param channels_per_output Input channels combined into each output channel (1 for IDENTITY).
    /// @param ns_per_tap Measured cost of one tap of one pixel: ~0.1 ns for the morphological
    ///        kernels, ~1 ns for Smorph, whose taps each need an exponential.
    double cost_ns(int64_t channels_per_output, double ns_per_tap) const
    {
        return static_cast<double>(batch_size * out_channels * height * width) * static_cast<double>(taps * channels_per_output) * ns_per_tap;
    }

    /// @brief Height of the bands of output rows the work is split in.
    /// Aims at a few bands per thread for load balancing, while keeping the padded tile of one band
    /// small enough to stay in cache and out of the allocator's mmap path.
    /// @param units_per_row_band Independent work units for each band (batch, times channels for IDENTITY).
    /// @param tile_channels Channels padded into the tile of one band.
    /// @param min_height Smallest acceptable band height.
    /// @param element_size sizeof of the element type.
    int64_t band_height(int64_t units_per_row_band, int64_t tile_channels, int64_t min_height, int64_t element_size) const
    {
        const int64_t wanted_bands = (8 * at::get_num_threads() + units_per_row_band - 1) / units_per_row_band;
        int64_t rows = (height + wanted_bands - 1) / wanted_bands;

        constexpr int64_t TILE_BYTES = 96 * 1024;
        const int64_t row_bytes = tile_channels * padded_width * element_size;
        rows = std::min(rows, TILE_BYTES / std::max<int64_t>(1, row_bytes) - 2 * pad_h);

        return std::min(height, std::max(rows, min_height));
    }
};

/// @brief Padded copy of the input rows that a band of output rows reads, for several channels.
/// Output row h of the band [h_begin, h_end) is at row h - h_begin of the tile, see morpho_row_extremum.
template <typename T>
struct MorphoTile
{
    std::vector<T> data;
    int64_t plane = 0;  // Elements per channel.

    /// @brief Pad rows [h_begin, h_end) plus their halo, for input channels [c_begin, c_end).
    /// @param input_batch Input planes of one batch element.
    template <MORPHO_PADDING_KIND kind>
    void load(const MorphoGeometryCPU& g, const T* input_batch, int64_t c_begin, int64_t c_end, int64_t h_begin, int64_t h_end, T fill)
    {
        const int64_t rows = h_end - h_begin + 2 * g.pad_h;
        plane = rows * g.padded_width;
        // The slack lets the last vector block of the last row overrun it; those lanes are discarded.
        const size_t size = static_cast<size_t>((c_end - c_begin) * plane + g.width_rounded - g.width + g.block);
        if (data.size() < size)
        {
            data.resize(size);
        }

        for (int64_t c = c_begin; c < c_end; c++)
        {
            morpho_pad_rows<kind, T>(input_batch + c * g.height * g.width, g.height, g.width, g.pad_h, g.pad_w, h_begin, rows, fill,
                                     data.data() + (c - c_begin) * plane);
        }
    }

    /// @brief Take the layout of another tile, for a tile accumulating per input pixel values.
    void resize_like(const MorphoTile& other)
    {
        plane = other.plane;
        if (data.size() < other.data.size())
        {
            data.resize(other.data.size());
        }
    }

    const T* channel(int64_t c) const { return data.data() + c * plane; }
    T* channel(int64_t c) { return data.data() + c * plane; }
};

/// @brief Per row scratch buffers of one worker, sized once per parallel chunk.
template <typename T>
struct MorphoRowScratch
{
    std::vector<T> best;
    std::vector<T> merged;
    std::vector<int32_t> arg;
    std::vector<int32_t> winner;

    explicit MorphoRowScratch(const MorphoGeometryCPU& g) : best(g.width_rounded), merged(g.width_rounded), arg(g.width_rounded), winner(g.width_rounded)
    {
    }
};

/// @brief Extremum over the structuring element of one block of MORPHO_CPU_BLOCK consecutive output pixels.
/// @tparam BYTES Vector width, see morpho_cpu_simd.hpp.
/// @tparam op Morphological operation type (DILATION or EROSION).
/// @tparam T Data type of tensor elements.
/// @tparam with_arg If true, also track the flat tap index (kh * kernel_width + kw) of the extremum.
/// @param origin Tile element read by tap (0, 0) of the first pixel of the block, see morpho_row_origin.
/// @param weight Structuring element of the (output, input) channel pair, kernel_height * kernel_width values.
/// @param acc Output, extremum of each pixel, in vectors of MorphoSimd<T, BYTES>::lanes pixels.
/// @param idx Output, tap index of each extremum when with_arg.
template <int BYTES, MORPHO_OPERATION op, typename T, bool with_arg>
MORPHO_INLINE void morpho_block_extremum(const MorphoGeometryCPU& g, const T* __restrict__ origin, const T* __restrict__ weight,
                                         typename MorphoSimd<T, BYTES>::vec* acc, typename MorphoSimd<T, BYTES>::mask* idx)
{
    using Policy = MorphoPolicyCPU<T, op>;
    using Simd = MorphoSimd<T, BYTES>;
    using vec = typename Simd::vec;
    using mask = typename Simd::mask;
    constexpr int64_t L = Simd::lanes;
    constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;  // Independent accumulators per block.

    for (int64_t r = 0; r < R; r++)
    {
        acc[r] = Simd::splat(Policy::initial_value());
        idx[r] = mask{};
    }

    int32_t k = 0;
    for (int64_t kh = 0; kh < g.kernel_height; kh++)
    {
        const T* row = origin - kh * g.padded_width;
        for (int64_t kw = 0; kw < g.kernel_width; kw++, k++)
        {
            const T wk = weight[k];
            for (int64_t r = 0; r < R; r++)
            {
                const vec v = Policy::combine(Simd::load(row - kw + r * L), wk);
                const vec previous = acc[r];
                acc[r] = Simd::select(Policy::is_better(v, previous), v, previous);
                if constexpr (with_arg)
                {
                    // Equivalent to reusing the is_better mask, since acc changes exactly when v is
                    // strictly better (acc is never NaN). Deriving the mask from the result lets the
                    // compiler emit a single max instruction on the accumulator dependency chain
                    // instead of a compare followed by a blend, which is ~40% faster on AVX2.
                    const mask better = acc[r] != previous;
                    idx[r] = Simd::select(better, Simd::splat_index(k), idx[r]);
                }
            }
        }
    }
}

/// @brief Tile element read by tap (0, 0) of output pixel (h, 0).
/// Tap (kh, kw) of output (h, w) reads input (h + pad_h - kh, w + pad_w - kw), that is tile row
/// h + 2 pad_h - kh and column w + 2 pad_w - kw.
/// @param plane Tile channel, see MorphoTile.
/// @param h Output row, relative to the first row of the band.
template <typename T>
MORPHO_INLINE T* morpho_row_origin(const MorphoGeometryCPU& g, T* plane, int64_t h)
{
    return plane + (h + 2 * g.pad_h) * g.padded_width + 2 * g.pad_w;
}

/// @brief Extremum over the structuring element of one channel, for every pixel of one output row.
/// @param plane Tile channel holding the padded input rows of the band, see MorphoTile.
/// @param weight Structuring element of the (output, input) channel pair, kernel_height * kernel_width values.
/// @param h Output row, relative to the first row of the band.
/// @param best Output, width_rounded values. Lanes past width hold garbage.
/// @param arg Output, width_rounded tap indices when with_arg, untouched otherwise.
/// @see morpho_block_extremum for the template parameters.
template <int BYTES, MORPHO_OPERATION op, typename T, bool with_arg>
MORPHO_INLINE void morpho_row_extremum(const MorphoGeometryCPU& g, const T* __restrict__ plane, const T* __restrict__ weight, int64_t h,
                                       T* __restrict__ best, int32_t* __restrict__ arg)
{
    using Simd = MorphoSimd<T, BYTES>;
    constexpr int64_t L = Simd::lanes;
    constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;
    const T* origin = morpho_row_origin(g, plane, h);

    for (int64_t w0 = 0; w0 < g.width_rounded; w0 += MORPHO_CPU_BLOCK<T>)
    {
        typename Simd::vec acc[R];
        typename Simd::mask idx[R];
        morpho_block_extremum<BYTES, op, T, with_arg>(g, origin + w0, weight, acc, idx);

        for (int64_t r = 0; r < R; r++)
        {
            Simd::store(best + w0 + r * L, acc[r]);
            if constexpr (with_arg)
            {
                Simd::store_index(arg + w0 + r * L, idx[r]);
            }
        }
    }
}

/// @brief Backward of one channel pair for one output row: route the gradient of every pixel to its
///        extremal tap, in the input gradient tile and in the weight gradient.
/// This is the transpose of morpho_row_extremum: for each tap k, the pixels whose extremum is k add
/// their gradient at the position that tap read. Taps that read the padding land in the halo of the
/// gradient tile, which is discarded, so the border never receives input gradient.
/// @param plane Tile channel holding the padded input rows of the band, see MorphoTile.
/// @param weight Structuring element of the (output, input) channel pair.
/// @param h Output row, relative to the first row of the band.
/// @param grad Gradient of the output row, width_rounded values, zero past width.
/// @param grad_plane Input gradient tile channel, same layout as plane.
/// @param grad_weight_lanes Weight gradient, one partial sum per tap and vector lane
///        (taps * MorphoSimd<T, BYTES>::lanes values), reduced by the caller.
template <int BYTES, MORPHO_OPERATION op, typename T>
MORPHO_INLINE void morpho_row_backward(const MorphoGeometryCPU& g, const T* __restrict__ plane, const T* __restrict__ weight, int64_t h,
                                       const T* __restrict__ grad, T* __restrict__ grad_plane, T* __restrict__ grad_weight_lanes)
{
    using Simd = MorphoSimd<T, BYTES>;
    using vec = typename Simd::vec;
    constexpr int64_t L = Simd::lanes;
    constexpr int64_t R = MORPHO_CPU_BLOCK<T> / L;
    const T* origin = morpho_row_origin(g, plane, h);
    T* grad_origin = morpho_row_origin(g, grad_plane, h);

    for (int64_t w0 = 0; w0 < g.width_rounded; w0 += MORPHO_CPU_BLOCK<T>)
    {
        vec acc[R];
        typename Simd::mask idx[R];
        morpho_block_extremum<BYTES, op, T, true>(g, origin + w0, weight, acc, idx);

        vec grad_block[R];
        for (int64_t r = 0; r < R; r++)
        {
            grad_block[r] = Simd::load(grad + w0 + r * L);
        }

        // kh is the inner loop on purpose: taps (kh, kw) and (kh, kw + 1) update the same row shifted
        // by one element, and a load that partially overlaps a store still in flight cannot be
        // forwarded and stalls. Visiting every kh in between keeps such accesses far apart.
        for (int64_t kw = 0; kw < g.kernel_width; kw++)
        {
            for (int64_t kh = 0; kh < g.kernel_height; kh++)
            {
                const int32_t k = static_cast<int32_t>(kh * g.kernel_width + kw);
                T* dst = grad_origin - kh * g.padded_width - kw + w0;
                vec routed_sum = vec{};
                for (int64_t r = 0; r < R; r++)
                {
                    const vec routed = Simd::select(idx[r] == Simd::splat_index(k), grad_block[r], vec{});
                    Simd::store(dst + r * L, Simd::load(dst + r * L) + routed);
                    routed_sum += routed;
                }
                T* weight_lanes = grad_weight_lanes + k * L;
                Simd::store(weight_lanes, Simd::load(weight_lanes) + routed_sum);
            }
        }
    }
}

/// @brief Merge every input channel of one output row into scratch.merged, for the ADD, MAX, MIN and
///        AVERAGE merges (not finalised: AVERAGE still holds the sum).
/// @tparam with_winner If true, scratch.winner receives the morpho_pack_index of the tap that won the
///         merge (MAX and MIN only). Channel 0 wins when no channel is better, which keeps the index
///         valid even for an output of -inf, as in the CUDA kernel.
/// @param tile Padded rows of every input channel, see MorphoTile.
/// @param weight_out Structuring elements of the output channel, [in_channels, kernel_height, kernel_width].
/// @param h Output row, relative to the first row of the band.
template <int BYTES, MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool with_winner>
MORPHO_INLINE void morpho_row_merge(const MorphoGeometryCPU& g, const MorphoTile<T>& tile, const T* __restrict__ weight_out, int64_t h,
                                    MorphoRowScratch<T>& scratch)
{
    using MergePolicy = MorphoMergePolicy<T, merge>;
    T* __restrict__ merged = scratch.merged.data();
    const T* __restrict__ best = scratch.best.data();
    const int32_t* __restrict__ arg = scratch.arg.data();
    int32_t* __restrict__ winner = scratch.winner.data();

    std::fill(scratch.merged.begin(), scratch.merged.end(), MergePolicy::get_init_value());

    for (int64_t i = 0; i < g.in_channels; i++)
    {
        // Tracking the argmax in every channel is cheaper than searching the tap of the winning
        // channel afterwards: that search is a data dependent branch per pixel.
        morpho_row_extremum<BYTES, op, T, with_winner>(g, tile.channel(i), weight_out + i * g.taps, h, scratch.best.data(),
                                                       scratch.arg.data());

        const int32_t channel_bits = morpho_pack_index(i, 0);
#pragma omp simd
        for (int64_t w = 0; w < g.width_rounded; w++)
        {
            if constexpr (with_winner)
            {
                const bool better = (i == 0) || MergePolicy::is_better(best[w], merged[w]);
                winner[w] = better ? (channel_bits | arg[w]) : winner[w];
            }
            merged[w] = MergePolicy::merge(merged[w], best[w]);
        }
    }
}
