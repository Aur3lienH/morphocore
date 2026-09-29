#pragma once
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "../cpp/morpho_op_enum.hpp"

// Building blocks shared by the CUDA kernels of the morphological operations and of Smorph.
//
// Every kernel works on output tiles. A block of MORPHO_BLOCK_X x block_rows threads computes a tile
// of MORPHO_TILE_W x (block_rows * MORPHO_ROWS) output pixels, each thread a column of MORPHO_ROWS
// pixels, for a group of OB output channels. Input channels are visited one at a time: the block copies the
// input window the tile reads (the tile plus its halo, border already resolved) into shared memory,
// and every tap of every pixel of every output channel of the group then reads that copy. The inner
// loops touch only shared memory and registers, and the input is never padded in global memory.
//
// Tap (kh, kw) of output (h, w) reads input (h + pad_h - kh, w + pad_w - kw), as on the CPU.

inline constexpr int MORPHO_BLOCK_X = 32;
/// Rows of threads of the largest blocks; small problems use fewer, see morpho_fit_block_rows.
inline constexpr int MORPHO_BLOCK_Y = 8;
inline constexpr int MORPHO_THREADS = MORPHO_BLOCK_X * MORPHO_BLOCK_Y;
/// Output rows per thread: a thread reuses each input value it reads for several pixels.
inline constexpr int MORPHO_ROWS = 4;
inline constexpr int MORPHO_TILE_W = MORPHO_BLOCK_X;

/// @brief Shapes of one operation, passed by value to the kernels.
struct MorphoGeometryCUDA
{
    int batch_size;
    int in_channels;
    int out_channels;
    int weight_in_channels;  // 1 for IDENTITY merge, in_channels otherwise.
    int height;
    int width;
    int kernel_height;
    int kernel_width;
    int pad_h;       // kernel_height / 2
    int pad_w;       // kernel_width / 2
    int block_rows;  // Rows of threads of the tile kernels' blocks, at most MORPHO_BLOCK_Y.

    __host__ __device__ int taps() const { return kernel_height * kernel_width; }

    __host__ __device__ int64_t plane() const { return static_cast<int64_t>(height) * width; }

    __host__ __device__ int tile_height() const { return block_rows * MORPHO_ROWS; }

    /// Input window of one output tile, see morpho_load_window.
    __host__ __device__ int window_rows() const { return tile_height() + kernel_height - 1; }

    __host__ __device__ int window_cols() const { return MORPHO_TILE_W + kernel_width - 1; }

    __host__ __device__ int window_size() const { return window_rows() * window_cols(); }
};

/// @brief How out of image taps are resolved. Smorph excludes them: its windows read NaN there.
template <typename T>
struct MorphoBorderCUDA
{
    bool replicate;  // Read the clamped edge pixel, otherwise read fill.
    T fill;
};

/// @brief Copy the input window read by the output tile starting at (h0, w0) into shared memory.
/// The window starts at input row h0 + pad_h - (kernel_height - 1) and column w0 + pad_w - (kernel_width - 1),
/// so tap (kh, kw) of tile pixel (i, j) reads window element (i + kernel_height - 1 - kh, j + kernel_width - 1 - kw).
/// Rows are read by consecutive threads, so the global loads are coalesced.
template <typename T>
__device__ __forceinline__ void morpho_load_window(const MorphoGeometryCUDA& g, const T* __restrict__ plane, int h0, int w0,
                                                   MorphoBorderCUDA<T> border, T* __restrict__ window)
{
    const int rows = g.window_rows();
    const int cols = g.window_cols();
    const int y0 = h0 + g.pad_h - (g.kernel_height - 1);
    const int x0 = w0 + g.pad_w - (g.kernel_width - 1);
    for (int r = threadIdx.y; r < rows; r += blockDim.y)
    {
        const int y = y0 + r;
        const bool row_inside = y >= 0 && y < g.height;
        const T* row = plane + static_cast<int64_t>(min(max(y, 0), g.height - 1)) * g.width;
        for (int c = threadIdx.x; c < cols; c += blockDim.x)
        {
            const int x = x0 + c;
            T value;
            if (row_inside && x >= 0 && x < g.width)
            {
                value = row[x];
            }
            else
            {
                value = border.replicate ? row[min(max(x, 0), g.width - 1)] : border.fill;
            }
            window[r * cols + c] = value;
        }
    }
}

/// @brief Value tap (kh, kw) of output (h, w) reads, straight from global memory (same rule as the window).
template <typename T>
__device__ __forceinline__ T morpho_read_tap(const MorphoGeometryCUDA& g, const T* __restrict__ plane, int h, int w, int kh, int kw,
                                             MorphoBorderCUDA<T> border)
{
    const int y = h + g.pad_h - kh;
    const int x = w + g.pad_w - kw;
    if (y >= 0 && y < g.height && x >= 0 && x < g.width)
    {
        return plane[static_cast<int64_t>(y) * g.width + x];
    }
    if (!border.replicate)
    {
        return border.fill;
    }
    return plane[static_cast<int64_t>(min(max(y, 0), g.height - 1)) * g.width + min(max(x, 0), g.width - 1)];
}

/// @brief Add a gradient accumulated in a shared window (same layout as morpho_load_window) to the
///        global gradient of the input plane. Window elements outside the image are the border: they
///        never receive gradient, as on the CPU. Neighbouring tiles overlap by the halo, hence the atomics.
template <typename T>
__device__ __forceinline__ void morpho_flush_window(const MorphoGeometryCUDA& g, const T* __restrict__ window, int h0, int w0,
                                                    T* __restrict__ grad_plane)
{
    const int rows = g.window_rows();
    const int cols = g.window_cols();
    const int y0 = h0 + g.pad_h - (g.kernel_height - 1);
    const int x0 = w0 + g.pad_w - (g.kernel_width - 1);
    for (int r = threadIdx.y; r < rows; r += blockDim.y)
    {
        const int y = y0 + r;
        if (y < 0 || y >= g.height)
        {
            continue;
        }
        for (int c = threadIdx.x; c < cols; c += blockDim.x)
        {
            const int x = x0 + c;
            const T value = window[r * cols + c];
            if (x >= 0 && x < g.width && value != static_cast<T>(0))
            {
                atomicAdd(grad_plane + static_cast<int64_t>(y) * g.width + x, value);
            }
        }
    }
}

/// @brief Fill n shared memory elements with zeros, with every thread of the block.
template <typename T>
__device__ __forceinline__ void morpho_zero_shared(T* data, int n)
{
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < n; i += blockDim.x * blockDim.y)
    {
        data[i] = static_cast<T>(0);
    }
}

/// @brief Sum of a value over the 32 lanes of a warp, in every lane.
template <typename T>
__device__ __forceinline__ T morpho_warp_sum(T value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

/// @brief Dilation and erosion as used by the tile kernels (see MorphoPolicyCPU for the CPU side).
template <MORPHO_OPERATION op>
struct MorphoOpCUDA
{
    static constexpr bool dilation = op == MORPHO_OPERATION::DILATION;

    /// Neutral element of the extremum, also the geodesic border value.
    template <typename T>
    __device__ static T initial_value()
    {
        return dilation ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    }

    template <typename T>
    __device__ static T combine(T input, T weight)
    {
        return dilation ? input + weight : input - weight;
    }

    template <typename T>
    __device__ static bool is_better(T value, T best)
    {
        return dilation ? value > best : value < best;
    }

    /// Same result as `is_better(value, best) ? value : best` while best is not NaN, in one instruction.
    template <typename T>
    __device__ static T extremum(T best, T value)
    {
        return dilation ? fmax(best, value) : fmin(best, value);
    }

    /// Sign of the weight in combine, which is also the sign of its gradient.
    template <typename T>
    __device__ static T weight_sign()
    {
        return dilation ? static_cast<T>(1) : static_cast<T>(-1);
    }
};

/// @brief Channel merge of the tile kernels, chosen at run time: it acts once per channel and pixel,
///        not per tap, so a uniform branch costs nothing and saves a kernel instantiation per mode.
template <typename T>
struct MorphoMergeCUDA
{
    MORPHO_MERGE merge;

    /// Same values as MorphoMergePolicy: MAX and MIN start from the largest finite values, not infinities.
    __device__ T initial_value() const
    {
        switch (merge)
        {
            case MORPHO_MERGE::MAX: return std::numeric_limits<T>::lowest();
            case MORPHO_MERGE::MIN: return std::numeric_limits<T>::max();
            default: return static_cast<T>(0);
        }
    }

    /// @return True when value replaces the running merge, for MAX and MIN (first channel wins ties).
    __device__ bool is_better(T value, T merged) const { return merge == MORPHO_MERGE::MAX ? value > merged : value < merged; }

    __device__ T merge_value(T merged, T value) const
    {
        switch (merge)
        {
            case MORPHO_MERGE::MAX: return value > merged ? value : merged;
            case MORPHO_MERGE::MIN: return value < merged ? value : merged;
            case MORPHO_MERGE::ADD:
            case MORPHO_MERGE::AVERAGE: return merged + value;
            default: return value;  // IDENTITY: a single channel.
        }
    }

    __device__ T finalize(T merged, int in_channels) const { return merge == MORPHO_MERGE::AVERAGE ? merged / static_cast<T>(in_channels) : merged; }
};
