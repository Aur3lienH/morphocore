#pragma once
#include "morpho_cuda_common.cuh"

// CUDA kernels of dilation and erosion, launched from morpho_op.cu.
//
// Results match the CPU kernels: taps are visited in the same order (kh then kw) with a strict
// comparison, so ties resolve to the first tap, and the first channel wins ties of MAX and MIN.

/// @brief Copy G structuring elements of taps values into shared memory, [G][taps]; members past
///        `valid` are zeros. Member m starts at first + m * member_stride.
template <typename T, int G>
__device__ __forceinline__ void morpho_load_weights(const T* __restrict__ first, int64_t member_stride, int valid, int taps, T* __restrict__ weights)
{
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < G * taps; i += blockDim.x * blockDim.y)
    {
        const int m = i / taps;
        weights[i] = m < valid ? first[m * member_stride + (i - m * taps)] : static_cast<T>(0);
    }
}

/// @brief Extremum over the structuring element for the MORPHO_ROWS pixels of this thread and the
///        G channel pairs of the block.
/// @tparam shared_window True when the G pairs read the same input window (G output channels of one
///         input channel, forward pass), false when pair m reads window m (G input channels).
/// @param windows Input windows, see morpho_load_window; G of them, window_size() apart, unless shared_window.
/// @param weights Structuring elements of the pairs, [G][taps].
/// @param best Output, extremum of each (pair, pixel).
/// @param arg Output when with_arg, flat tap index kh * kernel_width + kw of the first extremum.
template <MORPHO_OPERATION op, typename T, int G, bool shared_window, bool with_arg>
__device__ __forceinline__ void morpho_window_extremum(const MorphoGeometryCUDA& g, const T* __restrict__ windows, const T* __restrict__ weights,
                                                       T (&best)[G][MORPHO_ROWS], int (&arg)[G][MORPHO_ROWS])
{
    using Op = MorphoOpCUDA<op>;
    const int cols = g.window_cols();
    const int taps = g.taps();
    const int window_step = shared_window ? 0 : g.window_size();
    // Window element read by tap (0, 0) of the first pixel of this thread.
    const T* origin = windows + (threadIdx.y * MORPHO_ROWS + g.kernel_height - 1) * cols + threadIdx.x + g.kernel_width - 1;

#pragma unroll
    for (int m = 0; m < G; m++)
    {
#pragma unroll
        for (int r = 0; r < MORPHO_ROWS; r++)
        {
            best[m][r] = Op::template initial_value<T>();
            arg[m][r] = 0;
        }
    }

    int k = 0;
    for (int kh = 0; kh < g.kernel_height; kh++)
    {
        const T* row = origin - kh * cols;
        for (int kw = 0; kw < g.kernel_width; kw++, k++)
        {
            T x[MORPHO_ROWS];
#pragma unroll
            for (int m = 0; m < G; m++)
            {
                if (m == 0 || !shared_window)
                {
#pragma unroll
                    for (int r = 0; r < MORPHO_ROWS; r++)
                    {
                        x[r] = row[m * window_step + r * cols - kw];
                    }
                }
                const T w = weights[m * taps + k];
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    const T v = Op::combine(x[r], w);
                    if constexpr (with_arg)
                    {
                        if (Op::is_better(v, best[m][r]))
                        {
                            best[m][r] = v;
                            arg[m][r] = k;
                        }
                    }
                    else
                    {
                        best[m][r] = Op::extremum(best[m][r], v);
                    }
                }
            }
        }
    }
}

/// @brief First tap of a channel pair whose value is target, as tracked by the CPU argmax; 0 when none
///        is (every tap then reads the neutral element, and the CPU keeps tap 0 too). A row of taps is
///        read without early exit so that its loads are independent.
template <MORPHO_OPERATION op, typename T>
__device__ int morpho_find_tap(const MorphoGeometryCUDA& g, const T* __restrict__ plane, const T* __restrict__ weight_pair, int h, int w, T target,
                               MorphoBorderCUDA<T> border)
{
    for (int kh = 0; kh < g.kernel_height; kh++)
    {
        int found = -1;
#pragma unroll 4
        for (int kw = g.kernel_width - 1; kw >= 0; kw--)
        {
            const T v = MorphoOpCUDA<op>::combine(morpho_read_tap(g, plane, h, w, kh, kw, border), weight_pair[kh * g.kernel_width + kw]);
            found = v == target ? kw : found;
        }
        if (found >= 0)
        {
            return kh * g.kernel_width + found;
        }
    }
    return 0;
}

/// @brief Forward tile kernel, every merge; a block computes OB output channels. For MAX and MIN with
///        indices, the tap loops only track the winning channel; its winning tap is searched once at
///        the end, which costs one channel instead of an argmax in the inner loop of every channel.
template <MORPHO_OPERATION op, typename T, int OB>
__global__ void __launch_bounds__(MORPHO_THREADS)
    morpho_forward_kernel(const MorphoGeometryCUDA g, const T* __restrict__ input, const T* __restrict__ weight, T* __restrict__ output,
                          int32_t* __restrict__ indices, const MorphoMergeCUDA<T> m, const MorphoBorderCUDA<T> border)
{
    extern __shared__ __align__(sizeof(double)) unsigned char morpho_shared[];
    T* window = reinterpret_cast<T*>(morpho_shared);
    T* weights = window + g.window_size();

    const bool identity = m.merge == MORPHO_MERGE::IDENTITY;
    const int groups = (g.out_channels + OB - 1) / OB;
    const int h0 = blockIdx.y * MORPHO_TILE_H;
    const int w0 = blockIdx.x * MORPHO_TILE_W;
    const int h_first = h0 + threadIdx.y * MORPHO_ROWS;
    const int w = w0 + threadIdx.x;

    for (int z = blockIdx.z; z < g.batch_size * groups; z += gridDim.z)
    {
        const int b = z / groups;
        const int o0 = (z % groups) * OB;
        // IDENTITY pairs output channel o with input channel o only, in groups of one channel.
        const int c_begin = identity ? o0 : 0;
        const int c_end = identity ? o0 + 1 : g.in_channels;

        T merged[OB][MORPHO_ROWS];
        int winner[OB][MORPHO_ROWS];
#pragma unroll
        for (int ob = 0; ob < OB; ob++)
        {
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                merged[ob][r] = m.initial_value();
                winner[ob][r] = c_begin;
            }
        }

        for (int c = c_begin; c < c_end; c++)
        {
            __syncthreads();  // The previous channel is no longer read.
            morpho_load_window(g, input + (static_cast<int64_t>(b) * g.in_channels + c) * g.plane(), h0, w0, border, window);
            morpho_load_weights<T, OB>(weight + (static_cast<int64_t>(o0) * g.weight_in_channels + (identity ? 0 : c)) * g.taps(),
                                       static_cast<int64_t>(g.weight_in_channels) * g.taps(), g.out_channels - o0, g.taps(), weights);
            __syncthreads();

            T best[OB][MORPHO_ROWS];
            int unused[OB][MORPHO_ROWS];
            morpho_window_extremum<op, T, OB, true, false>(g, window, weights, best, unused);
#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    if (m.is_better(best[ob][r], merged[ob][r]))
                    {
                        winner[ob][r] = c;
                    }
                    merged[ob][r] = m.merge_value(merged[ob][r], best[ob][r]);
                }
            }
        }

#pragma unroll
        for (int ob = 0; ob < OB; ob++)
        {
            const int o = o0 + ob;
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                const int h = h_first + r;
                if (o >= g.out_channels || h >= g.height || w >= g.width)
                {
                    continue;
                }
                const int64_t out_index = ((static_cast<int64_t>(b) * g.out_channels + o) * g.height + h) * g.width + w;
                if (output != nullptr)
                {
                    output[out_index] = m.finalize(merged[ob][r], g.in_channels);
                }
                if (indices != nullptr)
                {
                    // MAX and MIN: the merged value is the extremum of the winning channel.
                    const int c = winner[ob][r];
                    const T* plane = input + (static_cast<int64_t>(b) * g.in_channels + c) * g.plane();
                    const T* weight_pair = weight + (static_cast<int64_t>(o) * g.weight_in_channels + c) * g.taps();
                    indices[out_index] = (c << 16) | morpho_find_tap<op>(g, plane, weight_pair, h, w, merged[ob][r], border);
                }
            }
        }
    }
}

/// @brief Backward tile kernel of the ADD, AVERAGE and IDENTITY merges: every channel pair routes the
///        gradient of each pixel to its extremal tap.
///        A block owns one input channel of a tile and visits the output channels in groups of OB,
///        which reuse each input value read, as in the forward kernel.
///        The input gradient goes straight to global memory: its atomics hardly conflict and run
///        asynchronously, and measured as fast as accumulating a shared window first. The weight
///        gradient, where every pixel of the block hits the same few taps, is accumulated in shared
///        memory. Shared float atomics are compare and swap loops, so the bins have `copies`
///        interleaved copies, one per lane modulo copies, and the lanes of a warp, which often pick
///        the same tap, rarely update the same element. With copies == 0 the weight gradient goes to
///        global memory directly, which is faster for doubles, whose shared memory atomics are 64 bit
///        compare and swap loops.
template <MORPHO_OPERATION op, typename T, int OB>
__global__ void __launch_bounds__(MORPHO_THREADS)
    morpho_backward_pairs_kernel(const MorphoGeometryCUDA g, const T* __restrict__ input, const T* __restrict__ weight, const T* __restrict__ grad_output,
                                 T* __restrict__ grad_input, T* __restrict__ grad_weight, const MorphoMergeCUDA<T> m, const MorphoBorderCUDA<T> border,
                                 int copies)
{
    using Op = MorphoOpCUDA<op>;
    extern __shared__ __align__(sizeof(double)) unsigned char morpho_shared[];
    const int window_size = g.window_size();
    const int taps = g.taps();
    T* window = reinterpret_cast<T*>(morpho_shared);
    T* weights = window + window_size;
    T* bins = weights + OB * taps;

    const bool identity = m.merge == MORPHO_MERGE::IDENTITY;
    const T scale = m.merge == MORPHO_MERGE::AVERAGE ? static_cast<T>(1) / static_cast<T>(g.in_channels) : static_cast<T>(1);
    const int copy = threadIdx.x & max(copies - 1, 0);
    const int h0 = blockIdx.y * MORPHO_TILE_H;
    const int w0 = blockIdx.x * MORPHO_TILE_W;
    const int h_first = h0 + threadIdx.y * MORPHO_ROWS;
    const int w = w0 + threadIdx.x;

    for (int z = blockIdx.z; z < g.batch_size * g.in_channels; z += gridDim.z)
    {
        const int b = z / g.in_channels;
        const int c = z - b * g.in_channels;

        __syncthreads();  // The previous channel is no longer read.
        morpho_load_window(g, input + static_cast<int64_t>(z) * g.plane(), h0, w0, border, window);
        T* grad_plane = grad_input + static_cast<int64_t>(z) * g.plane();

        // IDENTITY: input channel c only feeds output channel c.
        const int o_begin = identity ? c : 0;
        const int o_end = identity ? c + 1 : g.out_channels;
        for (int o0 = o_begin; o0 < o_end; o0 += OB)
        {
            const int valid = min(OB, o_end - o0);
            T grad[OB][MORPHO_ROWS];
            bool any_grad = false;
#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    const int h = h_first + r;
                    const bool inside = ob < valid && h < g.height && w < g.width;
                    grad[ob][r] = inside ? grad_output[((static_cast<int64_t>(b) * g.out_channels + o0 + ob) * g.height + h) * g.width + w] * scale
                                         : static_cast<T>(0);
                    any_grad |= grad[ob][r] != static_cast<T>(0);
                }
            }
            // Also the barrier after the previous group. Sparse gradients skip whole groups.
            if (!__syncthreads_or(any_grad))
            {
                continue;
            }
            morpho_load_weights<T, OB>(weight + (static_cast<int64_t>(o0) * g.weight_in_channels + (identity ? 0 : c)) * taps,
                                       static_cast<int64_t>(g.weight_in_channels) * taps, valid, taps, weights);
            morpho_zero_shared(bins, OB * taps * copies);
            __syncthreads();

            T best[OB][MORPHO_ROWS];
            int arg[OB][MORPHO_ROWS];
            morpho_window_extremum<op, T, OB, true, true>(g, window, weights, best, arg);
            T* grad_weight_group = grad_weight + (static_cast<int64_t>(o0) * g.weight_in_channels + (identity ? 0 : c)) * taps;
#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    if (grad[ob][r] != static_cast<T>(0))
                    {
                        const int kh = arg[ob][r] / g.kernel_width;
                        const int y = h_first + r + g.pad_h - kh;
                        const int x = w + g.pad_w - (arg[ob][r] - kh * g.kernel_width);
                        // Taps reading the border send no gradient to the input, as on the CPU.
                        if (y >= 0 && y < g.height && x >= 0 && x < g.width)
                        {
                            atomicAdd(grad_plane + y * g.width + x, grad[ob][r]);
                        }
                        if (copies > 0)
                        {
                            atomicAdd(bins + (ob * taps + arg[ob][r]) * copies + copy, grad[ob][r]);
                        }
                        else
                        {
                            atomicAdd(grad_weight_group + static_cast<int64_t>(ob) * g.weight_in_channels * taps + arg[ob][r],
                                      Op::template weight_sign<T>() * grad[ob][r]);
                        }
                    }
                }
            }
            __syncthreads();

            for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < valid * taps * min(copies, 1); i += blockDim.x * blockDim.y)
            {
                T sum = static_cast<T>(0);
                for (int j = 0; j < copies; j++)
                {
                    sum += bins[i * copies + j];
                }
                if (sum != static_cast<T>(0))
                {
                    const int ob = i / taps;
                    atomicAdd(grad_weight_group + static_cast<int64_t>(ob) * g.weight_in_channels * taps + (i - ob * taps), Op::template weight_sign<T>() * sum);
                }
            }
        }
    }
}

/// @brief Backward of the MAX and MIN merges from the packed winning taps: the gradient of each
///        output pixel goes to one input pixel and one weight. A block owns one output channel and
///        a range of pixels of every batch element; the weight gradient of that output channel
///        (in_channels * taps values) is accumulated in shared memory when it fits, which turns
///        heavily contended global atomics into one addition per block.
template <MORPHO_OPERATION op, typename T>
__global__ void __launch_bounds__(MORPHO_THREADS)
    morpho_backward_indices_kernel(const MorphoGeometryCUDA g, const int32_t* __restrict__ indices, const T* __restrict__ grad_output,
                                   T* __restrict__ grad_input, T* __restrict__ grad_weight, int pixels_per_block, bool shared_bins)
{
    using Op = MorphoOpCUDA<op>;
    extern __shared__ __align__(sizeof(double)) unsigned char morpho_shared[];
    T* bins = reinterpret_cast<T*>(morpho_shared);

    const int o = blockIdx.y;
    const int taps = g.taps();
    const int weight_count = g.in_channels * taps;
    const int plane = static_cast<int>(g.plane());
    const int p_begin = blockIdx.x * pixels_per_block;
    const int p_end = min(plane, p_begin + pixels_per_block);
    T* grad_weight_out = grad_weight + static_cast<int64_t>(o) * weight_count;

    if (shared_bins)
    {
        morpho_zero_shared(bins, weight_count);
        __syncthreads();
    }

    for (int b = 0; b < g.batch_size; b++)
    {
        const int64_t out_plane = (static_cast<int64_t>(b) * g.out_channels + o) * plane;
        for (int p = p_begin + threadIdx.y * blockDim.x + threadIdx.x; p < p_end; p += blockDim.x * blockDim.y)
        {
            const T grad = grad_output[out_plane + p];
            if (grad == static_cast<T>(0))
            {
                continue;
            }
            const int32_t packed = indices[out_plane + p];
            const int c = packed >> 16;
            const int k = packed & 0xFFFF;
            const int kh = k / g.kernel_width;
            const int h = p / g.width;
            const int y = h + g.pad_h - kh;
            const int x = p - h * g.width + g.pad_w - (k - kh * g.kernel_width);
            // Taps reading the border send no gradient to the input, as on the CPU.
            if (y >= 0 && y < g.height && x >= 0 && x < g.width)
            {
                atomicAdd(grad_input + (static_cast<int64_t>(b) * g.in_channels + c) * plane + y * g.width + x, grad);
            }
            if (shared_bins)
            {
                atomicAdd(bins + c * taps + k, grad);
            }
            else
            {
                atomicAdd(grad_weight_out + c * taps + k, Op::template weight_sign<T>() * grad);
            }
        }
    }

    if (shared_bins)
    {
        __syncthreads();
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < weight_count; i += blockDim.x * blockDim.y)
        {
            if (bins[i] != static_cast<T>(0))
            {
                atomicAdd(grad_weight_out + i, Op::template weight_sign<T>() * bins[i]);
            }
        }
    }
}
