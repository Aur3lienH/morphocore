#pragma once
#include "morpho_cuda_common.cuh"

// CUDA kernels of Smorph, launched from smorph.cu, on the tiles of morpho_cuda_common.cuh. Same
// maths as the CPU kernels (smorph_cpu_kernels.hxx): for an output pixel and a channel pair (o, i)
// with parameter a, over the taps k that read a pixel inside the image (the border is excluded),
//
//     v_k = x_k + w_k,   ref = max_k v_k if a >= 0 else min_k v_k,
//     e_k = exp(a (v_k - ref)),   S = sum_k e_k,   out = sum_k e_k v_k / S.
//
// The kernels work on u_k = s v_k with s = +1 when a >= 0 and -1 otherwise: ref becomes s max_k u_k
// and e_k = exp(|a| (u_k - max_k u_k)) whatever the sign, so the tap loops have no branch on it.
// The backward formulas carry over, since a (v_k - out) = |a| (u_k - s out) and
// (v_k - ref) (v_k - out) = (u_k - s ref) (u_k - s out).
//
// Out of image taps read NaN from the window: the maximum ignores them, and the sums mask them.

/// @brief Parameters of the G channel pairs of a block, uniform across its threads.
template <typename T, int G>
struct SmorphPairsCUDA
{
    T sign[G];        // s: +1 for a dilation like pair (a >= 0), -1 otherwise.
    T abs_alpha[G];   // |a|
    T exp2_scale[G];  // |a| / ln 2, so that e_k = exp2((u_k - ref_u) exp2_scale).

    /// @param alpha Parameter of every channel pair; `pair` gives the index of member m.
    template <typename PairIndex>
    __device__ void load(const T* __restrict__ alpha, int valid, PairIndex pair)
    {
#pragma unroll
        for (int m = 0; m < G; m++)
        {
            const T a = m < valid ? alpha[pair(m)] : static_cast<T>(0);
            sign[m] = a >= static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(-1);
            abs_alpha[m] = a >= static_cast<T>(0) ? a : -a;
            exp2_scale[m] = abs_alpha[m] * static_cast<T>(1.4426950408889634);  // log2(e)
        }
    }
};

template <typename T>
__device__ __forceinline__ T smorph_exp2(T x)
{
    return exp2(x);
}

template <>
__device__ __forceinline__ float smorph_exp2(float x)
{
    return exp2f(x);
}

/// @brief Copy the structuring elements of the G pairs into shared memory in the u domain, s * w,
///        [G][taps]; members past `valid` are zeros. Member m starts at first + m * member_stride.
template <typename T, int G>
__device__ __forceinline__ void smorph_load_weights(const T* __restrict__ first, int64_t member_stride, int valid, int taps,
                                                    const SmorphPairsCUDA<T, G>& pairs, T* __restrict__ weights)
{
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < G * taps; i += blockDim.x * blockDim.y)
    {
        const int m = i / taps;
        T sign = pairs.sign[0];
#pragma unroll
        for (int j = 1; j < G; j++)
        {
            sign = m == j ? pairs.sign[j] : sign;
        }
        weights[i] = m < valid ? sign * first[m * member_stride + (i - m * taps)] : static_cast<T>(0);
    }
}

/// @brief Soft extremum of the G pairs of the block (G output channels of one input channel, which
///        share the window), for the MORPHO_ROWS pixels of this thread, in the u domain.
/// @param window Input window of the input channel, NaN outside the image, see morpho_load_window.
/// @param weights Structuring elements of the pairs in the u domain, see smorph_load_weights.
/// @param ref Output, max_k u_k.
/// @param inv_sum Output, 1 / S.
/// @param out Output, sum_k e_k u_k / S.
template <typename T, int G>
__device__ __forceinline__ void smorph_window_forward(const MorphoGeometryCUDA& g, const T* __restrict__ window, const T* __restrict__ weights,
                                                      const SmorphPairsCUDA<T, G>& pairs, T (&ref)[G][MORPHO_ROWS], T (&inv_sum)[G][MORPHO_ROWS],
                                                      T (&out)[G][MORPHO_ROWS])
{
    const int cols = g.window_cols();
    const int taps = g.taps();
    const T* origin = window + (threadIdx.y * MORPHO_ROWS + g.kernel_height - 1) * cols + threadIdx.x + g.kernel_width - 1;

#pragma unroll
    for (int m = 0; m < G; m++)
    {
#pragma unroll
        for (int r = 0; r < MORPHO_ROWS; r++)
        {
            ref[m][r] = -std::numeric_limits<T>::infinity();
        }
    }
    int k = 0;
    for (int kh = 0; kh < g.kernel_height; kh++)
    {
        for (int kw = 0; kw < g.kernel_width; kw++, k++)
        {
            T x[MORPHO_ROWS];
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                x[r] = origin[(r - kh) * cols - kw];
            }
#pragma unroll
            for (int m = 0; m < G; m++)
            {
                const T w = weights[m * taps + k];
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    // s x + s w is s (x + w) exactly. fmax ignores the NaN of out of image taps.
                    ref[m][r] = fmax(ref[m][r], fma(pairs.sign[m], x[r], w));
                }
            }
        }
    }

    T sum[G][MORPHO_ROWS];
#pragma unroll
    for (int m = 0; m < G; m++)
    {
#pragma unroll
        for (int r = 0; r < MORPHO_ROWS; r++)
        {
            sum[m][r] = static_cast<T>(0);
            out[m][r] = static_cast<T>(0);
        }
    }
    k = 0;
    for (int kh = 0; kh < g.kernel_height; kh++)
    {
        for (int kw = 0; kw < g.kernel_width; kw++, k++)
        {
            T x[MORPHO_ROWS];
            bool valid[MORPHO_ROWS];
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                x[r] = origin[(r - kh) * cols - kw];
                valid[r] = x[r] == x[r];
                x[r] = valid[r] ? x[r] : static_cast<T>(0);
            }
#pragma unroll
            for (int m = 0; m < G; m++)
            {
                const T w = weights[m * taps + k];
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    const T u = fma(pairs.sign[m], x[r], w);
                    const T e = valid[r] ? smorph_exp2((u - ref[m][r]) * pairs.exp2_scale[m]) : static_cast<T>(0);
                    sum[m][r] += e;
                    out[m][r] += e * u;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < G; m++)
    {
#pragma unroll
        for (int r = 0; r < MORPHO_ROWS; r++)
        {
            // The pixels of a tile past the image have no valid tap; keep them finite (ref is -inf).
            const bool any_tap = sum[m][r] > static_cast<T>(0);
            inv_sum[m][r] = any_tap ? static_cast<T>(1) / sum[m][r] : static_cast<T>(0);
            ref[m][r] = any_tap ? ref[m][r] : static_cast<T>(0);
            out[m][r] *= inv_sum[m][r];
        }
    }
}

/// @brief Backward of the G pairs of the block for the pixels of this thread, from their forward state.
///        Tap k receives c_k = p_k (1 + |a| (u_k - out)) g in the input gradient (at the pixel it read)
///        and in the weight gradient, and a receives sum_k p_k (u_k - ref) (u_k - out) g.
/// @param scaled_grad g / S of each (pair, pixel): the gradient routed to the pair, times inv_sum.
/// @param grad_window Input gradient window of the input channel, same layout as the input window.
/// @param warp_bins Weight gradient of the pairs, [G][taps], private to the warp of this thread.
/// @param grad_alpha Output, this thread's part of the gradient of each pair's parameter.
template <typename T, int G>
__device__ __forceinline__ void smorph_window_backward(const MorphoGeometryCUDA& g, const T* __restrict__ window, const T* __restrict__ weights,
                                                       const SmorphPairsCUDA<T, G>& pairs, const T (&ref)[G][MORPHO_ROWS],
                                                       const T (&out)[G][MORPHO_ROWS], const T (&scaled_grad)[G][MORPHO_ROWS],
                                                       T* __restrict__ grad_window, T* __restrict__ warp_bins, T (&grad_alpha)[G])
{
    const int cols = g.window_cols();
    const int taps = g.taps();
    const int offset = (threadIdx.y * MORPHO_ROWS + g.kernel_height - 1) * cols + threadIdx.x + g.kernel_width - 1;
    const T* origin = window + offset;
    T* grad_origin = grad_window + offset;

    int k = 0;
    for (int kh = 0; kh < g.kernel_height; kh++)
    {
        for (int kw = 0; kw < g.kernel_width; kw++, k++)
        {
            T x[MORPHO_ROWS];
            bool valid[MORPHO_ROWS];
            T input_grad[MORPHO_ROWS];
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                x[r] = origin[(r - kh) * cols - kw];
                valid[r] = x[r] == x[r];
                x[r] = valid[r] ? x[r] : static_cast<T>(0);
                input_grad[r] = static_cast<T>(0);
            }
#pragma unroll
            for (int m = 0; m < G; m++)
            {
                const T w = weights[m * taps + k];
                T weight_grad = static_cast<T>(0);
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    const T u = fma(pairs.sign[m], x[r], w);
                    const T d = u - ref[m][r];
                    const T pg = valid[r] ? smorph_exp2(d * pairs.exp2_scale[m]) * scaled_grad[m][r] : static_cast<T>(0);
                    const T centered = u - out[m][r];
                    const T c = pg + pairs.abs_alpha[m] * pg * centered;
                    input_grad[r] += c;
                    weight_grad += c;
                    grad_alpha[m] += pg * d * centered;
                }
                weight_grad = morpho_warp_sum(weight_grad);
                if (threadIdx.x == 0)
                {
                    warp_bins[m * taps + k] += weight_grad;
                }
            }
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                if (input_grad[r] != static_cast<T>(0))
                {
                    atomicAdd(grad_origin + (r - kh) * cols - kw, input_grad[r]);
                }
            }
        }
    }
}

/// @brief Online soft extremum over the channel outputs c_i (MORPHO_MERGE::SMOOTH): with sb the sign of
///        beta, keeps m = max sb c_i, sum = sum_i exp(|beta| (sb c_i - m)) and the weighted sum of the c_i.
template <typename T>
struct SmorphSoftMergeCUDA
{
    T max_u;
    T sum;
    T weighted;

    __device__ void reset()
    {
        max_u = -std::numeric_limits<T>::infinity();
        sum = static_cast<T>(0);
        weighted = static_cast<T>(0);
    }

    __device__ void add(T c, T beta_sign, T beta_exp2_scale)
    {
        const T cu = beta_sign * c;
        if (cu > max_u)
        {
            // exp2(-inf) = 0 on the first channel.
            const T rescale = smorph_exp2((max_u - cu) * beta_exp2_scale);
            sum = sum * rescale + static_cast<T>(1);
            weighted = weighted * rescale + c;
            max_u = cu;
        }
        else
        {
            const T q = smorph_exp2((cu - max_u) * beta_exp2_scale);
            sum += q;
            weighted += q * c;
        }
    }
};

/// @brief Forward tile kernel of every merge; a block computes OB output channels. It also computes
///        what the backward pass needs to route the gradient of the MAX, MIN and SMOOTH merges, which
///        depends on every channel: with `stats` set and `output` null, it writes the number of
///        channels within `winner_tolerance` of `target` (the forward output) for MAX and MIN, and
///        m and 1 / sum of the soft merge (stats + plane) for SMOOTH.
/// @param alpha Parameter of every channel pair, [out_channels, in_channels] ([channels] for IDENTITY).
/// @param beta Parameter of the SMOOTH merge, [out_channels].
template <typename T, int OB>
__global__ void __launch_bounds__(MORPHO_THREADS)
    smorph_forward_kernel(const MorphoGeometryCUDA g, const T* __restrict__ input, const T* __restrict__ weight, const T* __restrict__ alpha,
                          const T* __restrict__ beta, T* __restrict__ output, const T* __restrict__ target, T* __restrict__ stats,
                          const MORPHO_MERGE merge, const T winner_tolerance)
{
    extern __shared__ __align__(sizeof(double)) unsigned char morpho_shared[];
    T* window = reinterpret_cast<T*>(morpho_shared);
    T* weights = window + g.window_size();

    const bool identity = merge == MORPHO_MERGE::IDENTITY;
    const bool smooth = merge == MORPHO_MERGE::SMOOTH;
    const bool winner_takes_all = merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN;
    const MorphoMergeCUDA<T> m{merge};
    const MorphoBorderCUDA<T> border{false, std::numeric_limits<T>::quiet_NaN()};
    const int groups = (g.out_channels + OB - 1) / OB;
    const int h0 = blockIdx.y * MORPHO_TILE_H;
    const int w0 = blockIdx.x * MORPHO_TILE_W;
    const int h_first = h0 + threadIdx.y * MORPHO_ROWS;
    const int w = w0 + threadIdx.x;
    const int64_t out_planes = static_cast<int64_t>(g.batch_size) * g.out_channels * g.plane();

    for (int z = blockIdx.z; z < g.batch_size * groups; z += gridDim.z)
    {
        const int b = z / groups;
        const int o0 = (z % groups) * OB;
        const int valid = min(OB, g.out_channels - o0);
        const int c_begin = identity ? o0 : 0;
        const int c_end = identity ? o0 + 1 : g.in_channels;

        T beta_sign[OB], beta_scale[OB];
        T merged[OB][MORPHO_ROWS];
        SmorphSoftMergeCUDA<T> soft[OB][MORPHO_ROWS];
        T winners[OB][MORPHO_ROWS];
        T wanted[OB][MORPHO_ROWS];
#pragma unroll
        for (int ob = 0; ob < OB; ob++)
        {
            const T bt = smooth && ob < valid ? beta[o0 + ob] : static_cast<T>(0);
            beta_sign[ob] = bt >= static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(-1);
            beta_scale[ob] = (bt >= static_cast<T>(0) ? bt : -bt) * static_cast<T>(1.4426950408889634);
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                merged[ob][r] = m.initial_value();
                soft[ob][r].reset();
                winners[ob][r] = static_cast<T>(0);
                const int h = h_first + r;
                const bool inside = ob < valid && h < g.height && w < g.width;
                wanted[ob][r] = winner_takes_all && stats != nullptr && inside
                                    ? target[((static_cast<int64_t>(b) * g.out_channels + o0 + ob) * g.height + h) * g.width + w]
                                    : static_cast<T>(0);
            }
        }

        for (int c = c_begin; c < c_end; c++)
        {
            SmorphPairsCUDA<T, OB> pairs;
            pairs.load(alpha, valid, [&](int ob) { return identity ? o0 : (o0 + ob) * g.in_channels + c; });
            __syncthreads();  // The previous channel is no longer read.
            morpho_load_window(g, input + (static_cast<int64_t>(b) * g.in_channels + c) * g.plane(), h0, w0, border, window);
            smorph_load_weights<T, OB>(weight + (static_cast<int64_t>(o0) * g.weight_in_channels + (identity ? 0 : c)) * g.taps(),
                                       static_cast<int64_t>(g.weight_in_channels) * g.taps(), valid, g.taps(), pairs, weights);
            __syncthreads();

            T ref[OB][MORPHO_ROWS], inv_sum[OB][MORPHO_ROWS], out[OB][MORPHO_ROWS];
            smorph_window_forward<T, OB>(g, window, weights, pairs, ref, inv_sum, out);
#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    const T channel_out = pairs.sign[ob] * out[ob][r];
                    if (smooth)
                    {
                        soft[ob][r].add(channel_out, beta_sign[ob], beta_scale[ob]);
                    }
                    else
                    {
                        merged[ob][r] = m.merge_value(merged[ob][r], channel_out);
                    }
                    const T distance = channel_out - wanted[ob][r];
                    winners[ob][r] += distance < winner_tolerance && distance > -winner_tolerance ? static_cast<T>(1) : static_cast<T>(0);
                }
            }
        }

#pragma unroll
        for (int ob = 0; ob < OB; ob++)
        {
#pragma unroll
            for (int r = 0; r < MORPHO_ROWS; r++)
            {
                const int h = h_first + r;
                if (ob >= valid || h >= g.height || w >= g.width)
                {
                    continue;
                }
                const int64_t out_index = ((static_cast<int64_t>(b) * g.out_channels + o0 + ob) * g.height + h) * g.width + w;
                if (output != nullptr)
                {
                    output[out_index] = smooth ? soft[ob][r].weighted / soft[ob][r].sum : m.finalize(merged[ob][r], g.in_channels);
                }
                if (stats != nullptr && winner_takes_all)
                {
                    stats[out_index] = winners[ob][r];
                }
                if (stats != nullptr && smooth)
                {
                    stats[out_index] = soft[ob][r].max_u;
                    stats[out_planes + out_index] = static_cast<T>(1) / soft[ob][r].sum;
                }
            }
        }
    }
}

/// @brief Backward tile kernel of every merge. A block owns one input channel of a tile and visits
///        the output channels in groups of OB, which share each input value read. Every tap of every
///        pair receives gradient, so the input gradient is accumulated in a shared window and added
///        to global memory once per block; the weight gradient goes through warp sums into bins
///        private to each warp, which need no atomics.
/// @param output Forward output, the merge target of MAX and MIN and the final value of SMOOTH.
/// @param stats Output of smorph_forward_kernel in stats mode, for MAX, MIN and SMOOTH.
/// @param grad_alpha Gradient of every channel pair, same layout as alpha.
/// @param grad_beta Gradient of beta, SMOOTH only.
template <typename T, int OB>
__global__ void __launch_bounds__(MORPHO_THREADS)
    smorph_backward_kernel(const MorphoGeometryCUDA g, const T* __restrict__ input, const T* __restrict__ weight, const T* __restrict__ alpha,
                           const T* __restrict__ beta, const T* __restrict__ output, const T* __restrict__ stats, const T* __restrict__ grad_output,
                           T* __restrict__ grad_input, T* __restrict__ grad_weight, T* __restrict__ grad_alpha, T* __restrict__ grad_beta,
                           const MORPHO_MERGE merge, const T winner_tolerance)
{
    extern __shared__ __align__(sizeof(double)) unsigned char morpho_shared[];
    const int window_size = g.window_size();
    const int taps = g.taps();
    T* window = reinterpret_cast<T*>(morpho_shared);
    T* grad_window = window + window_size;
    T* weights = grad_window + window_size;
    T* bins = weights + OB * taps;  // [warp][OB][taps]
    T* warp_bins = bins + threadIdx.y * OB * taps;

    const bool identity = merge == MORPHO_MERGE::IDENTITY;
    const bool smooth = merge == MORPHO_MERGE::SMOOTH;
    const bool winner_takes_all = merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN;
    const T scale = merge == MORPHO_MERGE::AVERAGE ? static_cast<T>(1) / static_cast<T>(g.in_channels) : static_cast<T>(1);
    const MorphoBorderCUDA<T> border{false, std::numeric_limits<T>::quiet_NaN()};
    const int h0 = blockIdx.y * MORPHO_TILE_H;
    const int w0 = blockIdx.x * MORPHO_TILE_W;
    const int h_first = h0 + threadIdx.y * MORPHO_ROWS;
    const int w = w0 + threadIdx.x;
    const int64_t out_planes = static_cast<int64_t>(g.batch_size) * g.out_channels * g.plane();

    for (int z = blockIdx.z; z < g.batch_size * g.in_channels; z += gridDim.z)
    {
        const int b = z / g.in_channels;
        const int c = z - b * g.in_channels;

        __syncthreads();  // The previous channel is flushed.
        morpho_load_window(g, input + static_cast<int64_t>(z) * g.plane(), h0, w0, border, window);
        morpho_zero_shared(grad_window, window_size);

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
            SmorphPairsCUDA<T, OB> pairs;
            pairs.load(alpha, valid, [&](int ob) { return identity ? c : (o0 + ob) * g.in_channels + c; });
            smorph_load_weights<T, OB>(weight + (static_cast<int64_t>(o0) * g.weight_in_channels + (identity ? 0 : c)) * taps,
                                       static_cast<int64_t>(g.weight_in_channels) * taps, valid, taps, pairs, weights);
            morpho_zero_shared(bins, MORPHO_BLOCK_Y * OB * taps);
            __syncthreads();

            T ref[OB][MORPHO_ROWS], inv_sum[OB][MORPHO_ROWS], out[OB][MORPHO_ROWS];
            smorph_window_forward<T, OB>(g, window, weights, pairs, ref, inv_sum, out);

            // Gradient routed to each pair by the merge, then scaled by 1 / S for the tap loop.
            T beta_grad[OB];
#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
                beta_grad[ob] = static_cast<T>(0);
                const T bt = smooth && ob < valid ? beta[o0 + ob] : static_cast<T>(0);
#pragma unroll
                for (int r = 0; r < MORPHO_ROWS; r++)
                {
                    if (grad[ob][r] == static_cast<T>(0))
                    {
                        continue;
                    }
                    const int64_t out_index = ((static_cast<int64_t>(b) * g.out_channels + o0 + ob) * g.height + h_first + r) * g.width + w;
                    const T channel_out = pairs.sign[ob] * out[ob][r];
                    if (winner_takes_all)
                    {
                        // The channels within winner_tolerance of the MAX / MIN share the gradient.
                        const T distance = channel_out - output[out_index];
                        const bool winner = distance < winner_tolerance && distance > -winner_tolerance;
                        grad[ob][r] = winner ? grad[ob][r] / stats[out_index] : static_cast<T>(0);
                    }
                    else if (smooth)
                    {
                        // Channel i receives dfinal/dc_i = p_i (1 + beta (c_i - final)), and beta
                        // receives p_i (c_i - ref) (c_i - final), with ref the hard extremum.
                        const T bt_sign = bt >= static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(-1);
                        const T max_u = stats[out_index];
                        const T p = smorph_exp2((bt_sign * channel_out - max_u) * (bt_sign * bt) * static_cast<T>(1.4426950408889634))
                                    * stats[out_planes + out_index];
                        const T centered = channel_out - output[out_index];
                        beta_grad[ob] += p * (channel_out - bt_sign * max_u) * centered * grad[ob][r];
                        grad[ob][r] *= p + bt * p * centered;
                    }
                    grad[ob][r] *= inv_sum[ob][r];
                }
            }

            T alpha_grad[OB];
#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
                alpha_grad[ob] = static_cast<T>(0);
            }
            smorph_window_backward<T, OB>(g, window, weights, pairs, ref, out, grad, grad_window, warp_bins, alpha_grad);

#pragma unroll
            for (int ob = 0; ob < OB; ob++)
            {
                alpha_grad[ob] = morpho_warp_sum(alpha_grad[ob]);
                beta_grad[ob] = morpho_warp_sum(beta_grad[ob]);
                if (threadIdx.x == 0 && ob < valid)
                {
                    atomicAdd(grad_alpha + (identity ? c : (o0 + ob) * g.in_channels + c), alpha_grad[ob]);
                    if (smooth)
                    {
                        atomicAdd(grad_beta + o0 + ob, beta_grad[ob]);
                    }
                }
            }
            __syncthreads();

            for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < valid * taps; i += blockDim.x * blockDim.y)
            {
                T sum = static_cast<T>(0);
                for (int warp = 0; warp < MORPHO_BLOCK_Y; warp++)
                {
                    sum += bins[warp * OB * taps + i];
                }
                if (sum != static_cast<T>(0))
                {
                    const int ob = i / taps;
                    atomicAdd(grad_weight + (static_cast<int64_t>(o0 + ob) * g.weight_in_channels + (identity ? 0 : c)) * taps + (i - ob * taps),
                              sum);
                }
            }
        }

        __syncthreads();
        morpho_flush_window(g, grad_window, h0, w0, grad_input + static_cast<int64_t>(z) * g.plane());
    }
}
