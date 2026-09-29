#include "morpho_op.cuh"
#include "morpho_cuda_launch.cuh"
#include "morpho_op_kernels.cuh"

// Launchers of the CUDA dilation and erosion kernels (morpho_op_kernels.cuh).

/// @brief Output channel group sizes compiled for the tile kernels of an element type. Doubles run at
///        a small fraction of the float rate on most GPUs and gain nothing from the larger groups,
///        whose register use also halves occupancy.
template <typename T, typename F>
void morpho_dispatch_output_group(int group, F f)
{
    if constexpr (sizeof(T) == sizeof(double))
    {
        morpho_dispatch_group<4, 1>(group, f);
    }
    else
    {
        morpho_dispatch_group<8, 4, 1>(group, f);
    }
}

template <MORPHO_OPERATION op, typename T>
void morpho_forward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& output, const at::Tensor& indices, MORPHO_MERGE merge,
                         bool replicate, T fill)
{
    MorphoGeometryCUDA g = morpho_geometry_cuda(input.sizes(), weight.sizes());
    if (static_cast<int64_t>(g.batch_size) * g.out_channels * g.plane() == 0)
    {
        return;
    }
    TORCH_CHECK(!indices.defined() || (g.taps() < (1 << 16) && g.in_channels < (1 << 15)),
                "Structuring element or channel count too large to save indices");

    const auto shared_bytes = [&](int group) { return (g.window_size() + group * g.taps()) * sizeof(T); };
    const bool doubles = sizeof(T) == sizeof(double);
    const int group =
        merge == MORPHO_MERGE::IDENTITY
            ? morpho_channel_group(g, 1, {1}, shared_bytes)
            : morpho_channel_group(g, g.out_channels, doubles ? std::initializer_list<int>{4, 1} : std::initializer_list<int>{8, 4, 1}, shared_bytes);
    morpho_fit_block_rows(g, static_cast<int64_t>(g.batch_size) * ((g.out_channels + group - 1) / group));
    const MorphoIndexMode index_mode = !indices.defined()                                      ? MorphoIndexMode::NONE
                                       : merge == MORPHO_MERGE::IDENTITY || g.in_channels == 1 ? MorphoIndexMode::TRACK
                                                                                               : MorphoIndexMode::SEARCH;

    morpho_dispatch_output_group<T>(group, [&](auto group_constant) {
        constexpr int OB = decltype(group_constant)::value;
        const auto kernel = index_mode == MorphoIndexMode::NONE     ? morpho_forward_kernel<op, T, OB, MorphoIndexMode::NONE>
                            : index_mode == MorphoIndexMode::SEARCH ? morpho_forward_kernel<op, T, OB, MorphoIndexMode::SEARCH>
                                                                    : morpho_forward_kernel<op, T, OB, MorphoIndexMode::TRACK>;
        morpho_allow_shared_memory(kernel, shared_bytes(OB));
        kernel<<<morpho_tile_grid(g, (g.out_channels + OB - 1) / OB), morpho_tile_block(g), shared_bytes(OB), at::cuda::getCurrentCUDAStream()>>>(
            g, input.data_ptr<T>(), weight.data_ptr<T>(), output.defined() ? output.data_ptr<T>() : nullptr,
            indices.defined() ? indices.data_ptr<int32_t>() : nullptr, MorphoMergeCUDA<T>{merge}, MorphoBorderCUDA<T>{replicate, fill});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

/// @brief Backward of the MAX and MIN merges, from saved or recomputed indices.
template <MORPHO_OPERATION op, typename T>
void morpho_backward_indices_cuda(const MorphoGeometryCUDA& g, const at::Tensor& indices, const at::Tensor& grad_output, const at::Tensor& grad_input,
                                  const at::Tensor& grad_weight)
{
    // Shared bins for floats; doubles, whose shared atomics are 64 bit compare and swap loops, measured
    // faster with global atomics.
    const size_t bins_bytes = static_cast<size_t>(g.in_channels) * g.taps() * sizeof(T);
    const bool shared_bins = sizeof(T) == sizeof(float) && bins_bytes <= morpho_max_shared_memory();
    // A few blocks per SM, and enough pixels per block to amortise the flush of its bins.
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t plane = g.plane();
    int64_t segments = std::max<int64_t>(1, (4 * sms + g.out_channels - 1) / g.out_channels);
    segments = std::min(segments, std::max<int64_t>(1, plane * g.batch_size / std::max<int64_t>(MORPHO_THREADS, 4 * g.in_channels * g.taps())));
    segments = std::min(segments, (plane + MORPHO_THREADS - 1) / MORPHO_THREADS);
    const int pixels_per_block = static_cast<int>((plane + segments - 1) / segments);

    const auto kernel = morpho_backward_indices_kernel<op, T>;
    const size_t shared = shared_bins ? bins_bytes : 0;
    morpho_allow_shared_memory(kernel, shared);
    kernel<<<dim3(static_cast<unsigned>((plane + pixels_per_block - 1) / pixels_per_block), g.out_channels), dim3(MORPHO_BLOCK_X, MORPHO_BLOCK_Y),
             shared, at::cuda::getCurrentCUDAStream()>>>(g, indices.data_ptr<int32_t>(), grad_output.data_ptr<T>(), grad_input.data_ptr<T>(),
                                                         grad_weight.data_ptr<T>(), pixels_per_block, shared_bins);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <MORPHO_OPERATION op, typename T>
void morpho_backward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& indices, const at::Tensor& grad_output,
                          const at::Tensor& grad_input, const at::Tensor& grad_weight, MORPHO_MERGE merge, bool replicate, T fill)
{
    MorphoGeometryCUDA g = morpho_geometry_cuda(input.sizes(), weight.sizes());
    if (static_cast<int64_t>(g.batch_size) * g.out_channels * g.plane() == 0)
    {
        return;
    }

    if (merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN)
    {
        // Without saved indices, the forward kernel recomputes them.
        at::Tensor winners = indices;
        if (!winners.defined())
        {
            winners = at::empty({g.batch_size, g.out_channels, g.height, g.width}, input.options().dtype(at::kInt));
            morpho_forward_cuda<op, T>(input, weight, at::Tensor(), winners, merge, replicate, fill);
        }
        morpho_backward_indices_cuda<op, T>(g, winners, grad_output, grad_input, grad_weight);
        return;
    }

    // Floats: enough copies of the weight gradient bins to spread the lanes of a warp, within a small
    // budget. Doubles: no shared bins (their shared atomics are 64 bit compare and swap loops), and
    // groups of one channel, which hide the latency of the slow double units best; both measured.
    const bool doubles = sizeof(T) == sizeof(double);
    const auto copies = [&](int group) {
        int count = doubles ? 0 : 32;
        while (count > 1 && group * g.taps() * count > 2048)
        {
            count /= 2;
        }
        return count;
    };
    const auto shared_bytes = [&](int group) { return (g.window_size() + group * g.taps() * (1 + copies(group))) * sizeof(T); };
    morpho_fit_block_rows(g, static_cast<int64_t>(g.batch_size) * g.in_channels);
    // Blocks own an input channel whatever the group size, so the larger group whenever it is useful and fits.
    const bool grouped = !doubles && merge != MORPHO_MERGE::IDENTITY && g.out_channels > 1 && shared_bytes(4) <= morpho_max_shared_memory();
    const int group = grouped ? 4 : 1;
    TORCH_CHECK(shared_bytes(group) <= morpho_max_shared_memory(), "Structuring element too large for the CUDA kernels: ", g.kernel_height, "x",
                g.kernel_width);

    const auto launch = [&](auto group_constant) {
        constexpr int OB = decltype(group_constant)::value;
        const auto kernel = morpho_backward_pairs_kernel<op, T, OB>;
        morpho_allow_shared_memory(kernel, shared_bytes(OB));
        kernel<<<morpho_tile_grid(g, g.in_channels), morpho_tile_block(g), shared_bytes(OB), at::cuda::getCurrentCUDAStream()>>>(
            g, input.data_ptr<T>(), weight.data_ptr<T>(), grad_output.data_ptr<T>(), grad_input.data_ptr<T>(), grad_weight.data_ptr<T>(),
            MorphoMergeCUDA<T>{merge}, MorphoBorderCUDA<T>{replicate, fill}, copies(OB));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if constexpr (sizeof(T) == sizeof(double))
    {
        morpho_dispatch_group<1>(group, launch);
    }
    else
    {
        morpho_dispatch_group<4, 1>(group, launch);
    }
}

#define MORPHO_INSTANTIATE_CUDA(OP, T)                                                                                                               \
    template void morpho_forward_cuda<MORPHO_OPERATION::OP, T>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,           \
                                                               MORPHO_MERGE, bool, T);                                                               \
    template void morpho_backward_cuda<MORPHO_OPERATION::OP, T>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,          \
                                                                const at::Tensor&, const at::Tensor&, MORPHO_MERGE, bool, T);

MORPHO_INSTANTIATE_CUDA(DILATION, float)
MORPHO_INSTANTIATE_CUDA(DILATION, double)
MORPHO_INSTANTIATE_CUDA(EROSION, float)
MORPHO_INSTANTIATE_CUDA(EROSION, double)
