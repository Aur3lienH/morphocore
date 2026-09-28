#include "smorph.cuh"
#include "morpho_cuda_launch.cuh"
#include "smorph_kernels.cuh"

// Launchers of the CUDA Smorph kernels (smorph_kernels.cuh).

/// @brief Channels within this distance of the MAX / MIN merge share its gradient, as on the CPU.
template <typename T>
constexpr T SMORPH_WINNER_TOLERANCE = static_cast<T>(1e-6);

/// @brief Forward kernel launch, also used in stats mode by the backward pass.
template <typename T>
void smorph_launch_forward(const MorphoGeometryCUDA& g, const at::Tensor& input, const at::Tensor& weight, const at::Tensor& alpha,
                           const at::Tensor& beta, T* output, const T* target, T* stats, MORPHO_MERGE merge)
{
    const auto shared_bytes = [&](int group) { return (g.window_size() + group * g.taps()) * sizeof(T); };
    const int group = merge == MORPHO_MERGE::IDENTITY ? morpho_channel_group(g, 1, {1}, shared_bytes)
                                                      : morpho_channel_group(g, g.out_channels, {4, 1}, shared_bytes);
    morpho_dispatch_group<4, 1>(group, [&](auto group_constant) {
        constexpr int OB = decltype(group_constant)::value;
        const auto kernel = smorph_forward_kernel<T, OB>;
        morpho_allow_shared_memory(kernel, shared_bytes(OB));
        kernel<<<morpho_tile_grid(g, (g.out_channels + OB - 1) / OB), dim3(MORPHO_BLOCK_X, MORPHO_BLOCK_Y), shared_bytes(OB),
                 at::cuda::getCurrentCUDAStream()>>>(g, input.data_ptr<T>(), weight.data_ptr<T>(), alpha.data_ptr<T>(),
                                                     beta.defined() ? beta.data_ptr<T>() : nullptr, output, target, stats, merge,
                                                     SMORPH_WINNER_TOLERANCE<T>);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template <typename T>
void smorph_forward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& alpha, const at::Tensor& beta, const at::Tensor& output,
                         MORPHO_MERGE merge)
{
    const MorphoGeometryCUDA g = morpho_geometry_cuda(input.sizes(), weight.sizes());
    if (static_cast<int64_t>(g.batch_size) * g.out_channels * g.plane() == 0)
    {
        return;
    }
    smorph_launch_forward<T>(g, input, weight, alpha, beta, output.data_ptr<T>(), nullptr, nullptr, merge);
}

template <typename T>
void smorph_backward_cuda(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& alpha, const at::Tensor& beta,
                          const at::Tensor& output, const at::Tensor& grad_output, const at::Tensor& grad_input, const at::Tensor& grad_weight,
                          const at::Tensor& grad_alpha, const at::Tensor& grad_beta, MORPHO_MERGE merge)
{
    const MorphoGeometryCUDA g = morpho_geometry_cuda(input.sizes(), weight.sizes());
    if (static_cast<int64_t>(g.batch_size) * g.out_channels * g.plane() == 0)
    {
        return;
    }

    // MAX, MIN and SMOOTH route the gradient of a pixel according to every channel: a first pass counts
    // the winning channels, or keeps the extremum and the sum of the soft merge.
    at::Tensor stats;
    const bool smooth = merge == MORPHO_MERGE::SMOOTH;
    if (smooth || merge == MORPHO_MERGE::MAX || merge == MORPHO_MERGE::MIN)
    {
        stats = at::empty({smooth ? 2 : 1, g.batch_size, g.out_channels, g.height, g.width}, input.options());
        smorph_launch_forward<T>(g, input, weight, alpha, beta, nullptr, output.data_ptr<T>(), stats.data_ptr<T>(), merge);
    }

    // Input window, its gradient, the group's weights and one set of weight gradient bins per warp.
    const auto shared_bytes = [&](int group) { return (2 * g.window_size() + (1 + MORPHO_BLOCK_Y) * group * g.taps()) * sizeof(T); };
    const bool grouped = merge != MORPHO_MERGE::IDENTITY && g.out_channels > 1 && shared_bytes(4) <= morpho_max_shared_memory();
    const int group = grouped ? 4 : 1;
    TORCH_CHECK(shared_bytes(group) <= morpho_max_shared_memory(), "Structuring element too large for the CUDA kernels: ", g.kernel_height, "x",
                g.kernel_width);

    morpho_dispatch_group<4, 1>(group, [&](auto group_constant) {
        constexpr int OB = decltype(group_constant)::value;
        const auto kernel = smorph_backward_kernel<T, OB>;
        morpho_allow_shared_memory(kernel, shared_bytes(OB));
        kernel<<<morpho_tile_grid(g, g.in_channels), dim3(MORPHO_BLOCK_X, MORPHO_BLOCK_Y), shared_bytes(OB), at::cuda::getCurrentCUDAStream()>>>(
            g, input.data_ptr<T>(), weight.data_ptr<T>(), alpha.data_ptr<T>(), beta.defined() ? beta.data_ptr<T>() : nullptr, output.data_ptr<T>(),
            stats.defined() ? stats.data_ptr<T>() : nullptr, grad_output.data_ptr<T>(), grad_input.data_ptr<T>(), grad_weight.data_ptr<T>(),
            grad_alpha.data_ptr<T>(), grad_beta.defined() ? grad_beta.data_ptr<T>() : nullptr, merge, SMORPH_WINNER_TOLERANCE<T>);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

#define SMORPH_INSTANTIATE_CUDA(T)                                                                                                                   \
    template void smorph_forward_cuda<T>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,              \
                                         MORPHO_MERGE);                                                                                              \
    template void smorph_backward_cuda<T>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,             \
                                          const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,             \
                                          MORPHO_MERGE);

SMORPH_INSTANTIATE_CUDA(float)
SMORPH_INSTANTIATE_CUDA(double)
