#pragma once
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>

#include "morpho_cuda_common.cuh"

// Host side helpers shared by the launchers of the tile kernels.

/// @brief Build the geometry from 4D input sizes and the weight's: 4D, or 5D with one structuring
///        element per batch element.
inline MorphoGeometryCUDA morpho_geometry_cuda(at::IntArrayRef input, at::IntArrayRef weight)
{
    const bool per_sample = weight.size() == 5;
    if (per_sample)
    {
        weight = weight.slice(1);
    }
    MorphoGeometryCUDA g;
    g.batch_size = static_cast<int>(input[0]);
    g.in_channels = static_cast<int>(input[1]);
    g.height = static_cast<int>(input[2]);
    g.width = static_cast<int>(input[3]);
    g.out_channels = static_cast<int>(weight[0]);
    g.weight_in_channels = static_cast<int>(weight[1]);
    g.kernel_height = static_cast<int>(weight[2]);
    g.kernel_width = static_cast<int>(weight[3]);
    g.pad_h = g.kernel_height / 2;
    g.pad_w = g.kernel_width / 2;
    g.block_rows = MORPHO_BLOCK_Y;
    g.weight_batch_stride = per_sample ? static_cast<int64_t>(g.out_channels) * g.weight_in_channels * g.taps() : 0;
    return g;
}

/// @brief Output tiles of one image plane.
inline int64_t morpho_tiles(const MorphoGeometryCUDA& g)
{
    return static_cast<int64_t>((g.width + MORPHO_TILE_W - 1) / MORPHO_TILE_W) * ((g.height + g.tile_height() - 1) / g.tile_height());
}

/// @brief Streaming multiprocessors of the current device.
inline int64_t morpho_sms()
{
    return at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

/// @brief Blocks a launch should have to keep every SM busy.
inline int64_t morpho_wanted_blocks()
{
    return 2 * morpho_sms();
}

/// @brief Zero a contiguous tensor with a memset, which costs the host much less than a fill kernel
///        dispatched through ATen: that matters for small problems, bound by the host.
inline void morpho_zero(const at::Tensor& tensor)
{
    if (tensor.numel() > 0)
    {
        C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr(), 0, tensor.nbytes(), at::cuda::getCurrentCUDAStream()));
    }
}

/// @brief Shared memory a block may use, opting in to more than the 48 KiB default when needed.
inline size_t morpho_max_shared_memory()
{
    return at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
}

template <typename Kernel>
void morpho_allow_shared_memory(Kernel kernel, size_t bytes)
{
    if (bytes > 48 * 1024)
    {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    }
}

/// @brief Largest channel group, among the compiled sizes, that fits in shared memory and still gives
///        the launch `wanted_blocks` blocks of full height tiles: larger groups reuse each shared
///        window more, but a small problem needs the blocks.
/// @param channels Channels split in groups.
/// @param sizes Compiled group sizes, decreasing.
/// @param shared_bytes Shared memory of a block for a group size.
template <typename SharedBytes>
int morpho_channel_group(const MorphoGeometryCUDA& g, int channels, std::initializer_list<int> sizes, SharedBytes shared_bytes,
                         int64_t wanted_blocks = morpho_wanted_blocks())
{
    int chosen = 0;
    for (int size : sizes)
    {
        if (shared_bytes(size) > morpho_max_shared_memory() || (size > channels && size != *(sizes.end() - 1)))
        {
            continue;
        }
        chosen = size;
        if (morpho_tiles(g) * g.batch_size * ((channels + size - 1) / size) >= wanted_blocks)
        {
            break;
        }
    }
    TORCH_CHECK(chosen > 0, "Structuring element too large for the CUDA kernels: ", g.kernel_height, "x", g.kernel_width);
    return chosen;
}

/// @brief Shorter tiles for a problem too small to give every SM a couple of blocks: the halo of the
///        taller tiles saves little there, and the idle SMs cost much more.
/// @param planes Planes of tiles in the launch (batch times channels or channel groups).
inline void morpho_fit_block_rows(MorphoGeometryCUDA& g, int64_t planes)
{
    while (g.block_rows > 1 && morpho_tiles(g) * planes < morpho_wanted_blocks())
    {
        g.block_rows /= 2;
    }
}

/// @brief Call f with a channel group size as a compile time constant, among the compiled sizes.
template <int... Sizes, typename F>
void morpho_dispatch_group(int group, F f)
{
    const bool found = ((group == Sizes ? (f(std::integral_constant<int, Sizes>{}), true) : false) || ...);
    TORCH_CHECK(found, "Unsupported channel group size ", group);
}

/// @brief Grid of the tile kernels: tiles in x and y, (batch, channel group) in z. The z dimension is
///        capped by the hardware limit and walked with a stride by the kernels.
inline dim3 morpho_tile_grid(const MorphoGeometryCUDA& g, int groups)
{
    const int64_t z = std::min<int64_t>(static_cast<int64_t>(g.batch_size) * groups, 65535);
    return dim3((g.width + MORPHO_TILE_W - 1) / MORPHO_TILE_W, (g.height + g.tile_height() - 1) / g.tile_height(), static_cast<unsigned>(z));
}

/// @brief Threads of a tile kernel block.
inline dim3 morpho_tile_block(const MorphoGeometryCUDA& g)
{
    return dim3(MORPHO_BLOCK_X, g.block_rows);
}
