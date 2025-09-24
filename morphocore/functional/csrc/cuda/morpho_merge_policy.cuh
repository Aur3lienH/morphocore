#pragma once
#include <limits>
#include "morpho_op_enum.hpp"

/// @brief Merge channels policy for GPU implementations.
/// @tparam scalar_t Data type of the tensors (float or double)
/// @tparam merge_op Merge channel mode to use.
template <typename scalar_t, MORPHO_MERGE merge_op>
struct MorphoMergePolicyGPU
{
    __device__ __forceinline__ static constexpr scalar_t get_init_value()
    {
        if constexpr (merge_op == MORPHO_MERGE::ADD)
        {
            return static_cast<scalar_t>(0);
        }
        else if constexpr (merge_op == MORPHO_MERGE::MAX)
        {
            return std::numeric_limits<scalar_t>::lowest();
        }
        else if constexpr (merge_op == MORPHO_MERGE::MIN)
        {
            return std::numeric_limits<scalar_t>::max();
        }
        else if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return static_cast<scalar_t>(0);
        }
    }

    __device__ __forceinline__ static scalar_t merge(scalar_t a, scalar_t b)
    {
        if constexpr (merge_op == MORPHO_MERGE::ADD)
        {
            return a + b;
        }
        else if constexpr (merge_op == MORPHO_MERGE::MAX)
        {
            return fmaxf(a, b);
        }
        else if constexpr (merge_op == MORPHO_MERGE::MIN)
        {
            return fminf(a, b);
        }
        else if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return a + b;
        }
    }

    __device__ __forceinline__ static constexpr scalar_t finalize(scalar_t a, int in_channels)
    {
        if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return a / static_cast<scalar_t>(in_channels);
        }
        else
        {
            return a;
        }
    }
};