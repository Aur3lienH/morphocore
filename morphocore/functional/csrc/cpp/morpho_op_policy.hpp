#pragma once
#include <limits>

#include "morpho_cpu_simd.hpp"
#include "morpho_op_enum.hpp"

/// @brief Policy for morphological operations to differentiate between erosion and dilation.
/// @tparam T Data type of tensor elements.
/// @tparam op Morphological operation type (DILATION or EROSION).
template <typename T, MORPHO_OPERATION op>
struct MorphoPolicyCPU
{
    static constexpr T padding_value()
    {
        if constexpr (op == MORPHO_OPERATION::DILATION)
        {
            return -std::numeric_limits<T>::infinity();
        }
        else
        {
            return std::numeric_limits<T>::infinity();
        }
    }

    static constexpr T initial_value()
    {
        if constexpr (op == MORPHO_OPERATION::DILATION)
        {
            return -std::numeric_limits<T>::infinity();
        }
        else
        {
            return std::numeric_limits<T>::infinity();
        }
    }

    /// @note Generic so that the SIMD kernels use it on whole vectors: V is T or a MorphoSimd vector.
    template <typename V>
    static MORPHO_INLINE V combine(V input_val, T weight_val)
    {
        if constexpr (op == MORPHO_OPERATION::DILATION)
        {
            return input_val + weight_val;
        }
        else
        {
            return input_val - weight_val;
        }
    }

    /// @return A bool for scalars, a lane mask for MorphoSimd vectors.
    template <typename V>
    static MORPHO_INLINE auto is_better(V val, V best)
    {
        if constexpr (op == MORPHO_OPERATION::DILATION)
        {
            return val > best;
        }
        else
        {
            return val < best;
        }
    }
};
