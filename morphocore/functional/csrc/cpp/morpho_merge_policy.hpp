#pragma once
#include "morpho_op_enum.hpp"

/// @brief Merge policy for morphological operations to differentiate between erosion and dilation.
/// @tparam T Data type of tensor elements.
/// @tparam merge_op Merge channel policy (ADD, MAX, MIN, AVERAGE).
template <typename T, MORPHO_MERGE merge_op>
struct MorphoMergePolicy
{
    static constexpr T get_init_value()
    {
        if constexpr (merge_op == MORPHO_MERGE::ADD)
        {
            return static_cast<T>(0);
        }
        else if constexpr (merge_op == MORPHO_MERGE::MAX)
        {
            return std::numeric_limits<T>::lowest();
        }
        else if constexpr (merge_op == MORPHO_MERGE::MIN)
        {
            return std::numeric_limits<T>::max();
        }
        else if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return static_cast<T>(0);
        }
        else
        {
            // static_assert(always_false<decltype(merge_op)>, "Invalid merge operation");
            return static_cast<T>(0);
        }
    }

    static T merge(T a, T b)
    {
        if constexpr (merge_op == MORPHO_MERGE::ADD)
        {
            return a + b;
        }
        else if constexpr (merge_op == MORPHO_MERGE::MAX)
        {
            return std::max(a, b);
        }
        else if constexpr (merge_op == MORPHO_MERGE::MIN)
        {
            return std::min(a, b);
        }
        else if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return a + b;
        }
        else
        {
            // static_assert(always_false<decltype(merge_op)>, "Invalid merge operation");
            return a;
        }
    }

    static T finalize(T a, int in_channels)
    {
        if constexpr (merge_op == MORPHO_MERGE::AVERAGE)
        {
            return a / static_cast<T>(in_channels);
        }
        else
        {
            return a;
        }
    }
};