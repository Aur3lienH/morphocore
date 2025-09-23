#pragma once
#include <limits>

#include "morpho_op_enum.hpp"

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

    static T combine(T input_val, T weight_val)
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

    static bool is_better(T val, T best)
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
