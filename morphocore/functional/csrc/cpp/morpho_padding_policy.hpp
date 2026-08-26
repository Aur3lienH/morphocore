#pragma once
#include <algorithm>
#include <cstdint>

#include "morpho_op_enum.hpp"
#include "morpho_op_policy.hpp"

// Every border rule of the library lives in this header. A new padding mode needs a value in
// MORPHO_PADDING plus either one case in morpho_fill_value (value based mode) or one branch in
// morpho_row / morpho_load_row (structural mode); nothing else in the CPU kernels has to change.
//
// Resolution is split in two so the kernels keep the per row work out of their innermost loop:
// morpho_row is called once per kernel row, morpho_load_row once per tap.

/// @brief Resolve the value an out of bounds tap reads for a given padding mode.
/// Called once per operation and hoisted out of the kernel loops, so the switch is free.
/// @tparam op Morphological operation type, only used to pick the geodesic neutral element.
/// @tparam T Data type of tensor elements.
/// @param padding Padding mode requested by the caller.
/// @param constant_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes.
/// @return The fill value. Meaningless for REPLICATE, which never reads one.
template <MORPHO_OPERATION op, typename T>
inline T morpho_fill_value(MORPHO_PADDING padding, double constant_value)
{
    switch (padding)
    {
        case MORPHO_PADDING::ZERO: return static_cast<T>(0);
        case MORPHO_PADDING::CONSTANT: return static_cast<T>(constant_value);
        // GEODESIC pads with the neutral element (-inf for dilation, +inf for erosion) so that a
        // border tap can never be selected as the extremum.
        default: return MorphoPolicyCPU<T, op>::padding_value();
    }
}

/// @brief Resolve the source row a tap row reads from, once per kernel row.
/// @tparam kind Padding family.
/// @tparam T Data type of tensor elements.
/// @param plane Base pointer of the [height, width] channel plane.
/// @return Pointer to the source row, or nullptr when the whole row is padding (FILL only).
/// @note The bounds test casts to unsigned so a negative coordinate wraps to a huge value and fails
///       the upper bound check: one comparison instead of two.
template <MORPHO_PADDING_KIND kind, typename T>
inline const T* morpho_row(const T* __restrict__ plane, int64_t ph, int64_t height, int64_t width)
{
    if constexpr (kind == MORPHO_PADDING_KIND::REPLICATE)
    {
        return plane + std::min(std::max(ph, int64_t(0)), height - 1) * width;
    }
    else
    {
        return static_cast<uint64_t>(ph) < static_cast<uint64_t>(height) ? plane + ph * width : nullptr;
    }
}

/// @brief Read one tap from a row already resolved by morpho_row.
/// @tparam kind Padding family.
/// @tparam T Data type of tensor elements.
/// @param row Row pointer from morpho_row; nullptr means the row lies entirely in the padding.
/// @param fill Fill value from morpho_fill_value, unused when kind is REPLICATE.
/// @note REPLICATE clamps with std::min/std::max, which lower to branchless cmov; the row is never
///       null so nothing else is tested. For FILL the null test is loop invariant and hoists out.
template <MORPHO_PADDING_KIND kind, typename T>
inline T morpho_load_row(const T* __restrict__ row, int64_t pw, int64_t width, T fill)
{
    if constexpr (kind == MORPHO_PADDING_KIND::REPLICATE)
    {
        return row[std::min(std::max(pw, int64_t(0)), width - 1)];
    }
    else
    {
        return (row != nullptr && static_cast<uint64_t>(pw) < static_cast<uint64_t>(width)) ? row[pw] : fill;
    }
}
