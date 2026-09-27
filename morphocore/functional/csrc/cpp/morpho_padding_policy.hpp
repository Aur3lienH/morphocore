#pragma once
#include <algorithm>
#include <cstdint>

#include "morpho_op_enum.hpp"
#include "morpho_op_policy.hpp"

// Every border rule of the CPU path lives in this header. A new padding mode needs a value in
// MORPHO_PADDING plus either one case in morpho_fill_value (value based mode) or one branch in
// morpho_pad_rows (structural mode); nothing else in the CPU kernels has to change.
//
// The kernels never test bounds: each work unit copies the input rows it reads into a small padded
// tile, so that every tap reads plain memory and the inner loops vectorise. The copy costs
// O(H * W) against the O(H * W * K^2) of the operation itself.

/// @brief Resolve the value an out of bounds tap reads for a given padding mode.
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

/// @brief Copy rows of one [height, width] plane into a padded buffer.
/// @tparam kind Padding family.
/// @tparam T Data type of tensor elements.
/// @param plane Source plane.
/// @param pad_h Rows added on the top and on the bottom of the plane.
/// @param pad_w Columns added on the left and on the right of each row.
/// @param row_begin First padded row to produce; padded row r holds input row r - pad_h.
/// @param rows Number of padded rows to produce.
/// @param fill Fill value from morpho_fill_value, unused when kind is REPLICATE.
/// @param dst Destination, rows * (width + 2 * pad_w) values.
template <MORPHO_PADDING_KIND kind, typename T>
void morpho_pad_rows(const T* __restrict__ plane, int64_t height, int64_t width, int64_t pad_h, int64_t pad_w, int64_t row_begin, int64_t rows,
                     T fill, T* __restrict__ dst)
{
    const int64_t padded_width = width + 2 * pad_w;
    for (int64_t r = 0; r < rows; r++)
    {
        const int64_t y = row_begin + r - pad_h;
        T* out = dst + r * padded_width;

        if constexpr (kind == MORPHO_PADDING_KIND::REPLICATE)
        {
            const T* row = plane + std::clamp<int64_t>(y, 0, height - 1) * width;
            std::fill(out, out + pad_w, row[0]);
            std::copy(row, row + width, out + pad_w);
            std::fill(out + pad_w + width, out + padded_width, row[width - 1]);
        }
        else if (y < 0 || y >= height)
        {
            std::fill(out, out + padded_width, fill);
        }
        else
        {
            const T* row = plane + y * width;
            std::fill(out, out + pad_w, fill);
            std::copy(row, row + width, out + pad_w);
            std::fill(out + pad_w + width, out + padded_width, fill);
        }
    }
}
