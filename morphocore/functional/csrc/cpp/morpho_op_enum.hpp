#pragma once

// Enum for morphological operations: Dilation and Erosion.
enum class MORPHO_OPERATION
{
    DILATION,
    EROSION
};

// Enum for optimization strategies. (Saving the indexes or not)
enum class MORPHO_OPTI
{
    NONE,
    INDEX
};

// Merge channel policy.
enum class MORPHO_MERGE
{
    MAX,
    MIN,
    ADD,
    AVERAGE,
    SMOOTH,
    IDENTITY
};

// Padding policy, as selected by the caller.
enum class MORPHO_PADDING
{
    ZERO,       // Border reads as 0.
    GEODESIC,   // Border reads as the neutral element of the operation, so it can never win.
    CONSTANT,   // Border reads as a caller supplied value.
    REPLICATE   // Border reads as the nearest edge pixel.
};

// Structural family a padding mode belongs to.
// Kernels are templated on the kind instead of on MORPHO_PADDING: every mode that fills the border
// with some value shares one implementation and differs only by a runtime scalar, so adding a new
// value based mode costs zero extra template instantiations.
enum class MORPHO_PADDING_KIND
{
    FILL,      // Out of bounds taps read a fill value (zero, geodesic, constant).
    REPLICATE  // Out of bounds taps read the clamped edge pixel.
};

/// @brief Map a padding mode onto the kernel family that implements it.
constexpr MORPHO_PADDING_KIND padding_kind(MORPHO_PADDING padding)
{
    return padding == MORPHO_PADDING::REPLICATE ? MORPHO_PADDING_KIND::REPLICATE : MORPHO_PADDING_KIND::FILL;
}