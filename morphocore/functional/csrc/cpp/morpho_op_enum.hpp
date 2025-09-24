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
    AVERAGE
};