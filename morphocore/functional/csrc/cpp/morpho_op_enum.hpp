#pragma once

enum class MORPHO_OPERATION
{
    DILATION,
    EROSION
};

enum class MORPHO_OPTI
{
    NONE,
    INDEX
};

enum class MORPHO_MERGE
{
    MAX,
    MIN,
    ADD,
    AVERAGE
};