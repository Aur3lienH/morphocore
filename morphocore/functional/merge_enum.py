from enum import IntEnum


class MorphoMerge(IntEnum):
    """Merge channel modes, same values as MORPHO_MERGE in csrc/cpp/morpho_op_enum.hpp."""
    MAX = 0
    MIN = 1
    ADD = 2
    AVERAGE = 3
    SMOOTH = 4
    IDENTITY = 5


class MorphoPadding(IntEnum):
    """Padding modes, same values as MORPHO_PADDING in csrc/cpp/morpho_op_enum.hpp."""
    ZERO = 0
    GEODESIC = 1
    CONSTANT = 2
    REPLICATE = 3


def str_to_merge(mode: str) -> MorphoMerge:
    """
    Convert a string to a MorphoMerge enum.
    
    Args:
        mode (str): "max", "min", "sum", "mean"
    
    Returns:
        MorphoMerge: Corresponding enum value.
    """

    if mode == "max":
        return MorphoMerge.MAX
    elif mode == "min":
        return MorphoMerge.MIN
    elif mode == "sum":
        return MorphoMerge.ADD
    elif mode == "mean":
        return MorphoMerge.AVERAGE
    elif mode == "identity":
        return MorphoMerge.IDENTITY
    else:
        raise ValueError(f"Unknown merge mode: {mode}")
    
_PADDING_MODES = {
    "zero": MorphoPadding.ZERO,
    "geodesic": MorphoPadding.GEODESIC,
    "constant": MorphoPadding.CONSTANT,
    "replicate": MorphoPadding.REPLICATE,
}


def str_to_padding(mode: str) -> MorphoPadding:
    """
    Convert a string to a MorphoPadding enum.

    Args:
        mode (str): "zero", "geodesic", "constant" or "replicate"

    Returns:
        MorphoPadding: Corresponding enum value.
    """

    try:
        return _PADDING_MODES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown padding mode: {mode}. Expected one of {sorted(_PADDING_MODES)}."
        ) from None
