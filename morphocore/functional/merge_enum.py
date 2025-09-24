from cpp_operation import MorphoMerge


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
    else:
        raise ValueError(f"Unknown merge mode: {mode}")