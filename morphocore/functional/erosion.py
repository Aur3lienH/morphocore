from cpp_operation import morpho_erosion
from morphocore.functional.merge_enum import str_to_merge
import torch

def erosion(x: torch.Tensor, w: torch.Tensor, channel_merge_mode: str = 'sum') -> torch.Tensor:

    """
    Make an erosion by calling C++ or CUDA.

    Args:
        x (torch.Tensor): input
        w (torch.Tensor): weight
        channel_merge_mode (str): channel merge mode
    """
    return morpho_erosion(x, w, str_to_merge(channel_merge_mode))