import torch
from cpp_operation import morpho_dilation
from morphocore.functional.merge_enum import str_to_merge 

def dilation(x: torch.Tensor, w: torch.Tensor, channel_merge_mode: str = 'sum') -> torch.Tensor:
    """
    Make a dilation by calling C++ or CUDA.

    Args:
        x (torch.Tensor): input
        w (torch.Tensor): weight
        channel_merge_mode (str): channel merge mode
    """
    return morpho_dilation(x, w.flip((-2, -1)), str_to_merge(channel_merge_mode))