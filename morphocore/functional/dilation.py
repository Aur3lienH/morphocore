import torch
from cpp_operation import morpho_dilation
from morphocore.functional.merge_enum import str_to_merge 

def dilation(x: torch.Tensor, w: torch.Tensor, channel_merge_mode: str = 'sum', save_indices: bool = True) -> torch.Tensor:
    """
    Make a dilation by calling C++ or CUDA.

    Args:
        x (torch.Tensor): input
        w (torch.Tensor): weight
        channel_merge_mode (str): channel merge mode
    """
    return morpho_dilation(x, w, str_to_merge(channel_merge_mode), save_indices)