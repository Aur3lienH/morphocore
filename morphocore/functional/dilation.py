import torch
from cpp_operation import morpho_dilation

def dilation(x, w):
    """
    Make a dilation by calling C++ or CUDA.

    Args:
        x (torch.Tensor): input
        w (torch.Tensor): weight
    """
    return morpho_dilation(x, w)